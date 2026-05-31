import os
import sys
import copy
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
import timm
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from constants import LABEL_TO_ID
from data.image_transforms import load_image_as_rgb_array, augment_rgb_array, make_deterministic_image_seed, get_image_aug_config
from data.data_reader import get_required_config_path, load_config, load_metadata
from data.data_analysis import compute_class_weights
from evaluation.metrics import ClassificationEvaluator

config_path = Path(os.environ.get("DUAL_BRANCH_CONFIG_PATH", "config.yaml"))


class PreprocessedChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, aug_cfg=None, is_training=False, random_state=42):
        self.filepaths = df["filepath"].tolist()
        self.labels = df["label"].map(LABEL_TO_ID).tolist()
        self.transform = transform
        self.aug_cfg = aug_cfg
        self.is_training = is_training
        self.random_state = random_state

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        arr = load_image_as_rgb_array(self.filepaths[idx], normalize=True)

        if self.is_training and self.aug_cfg and self.aug_cfg.get("enabled", False):
            rng = np.random.default_rng(make_deterministic_image_seed(idx, self.random_state))
            if rng.random() < self.aug_cfg.get("probability", 0.0):
                arr = augment_rgb_array(arr, self.aug_cfg, rng)

        image = Image.fromarray((arr * 255).astype(np.uint8))
        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


class CrossAttentionFusion(nn.Module):
    """
    Real cross-attention between the two branches.

    The ViT tokens are the GLOBAL CANVAS: they describe the overall chest /
    lung structure. They are used as queries. The CNN's spatial feature-map
    tokens carry LOCAL TEXTURE (the fine-grained opacity patterns). They are
    the keys/values. Each global token therefore *pulls in* the local detail
    that is relevant to it — this is the "paint local texture onto the global
    canvas" idea, now done over real spatial positions instead of two pooled
    vectors. We then mean-pool over the (locally-informed) ViT token axis to
    get a single descriptor for the classifier.
    """

    def __init__(self, embed_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.output_dim = embed_dim

    def forward(self, vit_tokens, cnn_tokens):
        # vit_tokens: (B, Tv, D) queries  |  cnn_tokens: (B, Tc, D) keys/values
        attn_out, _ = self.cross_attn(vit_tokens, cnn_tokens, cnn_tokens)
        x = self.norm1(vit_tokens + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x.mean(dim=1)  # (B, D)


class DualBranchConvViT(nn.Module):
    def __init__(self, num_classes=3, noise_dropout_rates=0.4, fusion_type="concat",
                 embed_dim=256, num_heads=4):
        super().__init__()
        self.fusion_type = fusion_type

        # --- CNN branch: keep the full spatial feature map (NO global pooling) ---
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.cnn_features = densenet.features  # -> (B, C, H, W), e.g. (B, 1024, 7, 7)

        # --- ViT branch: keep the full token sequence (CLS + patch tokens) ---
        self.vit_branch = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)

        # Auto-detect dims + CNN token count from a dummy forward pass.
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            cnn_feat = self.cnn_features(dummy)                  # (1, C, H, W)
            cnn_dim = cnn_feat.shape[1]
            num_cnn_tokens = cnn_feat.shape[2] * cnn_feat.shape[3]
            vit_tok = self.vit_branch.forward_features(dummy)    # (1, Tv, vit_dim)
            vit_dim = vit_tok.shape[-1]
            num_vit_tokens = vit_tok.shape[1]
        print(f"[DEBUG] CNN dim={cnn_dim}, CNN tokens={num_cnn_tokens} | "
              f"ViT dim={vit_dim}, ViT tokens={num_vit_tokens}")

        # Project both branches into a shared embedding space.
        self.cnn_proj = nn.Linear(cnn_dim, embed_dim)
        self.vit_proj = nn.Linear(vit_dim, embed_dim)

        # Learnable positional embedding for the CNN spatial tokens so the
        # fusion knows *where* each local texture lives.
        self.cnn_pos = nn.Parameter(torch.zeros(1, num_cnn_tokens, embed_dim))
        nn.init.trunc_normal_(self.cnn_pos, std=0.02)

        # Embedding-level noise on the ViT branch before fusion (heavy dropout
        # regularizer — forces reliance on both branches).
        self.noise_injection = nn.Dropout(p=noise_dropout_rates)

        if fusion_type == "attention":
            self.fusion_block = CrossAttentionFusion(
                embed_dim=embed_dim, num_heads=num_heads, dropout=0.1
            )
            classifier_dim = self.fusion_block.output_dim       # embed_dim
        elif fusion_type == "concat":
            classifier_dim = embed_dim * 2                      # ViT CLS + pooled CNN
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.classifier = nn.Linear(classifier_dim, num_classes)

    def forward(self, x):
        # --- CNN local spatial tokens ---
        feat = F.relu(self.cnn_features(x))                     # (B, C, H, W)
        cnn_tokens = feat.flatten(2).transpose(1, 2)           # (B, H*W, C)
        cnn_tokens = self.cnn_proj(cnn_tokens) + self.cnn_pos  # (B, H*W, D)

        # --- ViT global tokens (CLS + patches) ---
        vit_tokens = self.vit_branch.forward_features(x)       # (B, Tv, vit_dim)
        vit_tokens = self.vit_proj(vit_tokens)                 # (B, Tv, D)
        vit_tokens = self.noise_injection(vit_tokens)          # embedding-level noise

        if self.fusion_type == "attention":
            fused = self.fusion_block(vit_tokens, cnn_tokens)  # (B, D)
        else:  # concat baseline: ViT CLS token + mean-pooled CNN map
            vit_cls = vit_tokens[:, 0]                         # (B, D)
            cnn_pooled = cnn_tokens.mean(dim=1)                # (B, D)
            fused = torch.cat([vit_cls, cnn_pooled], dim=1)    # (B, 2D)

        return self.classifier(fused)


def resolve_freeze_epochs(dl_cfg: dict, epochs: int) -> int:
    """
    Freeze schedule that is *consistent across run lengths*.

    Prefer `freeze_frac` (fraction of total epochs) so the short search runs
    and the long final runs both go through the same freeze->unfreeze
    transition. A fixed `freeze_epochs` is still honoured for back-compat, but
    is clamped to `epochs - 1` so the unfreeze ALWAYS fires (the old default of
    5 with a 5-epoch search meant the backbones never unfroze during search).
    """
    if "freeze_frac" in dl_cfg:
        freeze = max(1, round(dl_cfg["freeze_frac"] * epochs))
    else:
        freeze = dl_cfg.get("freeze_epochs", max(1, round(0.3 * epochs)))
    return max(0, min(freeze, epochs - 1))


@torch.no_grad()
def tta_predict(model, images, use_amp):
    """
    Test-time augmentation. Averages class probabilities over a small set of
    mild geometric views drawn from the same distribution as training
    augmentation (rotation up to ~10deg). No horizontal flip — chest X-rays are
    laterally asymmetric (cardiac silhouette, gastric bubble), and the training
    aug config uses rotation/translation, not flips. Returns averaged probs.
    """
    views = [images, TF.rotate(images, 7), TF.rotate(images, -7)]
    probs = None
    for v in views:
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(v)
        p = torch.softmax(out.float(), dim=1)
        probs = p if probs is None else probs + p
    return probs / len(views)


def main():
    config = load_config(config_path)
    dl_cfg = config.get("dual_conv_vit", {})
    if not dl_cfg:
        raise Exception("No dual_conv_vit configuration")

    output_dir = get_required_config_path(dl_cfg, config_path, "output_dir")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    preprocessed_path = get_required_config_path(config, config_path, "preprocessed_metadata_output_path")
    df = load_metadata(preprocessed_path)
    # Filter for known labels, non-missing paths, and successful preprocessing
    mask = df["label"].isin(LABEL_TO_ID.keys()) & df["filepath"].notna()
    if "is_preprocessed" in df.columns:
        mask &= (df["is_preprocessed"] == True)
    df = df[mask].copy()

    df["filepath"] = df["filepath"].astype(str)
    df = df[df["filepath"].map(lambda p: Path(p).exists())].copy()

    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()

    aug_config = get_image_aug_config(config, model_key="dual_conv_vit")
    random_state = dl_cfg.get("random_state", 42)
    batch_size = dl_cfg.get("batch_size", 32)

    print(f"[INFO] Splits -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    num_workers = dl_cfg.get("num_workers", 4)
    train_loader = DataLoader(PreprocessedChestXrayDataset(train_df, transform=transform, aug_cfg=aug_config, is_training=True, random_state=random_state), batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(PreprocessedChestXrayDataset(val_df,   transform=transform), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(PreprocessedChestXrayDataset(test_df,  transform=transform), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Class weights computed from the actual training fold (not the full train set).
    weights_dict = compute_class_weights(train_df)
    class_weight = torch.zeros(len(LABEL_TO_ID)).to(device)
    for label_str, weight in weights_dict.items():
        class_weight[LABEL_TO_ID[label_str]] = weight

    epochs = dl_cfg.get("epochs", 10)
    fusion_types = dl_cfg.get("fusion_types") or [dl_cfg.get("fusion_type", "concat")]

    for fusion_strategy in fusion_types:
        print(f"\n[INFO] ── Fusion: {fusion_strategy.upper()} ──")

        # Extract fusion-specific overrides if they exist
        fusion_cfg = dl_cfg.get(fusion_strategy, {})
        current_lr = fusion_cfg.get("learning_rate", dl_cfg.get("learning_rate", 1e-4))
        current_dropout = fusion_cfg.get("noise_dropout_rates", dl_cfg.get("noise_dropout_rates", 0.4))
        current_wd = fusion_cfg.get("weight_decay", dl_cfg.get("weight_decay", 1e-4))

        model = DualBranchConvViT(
            num_classes=3,
            noise_dropout_rates=current_dropout,
            fusion_type=fusion_strategy,
        ).to(device)

        freeze_epochs = resolve_freeze_epochs(dl_cfg, epochs)
        cnn_backbone_params = list(model.cnn_features.parameters())
        vit_backbone_params = list(model.vit_branch.parameters())
        backbone_params = cnn_backbone_params + vit_backbone_params
        head_params = [p for p in model.parameters() if not any(p is bp for bp in backbone_params)]

        for p in backbone_params:
            p.requires_grad = False
        print(f"[INFO] Backbones frozen for first {freeze_epochs}/{epochs} epochs.")

        criterion = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=dl_cfg.get("label_smoothing", 0.0))
        optimizer = torch.optim.AdamW(head_params, lr=current_lr, weight_decay=current_wd)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=current_lr,
            steps_per_epoch=len(train_loader),
            epochs=max(1, freeze_epochs) if freeze_epochs > 0 else epochs,
            pct_start=0.1,
            anneal_strategy="cos",
        )

        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        evaluator = ClassificationEvaluator(model_name=f"DualBranchConvViT_{fusion_strategy}")
        best_val_recall = -1.0
        best_val_metrics = None
        best_weights = None
        val_recalls = []  # track every epoch so we can report a stable averaged metric

        for epoch in range(epochs):
            if epoch == freeze_epochs and freeze_epochs > 0:
                for p in backbone_params:
                    p.requires_grad = True
                main_lr = current_lr
                # The ViT branch fine-tunes ~5x harder than the CNN branch so it
                # reaches the strength it has standalone. The baseline ViT (the
                # SAME vit_tiny backbone) trains at ~5e-5, which is main_lr*0.5
                # when main_lr=1e-4. The CNN branch already dominates on
                # bacterial, so it stays gently fine-tuned.
                vit_bb_lr = main_lr * dl_cfg.get("vit_finetune_mult", 0.5)
                cnn_bb_lr = main_lr * dl_cfg.get("cnn_finetune_mult", 0.1)
                optimizer = torch.optim.AdamW([
                    {"params": cnn_backbone_params, "lr": cnn_bb_lr},
                    {"params": vit_backbone_params, "lr": vit_bb_lr},
                    {"params": head_params,         "lr": main_lr},
                ], weight_decay=current_wd)
                remaining = epochs - freeze_epochs
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=[cnn_bb_lr, vit_bb_lr, main_lr],
                    steps_per_epoch=len(train_loader),
                    epochs=remaining,
                    pct_start=0.1,
                    anneal_strategy="cos",
                )
                print(f"[INFO] Epoch {epoch+1}: backbones unfrozen "
                      f"(cnn_lr={cnn_bb_lr:.0e}, vit_lr={vit_bb_lr:.0e})")

            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running_loss += loss.item()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

            val_metrics = evaluator.compute_metrics(all_labels, all_preds)
            val_recalls.append(val_metrics["macro_recall"])
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} | Val Recall: {val_metrics['macro_recall']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}")

            if val_metrics["macro_recall"] > best_val_recall:
                best_val_recall = val_metrics["macro_recall"]
                best_val_metrics = copy.deepcopy(val_metrics)
                best_weights = copy.deepcopy(model.state_dict())

        # Attach averaged val metrics so the sweep can rank configs on something
        # more stable than a single lucky epoch (the val set is tiny).
        if best_val_metrics is not None:
            best_val_metrics["avg_macro_recall"] = float(np.mean(val_recalls))
            best_val_metrics["avg_macro_recall_last3"] = float(np.mean(val_recalls[-3:]))

        print(f"\n[INFO] Evaluating {fusion_strategy.upper()} on test split...")
        if best_weights is None:
            raise RuntimeError("Training finished without capturing best model weights.")
        model.load_state_dict(best_weights)
        model.eval()

        # TTA matches the baseline protocol (aligned to config).
        save_weights_env = os.environ.get("DUAL_BRANCH_SAVE_WEIGHTS")
        is_reportable = save_weights_env != "0"
        use_tta = bool(dl_cfg.get("use_tta", False)) and is_reportable
        print(f"[INFO] TTA at evaluation: {use_tta}")

        test_preds, test_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                if use_tta:
                    preds = torch.max(tta_predict(model, images, use_amp), 1)[1]
                else:
                    preds = torch.max(model(images), 1)[1]
                test_preds.extend(preds.cpu().tolist())
                test_labels.extend(labels.cpu().tolist())

        final_metrics = {
            "val":  best_val_metrics,
            "test": evaluator.evaluate_split(test_labels, test_preds, "test"),
        }

        run_dir = evaluator.save_run(
            base_output_dir=output_dir,
            config=dl_cfg,
            metrics_by_split=final_metrics,
            experiment_name=f"{dl_cfg.get('experiment_name', 'dual_conv_vit')}_{fusion_strategy}",
        )

        torch.save(best_weights, run_dir / "best_model_weights.pth")
        print(f"[INFO] {fusion_strategy.upper()} run saved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()