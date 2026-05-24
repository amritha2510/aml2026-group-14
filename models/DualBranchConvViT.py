import sys
import copy
import json
import os
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import timm
from PIL import Image
import pandas as pd
import numpy as np

from constants import LABEL_TO_ID
from data.data_reader import get_required_config_path, load_config, load_metadata
from evaluation.metrics import ClassificationEvaluator
from logistic_regression import filter_model_rows

from data.image_transforms import (
    get_image_aug_config,
    load_image_as_rgb_array,
    augment_rgb_array,
)
config_path = Path(os.environ.get("DUAL_BRANCH_CONFIG_PATH", "config_dual_branch.yaml"))

Image_net_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
Image_net_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PreprocessedChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, aug_config: dict | None = None, is_training: bool = False, random_state: int = 42, transform = None):
        self.filepaths = df["filepath"].tolist()
        self.labels = df["label"].map(LABEL_TO_ID).tolist()
        self.aug_config = aug_config
        self.is_training = is_training
        self.random_state = random_state

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        arr = load_image_as_rgb_array(str(path), normalize=True)
        pil_img = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
        pil_img = pil_img.resize((224, 224), Image.BILINEAR)
        arr = np.array(pil_img, dtype=np.float32) / 255.0  

        # Pre-trained ResNet/ViT strictly expects 3 color channels.
        # This converts our offline 1-channel grayscale safely back to 3 channels on the fly.
         # Apply augmentation (training only, when enabled in config)
        if (
            self.is_training
            and self.aug_config is not None
            and self.aug_config.get("enabled", False)
            and np.random.random() < self.aug_config.get("probability", 0.0)
        ):
            arr = augment_rgb_array(arr, self.aug_config, np.random.default_rng())
 
        # ImageNet normalization: (pixel - mean) / std  →  CxHxW tensor
        arr = (arr - Image_net_mean) / Image_net_std          # HxWx3
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()   # 3xHxW
 
        return tensor, self.labels[idx]

class AttentionFusionBlock(nn.Module):
    """
    We have to have the CNN and ViT features to the same dimension, treats them as a sequence of 2 tokens
    and uses Multihead Self-Attention to let them align and weigh each other dynamically before classification.
    """
    def __init__(self, conn_dim = 512, vit_dim = 192, embed_dim = 256):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos encoding"
        self.embed_dim = embed_dim
        self.cnn_proj  = nn.Linear(conn_dim, embed_dim)
        self.vit_proj  = nn.Linear(vit_dim,  embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.output_dim = embed_dim

    @staticmethod
    def _get_2d_sincos_pos_embed(H: int, W: int, embed_dim: int) -> torch.Tensor:
        """Generates (1, H*W, embed_dim) 2D sinusoidal positional encodings."""
        d = embed_dim // 2  # dimensions per spatial axis

        # Frequency terms shared by both axes
        omega = 1.0 / (10000.0 ** (torch.arange(0, d, 2, dtype=torch.float32) / d))

        # Height encoding — (H, d)
        h_pos = torch.arange(H, dtype=torch.float32).unsqueeze(1)
        h_enc = torch.zeros(H, d)
        h_enc[:, 0::2] = torch.sin(h_pos * omega)
        h_enc[:, 1::2] = torch.cos(h_pos * omega)

        # Width encoding — (W, d)
        w_pos = torch.arange(W, dtype=torch.float32).unsqueeze(1)
        w_enc = torch.zeros(W, d)
        w_enc[:, 0::2] = torch.sin(w_pos * omega)
        w_enc[:, 1::2] = torch.cos(w_pos * omega)

        # Broadcast to (H, W, d) per axis then cat → (H, W, embed_dim)
        h_grid = h_enc.unsqueeze(1).expand(H, W, d)
        w_grid = w_enc.unsqueeze(0).expand(H, W, d)
        pos_embed = torch.cat([h_grid, w_grid], dim=-1)  # (H, W, embed_dim)

        return pos_embed.reshape(H * W, embed_dim).unsqueeze(0)  # (1, H*W, embed_dim)

    def forward(self, local_feats, global_feats, H: int, W: int):
        cnn_tokens = self.cnn_proj(local_feats)                                   # (B, H*W, embed_dim)
        pos_embed  = self._get_2d_sincos_pos_embed(H, W, self.embed_dim).to(cnn_tokens.device)
        cnn_tokens = cnn_tokens + pos_embed                                       # inject spatial context

        vit_tokens = self.vit_proj(global_feats)                                  # (B, SeqLen_ViT, embed_dim)

        seq          = torch.cat([cnn_tokens, vit_tokens], dim=1)                 # (B, SeqLen_CNN+SeqLen_ViT, embed_dim)
        attended_seq = self.transformer(seq)
        return attended_seq.mean(dim=1)                                           # (B, embed_dim)

class DualBranchConvViT(nn.Module):
    def __init__(self, num_classes = 3, noise_dropout_rates = 0.4, fusion_type = "concat"):
        super().__init__()
        self.fusion_type = fusion_type

        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove avgpool AND fc → outputs (B, 512, 7, 7) for 224×224 input
        self.cnn_branch = nn.Sequential(*list(resnet18.children())[:-2])

        self.vit_branch = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit_branch.reset_classifier(0)

        dummy_tensor = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _cnn_map = self.cnn_branch(dummy_tensor)                              # (1, C, H, W)
            cnn_dim  = _cnn_map.flatten(2).permute(0, 2, 1).shape[-1]            # feature dim C
            vit_dim  = self.vit_branch.forward_features(dummy_tensor).shape[-1]  # feature dim

        print(f"[DEBUG] Auto-detected CNN dim: {cnn_dim}, ViT dim: {vit_dim}")

        self.cnn_dropout = nn.Dropout(p=noise_dropout_rates)
        self.vit_dropout = nn.Dropout(p=noise_dropout_rates)
        
        
        if self.fusion_type == "attention":
            self.fusion_block = AttentionFusionBlock(cnn_dim, vit_dim, embed_dim = 256)
            classifier_dimension = self.fusion_block.output_dim
        else:
            classifier_dimension = cnn_dim + vit_dim

        self.classifier = nn.Linear(classifier_dimension, num_classes)

    def forward(self, x):
        feat_map    = self.cnn_branch(x)                           # (B, C, H, W)
        H, W        = feat_map.shape[-2], feat_map.shape[-1]       # capture before flattening
        local_feats = feat_map.flatten(2).permute(0, 2, 1)        # (B, H*W, C)
        local_feats = self.cnn_dropout(local_feats)

        global_feats = self.vit_branch.forward_features(x)        # (B, SeqLen, D)
        global_feats = self.vit_dropout(global_feats)

        if self.fusion_type == "attention":
            fused_feats = self.fusion_block(local_feats, global_feats, H, W)
        else:
            # Mean-pool spatial sequences before concatenating so the
            # classifier receives a flat (B, cnn_dim + vit_dim) vector.
            fused_feats = torch.cat(
                (local_feats.mean(1), global_feats.mean(1)), dim=1
            )

        return self.classifier(fused_feats)

def set_backbone_trainable(model: DualBranchConvViT, is_trainable: bool) -> None:
    for param in model.cnn_branch.parameters():
        param.requires_grad = is_trainable
    for param in model.vit_branch.parameters():
        param.requires_grad = is_trainable


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed + worker_id)


def main():
    config = load_config(config_path)
    dl_cfg = config.get("deep_learning", {})
    if not dl_cfg:
        raise Exception("No Deep Learning configuration")

    output_dir = get_required_config_path(dl_cfg, config_path, "output_dir")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    preprocessed_path = get_required_config_path(config, config_path, "preprocessed_metadata_output_path")
    df = filter_model_rows(load_metadata(preprocessed_path))

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"[INFO] Dataset -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Note: ViT strictly expects 224x224. ImageNet normalization is required.
    aug_config = get_image_aug_config(config, model_key="Fusion_model_cnn_vit")
    random_state = dl_cfg.get("random_state", 42)
    batch_size = dl_cfg.get("batch_size", 32)
    train_loader = DataLoader(PreprocessedChestXrayDataset(train_df, aug_config=aug_config, is_training=True, random_state=random_state),  batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True, worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(PreprocessedChestXrayDataset(val_df, aug_config=aug_config, is_training=False, random_state=random_state),  batch_size=batch_size, shuffle=False,  num_workers=4, pin_memory=True)
    test_loader = DataLoader(PreprocessedChestXrayDataset(test_df, aug_config=aug_config, is_training=False, random_state=random_state),  batch_size=batch_size, shuffle=False,  num_workers=4, pin_memory=True)

    fusion_strategy = dl_cfg.get("fusion_type", "concat")
    print(f"[INFO] Initializing Dual-Branch Model with '{fusion_strategy.upper()}' fusion...")

    model = DualBranchConvViT(
        num_classes = 3,
        noise_dropout_rates = dl_cfg.get("noise_dropout_rate", 0.4),
        fusion_type = fusion_strategy
    ).to(device)

    freeze_epochs = dl_cfg.get("freeze_epochs", 3)
    set_backbone_trainable(model, False)
    print(f"[INFO] Backbones frozen for first {freeze_epochs} epochs — fusion head warming up.")

    class_weight = torch.tensor(dl_cfg.get("class_weight", [2.0, 2.0, 1.0]), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = float(dl_cfg.get("learning_rate", 1e-4)),
        weight_decay = float(dl_cfg.get("weight_decay", 1e-4)),
    )

    epochs = dl_cfg.get("epochs", 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr           = dl_cfg.get("learning_rate", 1e-4),
        steps_per_epoch  = len(train_loader),
        epochs           = epochs,
        pct_start        = 0.1,   # 10% of steps used for linear warmup
        anneal_strategy  = "cos",
    )

    evaluator = ClassificationEvaluator(model_name = f"DualBranchConvViT_{fusion_strategy}")
    best_val_recall = -1.0
    best_val_metrics = None

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    SOUP_K = 5
    soup_buffer: list[dict] = []
    epoch_val_recalls: list[float] = []
    epoch_history: list[dict] = []

    for epoch in range(epochs):
        if epoch == freeze_epochs:
            set_backbone_trainable(model, True)
            print(f"[INFO] Epoch {epoch+1}: Backbones unfrozen — full fine-tuning begins.")

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
                all_preds.append(preds.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

        val_metrics = evaluator.compute_metrics(
            np.concatenate(all_labels).astype(int),
            np.concatenate(all_preds).astype(int),
        )
        current_lr   = scheduler.get_last_lr()[0]
 
        print(
            f"Epoch [{epoch+1}/{epochs}]  "
            f"Loss: {running_loss/len(train_loader):.4f}  |  "
            f"Val Recall: {val_metrics['macro_recall']:.4f}  |  "
            f"Val F1: {val_metrics['macro_f1']:.4f}  |  "
            f"LR: {current_lr:.2e}"
        )
        
        epoch_val_recalls.append(val_metrics['macro_recall'])
        epoch_history.append({
            "epoch":            epoch + 1,
            "train_loss":       round(running_loss / len(train_loader), 4),
            "val_macro_recall": round(val_metrics["macro_recall"], 4),
            "val_macro_f1":     round(val_metrics["macro_f1"], 4),
            "lr":               round(current_lr, 8),
            "backbones_frozen": epoch < freeze_epochs,
        })

        soup_buffer.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        if len(soup_buffer) > SOUP_K:
            soup_buffer.pop(0)

        if val_metrics['macro_recall'] > best_val_recall:
            best_val_recall = val_metrics['macro_recall']
            best_val_metrics = copy.deepcopy(val_metrics)

    avg_val_recall = float(np.mean(epoch_val_recalls))
    print(f"\n[INFO] Val recall — best: {best_val_recall:.4f}  avg across epochs: {avg_val_recall:.4f}")

    print(f"[INFO] Averaging last {len(soup_buffer)} checkpoints (model soup)…")
    avg_state = {
        k: torch.stack([ckpt[k].float() for ckpt in soup_buffer]).mean(0)
        for k in soup_buffer[0]
    }
    model.load_state_dict(avg_state)
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_preds.append(preds.detach().cpu().numpy())
            test_labels.append(labels.detach().cpu().numpy())

    test_preds_flat  = np.concatenate(test_preds).astype(int)
    test_labels_flat = np.concatenate(test_labels).astype(int)

    final_metrics = {
        "val": {**best_val_metrics, "avg_macro_recall": avg_val_recall},
        "test": evaluator.evaluate_split(test_labels_flat, test_preds_flat, "test"),
    }

    run_dir = evaluator.save_run(
        base_output_dir = output_dir,
        config = dl_cfg,
        metrics_by_split=final_metrics,
        experiment_name=f"{dl_cfg.get('experiment_name', 'deep_learning')}_{fusion_strategy}"
    )

    if os.environ.get("DUAL_BRANCH_SAVE_WEIGHTS", "1") == "1":
        torch.save(avg_state, run_dir / "best_model_weights.pth")

    test_report = final_metrics["test"].get("classification_report", {})
    model_card = {
        "model": {
            "name":               "DualBranchConvViT",
            "cnn_backbone":       "resnet18",
            "vit_backbone":       "vit_base_patch16_224",
            "fusion_type":        fusion_strategy,
            "num_classes":        3,
            "class_names":        list(LABEL_TO_ID.keys()),
            "noise_dropout_rate": dl_cfg.get("noise_dropout_rate", 0.4),
        },
        "training": {
            "optimizer":              "AdamW",
            "scheduler":              "OneCycleLR (pct_start=0.1, anneal=cos)",
            "learning_rate":          float(dl_cfg.get("learning_rate", 1e-4)),
            "weight_decay":           float(dl_cfg.get("weight_decay", 1e-4)),
            "batch_size":             batch_size,
            "epochs":                 epochs,
            "freeze_epochs":          freeze_epochs,
            "class_weights":          dl_cfg.get("class_weight", []),
            "viral_boost_multiplier": dl_cfg.get("viral_boost_multiplier", 1.0),
            "model_soup_k":           SOUP_K,
            "augmentation":           aug_config or {},
        },
        "data": {
            "train_size":   len(train_df),
            "val_size":     len(val_df),
            "test_size":    len(test_df),
            "image_size":   [224, 224],
            "preprocessing": "RGB 224×224 native, ImageNet normalisation",
        },
        "history": epoch_history,
        "results": {
            "val": {
                "best_macro_recall": round(best_val_recall, 4),
                "avg_macro_recall":  round(avg_val_recall, 4),
                "best_macro_f1":     round(best_val_metrics.get("macro_f1", -1), 4),
            },
            "test": {
                "macro_recall": round(final_metrics["test"].get("macro_recall", -1), 4),
                "macro_f1":     round(final_metrics["test"].get("macro_f1", -1), 4),
                "per_class_recall": {
                    cls: round(test_report.get(cls, {}).get("recall", -1), 4)
                    for cls in ("normal", "bacterial", "viral")
                },
                "per_class_f1": {
                    cls: round(test_report.get(cls, {}).get("f1-score", -1), 4)
                    for cls in ("normal", "bacterial", "viral")
                },
            },
        },
        "weights": {
            "filename":    "best_model_weights.pth",
            "description": f"Model soup: average of last {SOUP_K} epoch checkpoints",
        },
        "environment": {
            "torch_version":  torch.__version__,
            "timm_version":   timm.__version__,
            "python_version": sys.version.split()[0],
            "device":         str(device),
        },
        "timestamp":       datetime.now().isoformat(),
        "experiment_name": dl_cfg.get("experiment_name", "deep_learning"),
    }

    with open(run_dir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)

    print(f"\n[INFO] Run successfully saved to: {run_dir.resolve()}")

if __name__ == "__main__":
    main()






