import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from constants import LABEL_TO_ID
from data.data_reader import get_required_config_path, load_config, load_metadata
from evaluation.metrics import ClassificationEvaluator

from models.vit.dataset import ChestXrayViTDataset
from models.vit.transforms import get_train_transforms, get_eval_transforms
from models.vit.model import build_vit_model

CONFIG_PATH = Path("config.yaml")


# ── helpers ──────────────────────────────────────────────

def get_vit_config(config: dict) -> dict:
    if "vit" not in config:
        raise ValueError("Missing 'vit' section in config.yaml")
    return config["vit"]


def compute_class_weights_tensor(metadata_df, device):
    """Same formula as data_analysis.compute_class_weights, but returns a tensor."""
    train_df = metadata_df[
        (metadata_df["split"] == "train") & (metadata_df["label"] != "unknown")
    ]
    counts = train_df["label"].value_counts()
    total = counts.sum()
    n_classes = len(LABEL_TO_ID)

    weights = torch.zeros(n_classes)
    for label_str, count in counts.items():
        weights[LABEL_TO_ID[label_str]] = total / (n_classes * count)
    return weights.to(device)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def filter_model_rows(df):
    """Keep only rows with known labels and existing files."""
    df = df[df["label"].isin(LABEL_TO_ID.keys())].copy()
    df = df[df["filepath"].notna()].copy()
    df["filepath"] = df["filepath"].astype(str)
    df = df[df["filepath"].map(lambda p: Path(p).exists())].copy()
    if len(df) == 0:
        raise ValueError("No usable rows after filtering.")
    return df


# ── training loop ────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    return running_loss / n, np.array(all_labels), np.array(all_preds)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(loader.dataset)
    return running_loss / n, np.array(all_labels), np.array(all_preds)


@torch.no_grad()
def evaluate_with_tta(model, loader, device):
    """Average logits over original + horizontal flip for a free boost at inference."""
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits_orig = model(imgs)
        logits_flip = model(torch.flip(imgs, dims=[3]))
        avg_logits = (logits_orig + logits_flip) / 2
        all_preds.extend(avg_logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# ── main ─────────────────────────────────────────────────

def main():
    config = load_config(CONFIG_PATH)
    vit_cfg = get_vit_config(config)
    evaluator = ClassificationEvaluator(model_name="vit")

    # Use preprocessed metadata (grayscale 128×128) for fair comparison with CNN and DualBranch
    metadata_path = get_required_config_path(config, CONFIG_PATH, "preprocessed_metadata_output_path")
    metadata_df = load_metadata(metadata_path)
    metadata_df = filter_model_rows(metadata_df)

    print(f"[ViT] Dataset splits:")
    print(metadata_df.groupby(["split", "label"]).size())

    device = get_device()
    print(f"[ViT] Device: {device}")

    image_size = vit_cfg.get("image_size", 224)
    batch_size = vit_cfg.get("batch_size", 32)
    num_workers = vit_cfg.get("num_workers", 2)
    epochs = vit_cfg.get("epochs", 20)

    # Datasets & loaders
    preserve_ar = vit_cfg.get("preserve_aspect_ratio", False)
    train_ds = ChestXrayViTDataset(metadata_df, "train", get_train_transforms(image_size, preserve_ar))
    val_ds   = ChestXrayViTDataset(metadata_df, "val",   get_eval_transforms(image_size, preserve_ar))
    test_ds  = ChestXrayViTDataset(metadata_df, "test",  get_eval_transforms(image_size, preserve_ar))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"[ViT] Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Model
    model = build_vit_model(
        model_name=vit_cfg["model_name"],
        pretrained=vit_cfg.get("pretrained", True),
        num_classes=vit_cfg.get("num_classes", 3),
        freeze_backbone=vit_cfg.get("freeze_backbone", False),
        dropout=vit_cfg.get("dropout", 0.1),
    ).to(device)

    # Loss with class weights + label smoothing
    if vit_cfg.get("use_class_weights", True):
        weights = compute_class_weights_tensor(metadata_df, device)
        print(f"[ViT] Class weights: {[f'{w:.3f}' for w in weights.tolist()]}")
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=vit_cfg.get("label_smoothing", 0.0),
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=vit_cfg.get("lr", 1e-4),
        weight_decay=vit_cfg.get("weight_decay", 0.01),
    )
    warmup_epochs = vit_cfg.get("warmup_epochs", 3)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    # Output dir
    output_dir = Path(vit_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    best_val_recall = -1.0
    history = []
    unfreeze_at = vit_cfg.get("unfreeze_at_epoch", None)

    from sklearn.metrics import recall_score, f1_score

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Phase-2 unfreezing
        if unfreeze_at and epoch == unfreeze_at and vit_cfg.get("freeze_backbone", False):
            for param in model.parameters():
                param.requires_grad = True
            backbone_lr = vit_cfg.get("lr", 1e-4) * 0.1
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=backbone_lr,
                weight_decay=vit_cfg.get("weight_decay", 0.01),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch
            )
            print(f"[ViT] Epoch {epoch}: unfroze all layers, LR → {backbone_lr:.2e}")

        train_loss, train_y, train_pred = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_y, val_pred = evaluate_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()

        train_recall = recall_score(train_y, train_pred, average="macro", zero_division=0)
        train_f1 = f1_score(train_y, train_pred, average="macro", zero_division=0)
        val_recall = recall_score(val_y, val_pred, average="macro", zero_division=0)
        val_f1 = f1_score(val_y, val_pred, average="macro", zero_division=0)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
            f"Train Loss={train_loss:.4f} Recall={train_recall:.4f} | "
            f"Val Loss={val_loss:.4f} Recall={val_recall:.4f} F1={val_f1:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_recall": train_recall, "train_f1": train_f1,
            "val_loss": val_loss, "val_recall": val_recall, "val_f1": val_f1,
        })

        if val_recall > best_val_recall:
            best_val_recall = val_recall
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  → Saved best model (val macro recall={val_recall:.4f})")

    # ── Final evaluation using ClassificationEvaluator ──
    # Load best model for test evaluation
    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))

    use_tta = vit_cfg.get("use_tta", False)
    if use_tta:
        print("[ViT] Running final evaluation with TTA (orig + h-flip)...")
        test_y, test_pred = evaluate_with_tta(model, test_loader, device)
        train_y_final, train_pred_final = evaluate_with_tta(model, train_loader, device)
        val_y_final, val_pred_final = evaluate_with_tta(model, val_loader, device)
    else:
        _, test_y, test_pred = evaluate_epoch(model, test_loader, criterion, device)
        _, train_y_final, train_pred_final = evaluate_epoch(model, train_loader, criterion, device)
        _, val_y_final, val_pred_final = evaluate_epoch(model, val_loader, criterion, device)

    # Use ClassificationEvaluator — same as logistic_regression.py
    metrics_by_split = {
        "train": evaluator.evaluate_split(train_y_final, train_pred_final, "train"),
        "val": evaluator.evaluate_split(val_y_final, val_pred_final, "val"),
        "test": evaluator.evaluate_split(test_y, test_pred, "test"),
    }

    extra_artifacts = {
        "training_history.json": history,
        "dataset_summary.json": {
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "labels": LABEL_TO_ID,
        },
    }

    run_dir = evaluator.save_run(
        base_output_dir=output_dir,
        config=vit_cfg,
        metrics_by_split=metrics_by_split,
        experiment_name=vit_cfg.get("experiment_name", "vit"),
        extra_artifacts=extra_artifacts,
    )

    # Save final model to run dir
    torch.save(model.state_dict(), run_dir / "best_model.pt")

    print(f"\n[INFO] Run saved to: {run_dir.resolve()}")
    print(f"[INFO] Best val macro recall: {best_val_recall:.4f}")


if __name__ == "__main__":
    main()