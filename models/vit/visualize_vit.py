import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from data.data_reader import load_config, get_required_config_path, load_metadata
from models.vit.dataset import ChestXrayViTDataset
from models.vit.transforms import get_eval_transforms
from models.vit.model import build_vit_model
from models.vit.train_vit import filter_model_rows, get_device


@torch.no_grad()
def evaluate_with_tta(model, loader, device):
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

CONFIG_PATH = Path("config.yaml")
CLASS_NAMES = ["Normal", "Bacterial", "Viral"]


def load_model_and_preds():
    config = load_config(CONFIG_PATH)
    vit_cfg = config["vit"]

    metadata_path = get_required_config_path(config, CONFIG_PATH, "preprocessed_metadata_output_path")
    metadata_df = filter_model_rows(load_metadata(metadata_path))

    device = get_device()
    image_size = vit_cfg.get("image_size", 224)

    test_ds = ChestXrayViTDataset(metadata_df, "test", get_eval_transforms(image_size, False))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    model = build_vit_model(
        model_name=vit_cfg["model_name"],
        pretrained=False,
        num_classes=3,
        freeze_backbone=False,
        dropout=vit_cfg.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(torch.load("outputs/vit/best_model.pt", weights_only=True))

    test_y, test_pred = evaluate_with_tta(model, test_loader, device)
    return test_y, test_pred


def plot_confusion_matrix(test_y, test_pred, out_dir):
    cm = confusion_matrix(test_y, test_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_pct, annot=False, fmt=".1f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5, linecolor="gray", vmin=0, vmax=100
    )
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j + 0.5, i + 0.5,
                    f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm_pct[i, j] > 60 else "black")

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("ViT — Test Set Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: confusion_matrix.png")


def plot_per_class_metrics(test_y, test_pred, out_dir):
    precision = precision_score(test_y, test_pred, average=None, zero_division=0)
    recall = recall_score(test_y, test_pred, average=None, zero_division=0)
    f1 = f1_score(test_y, test_pred, average=None, zero_division=0)

    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (vals, label) in enumerate(zip([precision, recall, f1], ["Precision", "Recall", "F1"])):
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=colors[i])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    macro_f1 = f1_score(test_y, test_pred, average="macro", zero_division=0)
    ax.axhline(y=macro_f1, color="gray", linestyle="--", alpha=0.7)
    ax.text(2.4, macro_f1 + 0.01, f"Macro F1={macro_f1:.2f}", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("ViT — Per-Class Metrics (Test Set)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "per_class_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: per_class_metrics.png")


def plot_training_history(out_dir):
    history_files = sorted(Path("outputs/vit/experiments").rglob("training_history.json"))
    if not history_files:
        print("No training history found, skipping.")
        return

    with open(history_files[-1]) as f:
        history = json.load(f)

    epochs      = [h["epoch"] for h in history]
    train_loss  = [h["train_loss"] for h in history]
    train_f1    = [h["train_f1"] for h in history]
    val_f1      = [h["val_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss, "b-o", markersize=4)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss", fontsize=13, fontweight="bold")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_f1, "b-o", markersize=4, label="Train F1")
    ax2.plot(epochs, val_f1, "r-o", markersize=4, label="Val F1")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Macro F1", fontsize=12)
    ax2.set_title("Train vs Val Macro F1", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)

    plt.suptitle("ViT Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: training_history.png")


def plot_experiment_comparison(out_dir):
    # 224px preprocessed RGB configs only
    experiments = [
        {"label": "lr=5e-5, wd=0.01\ndropout=0.2, no TTA", "macro_f1": 0.7943, "normal": 0.6700, "bacterial": 0.8300, "viral": 0.9500},
        {"label": "lr=5e-5, wd=0.01\ndropout=0.2, TTA",    "macro_f1": 0.8329, "normal": 0.7436, "bacterial": 0.8843, "viral": 0.9189},
        {"label": "lr=5e-5, wd=0.001\ndropout=0.1, TTA",   "macro_f1": 0.8497, "normal": 0.8205, "bacterial": 0.9752, "viral": 0.7432},
    ]

    labels = [e["label"] for e in experiments]
    macro_f1s = [e["macro_f1"] for e in experiments]
    normals = [e["normal"] for e in experiments]
    bacterials = [e["bacterial"] for e in experiments]
    virals = [e["viral"] for e in experiments]

    x = np.arange(len(experiments))
    width = 0.22
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars = ax1.bar(x, macro_f1s, color="#4C72B0", width=0.5, edgecolor="white")
    for bar, val in zip(bars, macro_f1s):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0.70, 0.88)
    ax1.set_ylabel("Macro F1", fontsize=12)
    ax1.set_title("Macro F1 — 224px RGB Configs", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    for i, (vals, cls_label) in enumerate(zip([normals, bacterials, virals], ["Normal", "Bacterial", "Viral"])):
        bars2 = ax2.bar(x + (i - 1) * width, vals, width, label=cls_label, color=colors[i])
        for bar in bars2:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylim(0.55, 1.05)
    ax2.set_ylabel("Recall", fontsize=12)
    ax2.set_title("Per-Class Recall — 224px RGB Configs", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("ViT — 224px Preprocessed RGB Config Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "experiment_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: experiment_comparison.png")


def main():
    out_dir = Path("outputs/vit/visualizations")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model and running inference...")
    test_y, test_pred = load_model_and_preds()

    plot_confusion_matrix(test_y, test_pred, out_dir)
    plot_per_class_metrics(test_y, test_pred, out_dir)
    plot_training_history(out_dir)
    plot_experiment_comparison(out_dir)

    print(f"\nAll visualizations saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
