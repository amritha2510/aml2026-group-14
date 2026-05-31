import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BEST_RUN = Path("outputs/vit/experiments/vit_224_rgb_tuned_20260531_134459")
OUT_DIR  = Path("outputs/vit/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ── 1. Training curves ────────────────────────────────────────────────────────

def plot_training_curves():
    history = load_json(BEST_RUN / "training_history.json")

    epochs       = [h["epoch"]        for h in history]
    train_loss   = [h["train_loss"]   for h in history]
    train_recall = [h["train_recall"] for h in history]
    val_recall   = [h["val_recall"]   for h in history]
    val_loss     = [h["val_loss"]     for h in history]

    best_epoch = max(history, key=lambda h: h["val_recall"])["epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss", color="steelblue")
    ax1.plot(epochs, val_loss,   label="Val Loss",   color="tomato", linestyle="--")
    ax1.axvline(best_epoch, color="gray", linestyle=":", linewidth=1, label=f"Best epoch ({best_epoch})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Recall
    ax2.plot(epochs, train_recall, label="Train Macro Recall", color="steelblue")
    ax2.plot(epochs, val_recall,   label="Val Macro Recall",   color="tomato", linestyle="--")
    ax2.axvline(best_epoch, color="gray", linestyle=":", linewidth=1, label=f"Best epoch ({best_epoch})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro Recall")
    ax2.set_title("Training & Validation Macro Recall")
    ax2.legend()
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("ViT Training Curves (best config: lr=5e-5, wd=0.001, dropout=0.1, TTA)", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── 2. Confusion matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix():
    metrics  = load_json(BEST_RUN / "metrics.json")
    cm       = np.array(metrics["test"]["confusion_matrix"])
    classes  = ["Normal", "Bacterial", "Viral"]

    cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set\n(ViT best config, TTA)")

    thresh = 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i,j] > thresh else "black")

    fig.tight_layout()
    out = OUT_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── 3. Per-class recall comparison (ViT configs + CNN) ───────────────────────

def plot_experiment_comparison():
    experiments = [
        {
            "label": "ViT\n(224px, no TTA)\nwd=0.01, d=0.2",
            "normal": 0.6700, "bacterial": 0.8300, "viral": 0.9500,
            "macro_recall": 0.8158, "macro_f1": 0.7943,
        },
        {
            "label": "ViT\n(224px, +TTA)\nwd=0.01, d=0.2",
            "normal": 0.7436, "bacterial": 0.8843, "viral": 0.9189,
            "macro_recall": 0.8489, "macro_f1": 0.8329,
        },
        {
            "label": "ViT\n(224px, tuned+TTA)\nwd=0.001, d=0.1",
            "normal": 0.8205, "bacterial": 0.9752, "viral": 0.7432,
            "macro_recall": 0.8463, "macro_f1": 0.8497,
        },
        {
            "label": "CNN\nDenseNet121\n(224px)",
            "normal": 0.7479, "bacterial": 0.9256, "viral": 0.7703,
            "macro_recall": 0.8146, "macro_f1": 0.8089,
        },
    ]

    classes = ["Normal", "Bacterial", "Viral"]
    colors  = ["#4C72B0", "#DD8452", "#55A868"]
    x       = np.arange(len(experiments))
    width   = 0.22

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Per-class recall
    for i, (cls, col) in enumerate(zip(classes, colors)):
        vals = [e[cls.lower()] for e in experiments]
        bars = ax1.bar(x + (i - 1) * width, vals, width, label=cls, color=col)
        for bar, v in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([e["label"] for e in experiments], fontsize=8.5)
    ax1.set_ylabel("Recall")
    ax1.set_ylim(0, 1.08)
    ax1.set_title("Per-Class Test Recall by Experiment")
    ax1.legend()
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.7)

    # Macro recall + F1
    recall_vals = [e["macro_recall"] for e in experiments]
    f1_vals     = [e["macro_f1"]     for e in experiments]
    bars_r = ax2.bar(x - width / 2, recall_vals, width, label="Macro Recall", color="#4C72B0")
    bars_f = ax2.bar(x + width / 2, f1_vals,     width, label="Macro F1",     color="#DD8452")

    for bar, v in zip(bars_r, recall_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)
    for bar, v in zip(bars_f, f1_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=7.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels([e["label"] for e in experiments], fontsize=8.5)
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Macro Recall & F1 by Experiment")
    ax2.legend()

    fig.suptitle("ViT Experiment Progression vs CNN — Test Set (224px RGB preprocessed)", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / "experiment_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating ViT visualizations...")
    plot_training_curves()
    plot_confusion_matrix()
    plot_experiment_comparison()
    print(f"\nAll plots saved to: {OUT_DIR.resolve()}")
