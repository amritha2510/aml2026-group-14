import sys
import itertools
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score

from data.data_reader import load_config, get_required_config_path, load_metadata
from models.vit.dataset import ChestXrayViTDataset
from models.vit.transforms import get_train_transforms, get_eval_transforms
from models.vit.model import build_vit_model
from models.vit.train_vit import (
    train_one_epoch, evaluate_epoch,
    compute_class_weights_tensor, filter_model_rows, get_device,
)

CONFIG_PATH = Path("config.yaml")


def search_space():
    return {
        "lr":           [1e-4, 5e-5, 1e-5],
        "weight_decay": [0.01, 0.001],
        "dropout":      [0.1, 0.2],
    }


def build_configs():
    grid = search_space()
    return [dict(zip(grid.keys(), vals)) for vals in itertools.product(*grid.values())]


def train_and_evaluate(cfg, train_ds, val_ds, metadata_df, vit_cfg, device, epochs=5):
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)

    model = build_vit_model(
        model_name=vit_cfg["model_name"],
        pretrained=vit_cfg.get("pretrained", True),
        num_classes=vit_cfg.get("num_classes", 3),
        freeze_backbone=False,
        dropout=cfg["dropout"],
    ).to(device)

    weights = compute_class_weights_tensor(metadata_df, device)
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=vit_cfg.get("label_smoothing", 0.1),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_recall = -1.0
    for _ in range(1, epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, val_y, val_pred = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        val_recall = recall_score(val_y, val_pred, average="macro", zero_division=0)
        if val_recall > best_val_recall:
            best_val_recall = val_recall

    return best_val_recall


def main():
    config  = load_config(CONFIG_PATH)
    vit_cfg = config["vit"]

    metadata_path = get_required_config_path(config, CONFIG_PATH, "preprocessed_metadata_output_path")
    metadata_df   = filter_model_rows(load_metadata(metadata_path))

    device     = get_device()
    image_size = vit_cfg.get("image_size", 224)

    train_ds = ChestXrayViTDataset(metadata_df, "train", get_train_transforms(image_size, False))
    val_ds   = ChestXrayViTDataset(metadata_df, "val",   get_eval_transforms(image_size, False))

    configs = build_configs()
    print(f"[ViT-Tune] Running {len(configs)} configs × 5 epochs each on {device}")

    results   = []
    best_score = -1.0
    best_cfg   = None

    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {cfg}")
        score = train_and_evaluate(cfg, train_ds, val_ds, metadata_df, vit_cfg, device, epochs=5)
        print(f"  → val macro recall = {score:.4f}")

        results.append({**cfg, "val_macro_recall": score})

        if score > best_score:
            best_score = score
            best_cfg   = cfg

    results_df = pd.DataFrame(results).sort_values("val_macro_recall", ascending=False)
    out_path   = Path("vit_hyperparameter_tuning_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    print("\nBest config:")
    print(best_cfg)
    print(f"Best val macro recall: {best_score:.4f}")


if __name__ == "__main__":
    main()
