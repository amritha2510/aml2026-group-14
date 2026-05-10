import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from pathlib import Path

from data.data_reader import load_config, load_metadata, get_required_config_path
from data.image_transforms import (load_image_as_rgb_array, augment_flattened_rgb_training_data, get_image_aug_config)
from data.cnn_dataset import CNNDataset, get_cnn_transforms
from models.cnn.cnn import DenseNet121Baseline
from .cnn_utils import *
from constants import LABEL_TO_ID
from evaluation.metrics import ClassificationEvaluator

CONFIG_PATH = Path("config.yaml")

def main():
    config = load_config(CONFIG_PATH)
    cnn_cfg = config["cnn"]
    aug_cfg = get_image_aug_config(config, model_key="cnn")

    metadata_path = get_required_config_path(
        config, CONFIG_PATH, "preprocessed_metadata_output_path"
    )

    df = load_metadata(metadata_path)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    y_train = encode_labels(train_df)
    y_val = encode_labels(val_df)
    y_test = encode_labels(test_df)

    train_dataset = CNNDataset(
        train_df,
        split="train",
        aug_cfg=aug_cfg,
        random_state=cnn_cfg["random_state"]
    )
    val_dataset = CNNDataset(val_df, split="val")
    test_dataset = CNNDataset(test_df, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cnn_cfg["batch_size"],
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cnn_cfg["batch_size"],
        num_workers=2,
        shuffle = False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cnn_cfg["batch_size"],
        num_workers=2,
        shuffle = False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121Baseline(
        num_classes=cnn_cfg["num_classes"],
        pretrained=cnn_cfg["pretrained"],
        freeze_backbone=cnn_cfg["freeze_backbone"],
    ).to(device)

    class_weights = compute_class_weights(y_train).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cnn_cfg["learning_rate"]),
        weight_decay=float(cnn_cfg["weight_decay"]),
    )

    evaluator = ClassificationEvaluator(model_name="cnn_densenet121")

    best_val_recall = -1
    best_val_metrics = None

    best_model_path = Path(cnn_cfg["output_dir"]) / "best_model.pth"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cnn_cfg["epochs"]):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        y_val_true, y_val_pred = evaluate(model, val_loader, device)
        val_metrics = evaluator.evaluate_split(y_val_true, y_val_pred, "val")

        print(
            f"[Epoch {epoch+1}] "
            f"Loss={train_loss:.4f} | "
            f"Val Recall={val_metrics['macro_recall']:.4f}"
        )

        if val_metrics["macro_recall"] > best_val_recall:
            best_val_recall = val_metrics["macro_recall"]
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), best_model_path)

    model.load_state_dict(torch.load(best_model_path))

    y_test_true, y_test_pred = evaluate(model, test_loader, device)
    test_metrics = evaluator.evaluate_split(y_test_true, y_test_pred, "test")

    metrics_by_split = {
        "val": best_val_metrics,
        "test": test_metrics,
    }

    extra_artifacts = {
        "model_name": "cnn_densenet121",
        "num_epochs": cnn_cfg["epochs"],
        "batch_size": cnn_cfg["batch_size"],
    }

    run_dir = evaluator.save_run(
        base_output_dir=Path(cnn_cfg["output_dir"]),
        config=cnn_cfg,
        metrics_by_split=metrics_by_split,
        experiment_name=cnn_cfg.get("experiment_name", "cnn_densenet121"),
        extra_artifacts=extra_artifacts,
    )

    print("\n[FINAL TEST METRICS]")
    print(test_metrics)

    print(f"\n[INFO] Run saved to: {run_dir.resolve()}")
    print(f"[INFO] Best val macro recall: {best_val_recall:.4f}")

if __name__ == "__main__":
    main()