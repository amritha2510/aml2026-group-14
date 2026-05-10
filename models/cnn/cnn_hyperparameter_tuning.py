import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools
import numpy as np
from pathlib import Path
import pandas as pd
from data.data_reader import load_config, load_metadata, get_required_config_path
from data.image_transforms import (load_image_as_rgb_array, augment_flattened_rgb_training_data, get_image_aug_config, make_deterministic_image_seed)
from data.cnn_dataset import CNNDataset, get_cnn_transforms
from models.cnn.cnn import DenseNet121Baseline
from constants import LABEL_TO_ID
from evaluation.metrics import ClassificationEvaluator
from .cnn_utils import *

def search_space():
    return {
        "learning_rate": [1e-3, 3e-4, 1e-4],
        "weight_decay": [0, 1e-4],
        "batch_size": [16, 32],
        "freeze_backbone": [True, False],
    }

def build_configs():
    param_grid = search_space()
    keys = param_grid.keys()
    values = param_grid.values()
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def train_and_evaluate(cfg, train_df, val_df, cnn_cfg, device, aug_cfg, epochs):

    train_dataset = CNNDataset(
        train_df,
        split="train",
        aug_cfg=aug_cfg,
        random_state=cnn_cfg["random_state"]
    )

    val_dataset = CNNDataset(val_df, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True, 
        num_workers = 2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        num_workers = 2
    )

    y_train = encode_labels(train_df)
    y_val = encode_labels(val_df)


    model = DenseNet121Baseline(
        num_classes=cnn_cfg["num_classes"],
        pretrained=cnn_cfg["pretrained"],
        freeze_backbone=cfg["freeze_backbone"],
    ).to(device)

    class_weights = compute_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    evaluator = ClassificationEvaluator(model_name="cnn")

    best_val_recall = -1

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, criterion, optimizer, device)

        y_val_true, y_val_pred = evaluate(model, val_loader, device)
        val_metrics = evaluator.evaluate_split(y_val_true, y_val_pred, "val")

        if val_metrics["macro_recall"] > best_val_recall:
            best_val_recall = val_metrics["macro_recall"]

    return best_val_recall

def main():
    config = load_config(Path("config.yaml"))
    cnn_cfg = config["cnn"]
    aug_cfg = get_image_aug_config(config, model_key="cnn")

    metadata_path = get_required_config_path(
        config, Path("config.yaml"), "preprocessed_metadata_output_path"
    )

    df = load_metadata(metadata_path)

    # splits
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    y_train = encode_labels(train_df)
    y_val = encode_labels(val_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = build_configs()

    results = []
    best_score = -1
    best_cfg = None

    for i, cfg in enumerate(configs, 1):
        print(f"\n[INFO] Running config {i}/{len(configs)}: {cfg}")

        score = train_and_evaluate(
            cfg,
            train_df,
            val_df,
            cnn_cfg,
            device, 
            aug_cfg,
            epochs=5,
        )

        results.append({
            **cfg,
            "val_macro_recall": score
        })

        if score > best_score:
            best_score = score
            best_cfg = cfg

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("val_macro_recall", ascending=False)

    results_df.to_csv("cnn_model_selection.csv", index=False)

    print("\nBest config:")
    print(best_cfg)
    print(f"Best val recall: {best_score:.4f}")
if __name__ == "__main__":
    main()