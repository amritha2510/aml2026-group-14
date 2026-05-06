import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from pathlib import Path

from data.data_reader import load_config, load_metadata, get_required_config_path
from data.cnn_dataset import ChestXrayCNNDataset
from models.cnn.cnn import DenseNet121Baseline
from constants import LABEL_TO_ID
from evaluation.metrics import ClassificationEvaluator

CONFIG_PATH = Path("config.yaml")


def compute_class_weights(df):
    counts = df["label"].value_counts().to_dict()
    total = sum(counts.values())

    weights = []
    for label in sorted(LABEL_TO_ID, key=lambda x: LABEL_TO_ID[x]):
        weights.append(total / counts[label])

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()

    return weights


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def main():
    config = load_config(CONFIG_PATH)
    cnn_cfg = config["cnn"]

    metadata_path = get_required_config_path(
        config, CONFIG_PATH, "preprocessed_metadata_output_path"
    )

    df = load_metadata(metadata_path)

    # splits
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    # datasets
    train_dataset = ChestXrayCNNDataset(train_df, split="train")
    val_dataset = ChestXrayCNNDataset(val_df, split="val")
    test_dataset = ChestXrayCNNDataset(test_df, split="test")

    train_loader = DataLoader(train_dataset, batch_size=cnn_cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cnn_cfg["batch_size"], num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=cnn_cfg["batch_size"], num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenseNet121Baseline(
        num_classes=cnn_cfg["num_classes"],
        pretrained=cnn_cfg["pretrained"],
        freeze_backbone=cnn_cfg["freeze_backbone"],
    ).to(device)

    model.to(device)

    class_weights = compute_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(cnn_cfg["learning_rate"]),
        weight_decay=float(cnn_cfg["weight_decay"]),
    )

    evaluator = ClassificationEvaluator(model_name="cnn_densenet121")

    best_val_metrics = None
    best_val_recall = -1
    best_model_path = Path(cnn_cfg["output_dir"]) / "best_model.pth"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cnn_cfg["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        y_val, y_pred = evaluate(model, val_loader, device)
        val_metrics = evaluator.evaluate_split(y_val, y_pred, "val")

        print(f"[Epoch {epoch+1}] Loss={train_loss:.4f} | Val Recall={val_metrics['macro_recall']:.4f}")

        # save best model (based on recall)
        if val_metrics["macro_recall"] > best_val_recall:
            best_val_recall = val_metrics["macro_recall"]
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), best_model_path)

    # final test evaluation
    model.load_state_dict(torch.load(best_model_path))

    y_test, y_pred = evaluate(model, test_loader, device)
    test_metrics = evaluator.evaluate_split(y_test, y_pred, "test")

    metrics_by_split = {
        "val": best_val_metrics,
        "test": test_metrics,
    }

    extra_artifacts = {
        "model_name": "cnn_densenet121",
        "num_epochs": cnn_cfg["epochs"],
        "batch_size": cnn_cfg["batch_size"],
    }
    
    output_dir = Path(cnn_cfg["output_dir"])
    run_dir = evaluator.save_run(
        base_output_dir=output_dir,
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