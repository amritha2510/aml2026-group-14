import sys
import copy
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

from constants import LABEL_TO_ID
from data.data_reader import get_required_config_path, load_config, load_metadata
from evaluation.metrics import ClassificationEvaluator
from logistic_regression import filter_model_rows

config_path = Path("config.yaml")

class PreprocessedChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform = None):
        self.filepaths = df["filepath"].tolist()
        self.labels = df["label"].map(LABEL_TO_ID).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        image = Image.open(path)

        # Pre-trained ResNet/ViT strictly expects 3 color channels.
        # This converts our offline 1-channel grayscale safely back to 3 channels on the fly.
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

class AttentionFusionBlock(nn.Module):
    """
    We have to have the CNN and ViT features to the same dimension, treats them as a sequence of 2 tokens
    and uses Multihead Self-Attention to let them align and weigh each other dynamically before classification.
    """
    def __init__(self, conn_dim = 512, vit_dim = 192, embed_dim = 256):
        super().__init__()
        self.cnn_proj = nn.Linear(conn_dim, embed_dim)
        self.vit_proj = nn.Linear(vit_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.output_dim = embed_dim * 2

    def forward(self, local_feats, global_feats):
        cnn_token = self.cnn_proj(local_feats).unsqueeze(1)
        vit_token = self.vit_proj(global_feats).unsqueeze(1)

        seq = torch.cat([cnn_token, vit_token], dim=1)

        attended_seq = self.transformer(seq)

        fused_features = attended_seq.view(attended_seq.size(0), -1)

        return fused_features

class DualBranchConvViT(nn.Module):
    def __init__(self, num_classes = 3, noise_dropout_rates = 0.4, fusion_type = "concat"):
        super().__init__()
        self.fusion_type = fusion_type
        cnn_dim = 512
        vit_dim = 192

        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn_branch = nn.Sequential(*list(resnet18.children())[:-1])
        self.vit_branch = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit_branch.reset_classifier(0)

        dummy_tensor = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            cnn_dim = self.cnn_branch(dummy_tensor).view(1, -1).shape[1]
            vit_dim = self.vit_branch(dummy_tensor).shape[1]

        print(f"[DEBUG] Auto-detected CNN dim: {cnn_dim}, ViT dim: {vit_dim}")

        self.noise_injection = nn.Dropout(p=noise_dropout_rates)

        if self.fusion_type == "attention":
            self.fusion_block = AttentionFusionBlock(cnn_dim, vit_dim, embed_dim = 256)
            classifier_dimension = self.fusion_block.output_dim
        else:
            classifier_dimension = cnn_dim + vit_dim

        self.classifier = nn.Linear(classifier_dimension, num_classes)

    def forward(self, x):
        load_feats = self.cnn_branch(x).view(x.size(0), -1)

        global_feats = self.vit_branch(x)
        global_feats = self.noise_injection(global_feats)

        if self.fusion_type == "attention":
            fused_feats = self.fusion_block(load_feats, global_feats)
        else:
            fused_feats = torch.cat((load_feats, global_feats), dim = 1)

        return self.classifier(fused_feats)

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
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_loader = DataLoader(PreprocessedChestXrayDataset(train_df, transform = transform),batch_size = dl_cfg.get("batch_size", 32), shuffle = True)
    val_loader = DataLoader(PreprocessedChestXrayDataset(val_df, transform = transform),batch_size = dl_cfg.get("batch_size", 32), shuffle = False)
    test_loader = DataLoader(PreprocessedChestXrayDataset(test_df, transform = transform),batch_size = dl_cfg.get("batch_size", 32), shuffle = False)

    fusion_strategy = dl_cfg.get("fusion_type", "concat")
    print(f"[INFO] Initializing Dual-Branch Model with '{fusion_strategy.upper()}' fusion...")

    model = DualBranchConvViT(
        num_classes = 3,
        noise_dropout_rates = dl_cfg.get("noise_dropout_rates", 0.4),
        fusion_type = fusion_strategy
    ).to(device)

    class_weight = torch.tensor(dl_cfg.get("class_weight", [2.0, 2.0, 1.0]), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=dl_cfg.get("learning_rate", 1e-4))

    evaluator = ClassificationEvaluator(model_name = f"DualBranchConvViT_{fusion_strategy}")
    best_val_recall = -1.0
    best_val_metrics = None
    best_weights = None

    epochs = dl_cfg.get("epochs", 10)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _ , preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_metrics = evaluator.compute_metrics(all_labels, all_preds)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} | Val Recall: {val_metrics['macro_recall']:.4f} | Val F1: {val_metrics['macro_f1']:.4f}")

        if val_metrics['macro_recall'] > best_val_recall:
            best_val_recall = val_metrics['macro_recall']
            best_val_metrics = copy.deepcopy(val_metrics)
            best_weights = copy.deepcopy(model.state_dict())

    print("\n[INFO] Evaluating on Test Split with Best Weights...")
    if best_weights is None:
        raise RuntimeError("Training finished without capturing best model weights.")
    model.load_state_dict(best_weights)
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().tolist())
            test_labels.extend(labels.cpu().tolist())

    final_metrics = {
            "val": best_val_metrics,
            "test": evaluator.evaluate_split(test_labels, test_preds, "test")
        }

    run_dir = evaluator.save_run(
        base_output_dir = output_dir,
        config = dl_cfg,
        metrics_by_split=final_metrics,
        experiment_name=f"{dl_cfg.get('experiment_name', 'deep_learning')}_{fusion_strategy}"
    )

    torch.save(best_weights, run_dir / "best_model_weights.pth")
    print(f"\n[INFO] Run successfully saved to: {run_dir.resolve()}")

if __name__ == "__main__":
    main()






