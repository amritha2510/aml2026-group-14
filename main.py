

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import timm  # pip install timm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score


# ==========================================
# 1. DATASET DEFINITION
# ==========================================
class Pneumonia3ClassDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.filepaths = []
        self.labels = []
        self.transform = transform

        # Print where the script is looking
        print(f"  -> Searching for images in: {os.path.abspath(root_dir)}")

        # Look for both .jpeg and .jpg
        normal_paths = []
        pneumonia_paths = []
        for ext in ('*.jpeg', '*.jpg', '*.png'):
            normal_paths.extend(glob.glob(os.path.join(root_dir, 'NORMAL', ext)))
            pneumonia_paths.extend(glob.glob(os.path.join(root_dir, 'PNEUMONIA', ext)))

        if len(normal_paths) == 0 and len(pneumonia_paths) == 0:
            raise FileNotFoundError(
                f"CRITICAL: Could not find any images in {root_dir}/NORMAL or {root_dir}/PNEUMONIA. Please check your folder paths!")

        for path in normal_paths:
            self.filepaths.append(path)
            self.labels.append(0)  # Normal

        for path in pneumonia_paths:
            if 'bacteria' in os.path.basename(path).lower():
                self.filepaths.append(path)
                self.labels.append(1)  # Bacterial
            elif 'virus' in os.path.basename(path).lower():
                self.filepaths.append(path)
                self.labels.append(2)  # Viral

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# ==========================================
# 2. MODEL DEFINITIONS
# ==========================================
class DualBranchConvViT(nn.Module):
    def __init__(self, num_classes=3, noise_dropout_rate=0.4):
        super().__init__()
        # Stream 1: Local CNN
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn_branch = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_features_dim = 512

        # Stream 2: Global ViT
        self.vit_branch = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
        self.vit_features_dim = 192

        # The Twist: Noise Injection
        self.noise_injection = nn.Dropout(p=noise_dropout_rate)

        # Fusion
        self.classifier = nn.Linear(self.cnn_features_dim + self.vit_features_dim, num_classes)

    def forward(self, x):
        local_features = self.cnn_branch(x).view(x.size(0), -1)
        global_features = self.vit_branch(x)
        global_features = self.noise_injection(global_features)
        fused_features = torch.cat((local_features, global_features), dim=1)
        return self.classifier(fused_features)


def get_resnet_baseline(num_classes=3):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_normal_vit_baseline(num_classes=3):
    """
    Standard Vision Transformer Baseline.
    Using 'vit_tiny' for the MVP to match the dual-branch parameters,
    but you can change this to 'vit_base_patch16_224' for your final run.
    """
    return timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)


# ==========================================
# 3. TRAINING & EVALUATION ENGINES
# ==========================================
def train_and_eval_dl_model(model, train_loader, val_loader, model_name, epochs=5, device='mps'):
    print(f"\n{'=' * 40}")
    print(f"🚀 STARTING EXPERIMENT: {model_name}")
    print(f"{'=' * 40}")

    model = model.to(device)
    # Applying weights for imbalanced classes; move the tensor to MPS
    class_weights = torch.tensor([2.0, 2.0, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    final_recall, final_f1 = 0.0, 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            # Move data to Mac GPU
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation Phase
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                # Must move predictions back to CPU for Scikit-Learn to calculate metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(
            f"[Epoch {epoch + 1}/{epochs}] Loss: {running_loss / len(train_loader):.4f} | Val Recall: {macro_recall:.4f} | Val F1: {macro_f1:.4f}")

        final_recall, final_f1 = macro_recall, macro_f1

    return final_recall, final_f1


def run_logistic_regression_baseline(train_loader, val_loader):
    print(f"\n{'=' * 40}")
    print(f"📊 STARTING EXPERIMENT: Logistic Regression + PCA")
    print(f"{'=' * 40}")

    def extract_flattened_data(loader):
        X, y = [], []
        pool = nn.AdaptiveAvgPool2d((32, 32))
        for images, labels in loader:
            pooled_images = pool(images).view(images.size(0), -1).numpy()
            X.append(pooled_images)
            y.extend(labels.numpy())
        return np.vstack(X), np.array(y)

    print("Extracting and downsampling images...")
    X_train, y_train = extract_flattened_data(train_loader)
    X_val, y_val = extract_flattened_data(val_loader)

    print("Running PCA (reducing to 100 components)...")
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    print("Training Logistic Regression...")
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_val_pca)
    macro_recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

    print(f"[Finished] Val Recall: {macro_recall:.4f} | Val F1: {macro_f1:.4f}")
    return macro_recall, macro_f1


# ==========================================
# 4. MASTER EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Settings
    TRAIN_DIR = 'chest_xray/train'
    VAL_DIR = 'chest_xray/val'
    BATCH_SIZE = 16
    EPOCHS = 5

    # --- MAC OS SILICON / MPS DETECTION ---
    if torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    elif torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    print(f"Using Hardware Acceleration: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        print("\nLoading datasets...")
        train_dataset = Pneumonia3ClassDataset(root_dir=TRAIN_DIR, transform=transform)
        val_dataset = Pneumonia3ClassDataset(root_dir=VAL_DIR, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")

        results = {}

        # 1. Run Logistic Regression Baseline
        lr_recall, _ = run_logistic_regression_baseline(train_loader, val_loader)
        results['Logistic Regression (PCA)'] = lr_recall

        # 2. Run ResNet-50 ML Baseline
        resnet = get_resnet_baseline()
        res_recall, _ = train_and_eval_dl_model(resnet, train_loader, val_loader, "ResNet-50 Baseline", EPOCHS, DEVICE)
        results['ResNet-50 Baseline'] = res_recall

        # 3. Run Normal ViT Baseline
        normal_vit = get_normal_vit_baseline()
        vit_recall, _ = train_and_eval_dl_model(normal_vit, train_loader, val_loader, "Normal ViT Baseline", EPOCHS,
                                                DEVICE)
        results['Normal ViT Baseline'] = vit_recall

        # 4. Run Proposed Dual-Branch Model
        our_model = DualBranchConvViT(num_classes=3, noise_dropout_rate=0.4)
        our_recall, _ = train_and_eval_dl_model(our_model, train_loader, val_loader, "Dual-Branch Conv-ViT (Ours)",
                                                EPOCHS, DEVICE)
        results['Dual-Branch Conv-ViT (Ours)'] = our_recall

        # --- FINAL LEADERBOARD ---
        print("\n" + "*" * 50)
        print("🏆 FINAL RESULTS LEADERBOARD (MACRO RECALL) 🏆")
        print("*" * 50)
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        for rank, (model_name, score) in enumerate(sorted_results, 1):
            print(f"{rank}. {model_name:<30} : {score:.4f}")
        print("*" * 50)

    except FileNotFoundError:
        print(f"\n[ERROR] Dataset not found at {TRAIN_DIR} or {VAL_DIR}.")
        print("Please ensure your Kaggle dataset is unzipped into a 'data/chest_xray/' folder.")


