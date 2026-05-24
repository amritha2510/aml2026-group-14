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
from constants import LABEL_TO_ID
from evaluation.metrics import ClassificationEvaluator


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


def encode_labels(df):
    return df["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)

def load_images_as_arrays(df):
    images = []

    for i, path in enumerate(df["filepath"].tolist()):
        arr = load_image_as_rgb_array(path, normalize=True)
        images.append(arr)

        if (i + 1) % 500 == 0 or (i + 1) == len(df):
            print(f"[INFO] Loaded {i + 1}/{len(df)} images")

    return np.stack(images)  # shape: (N, H, W, C)


def compute_class_weights(y): #to count number of samples per class
    unique, counts = np.unique(y, return_counts=True)
    total = counts.sum()

    weights = np.zeros(len(LABEL_TO_ID))
    for u, c in zip(unique, counts):
        weights[u] = total / c

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() #normalising weights
    return weights

