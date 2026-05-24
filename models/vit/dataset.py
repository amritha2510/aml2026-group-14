import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from PIL import Image
from torch.utils.data import Dataset

from constants import LABEL_TO_ID


class ChestXrayViTDataset(Dataset):
    """
    Reads RAW images (not preprocessed) with ViT-specific transforms.
    Mirrors Pneumonia3ClassDataset from main.py but uses metadata DataFrame.
    """

    def __init__(self, metadata_df, split, transform=None):
        self.df = metadata_df[metadata_df["split"] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # .convert("RGB") handles the grayscale→3ch issue:
        # most images are mode "L", some are "RGB" but visually grayscale.
        # This repeats the single channel across R,G,B — works fine with
        # ImageNet-pretrained weights.
        img = Image.open(row["filepath"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = LABEL_TO_ID[row["label"]]
        return img, label