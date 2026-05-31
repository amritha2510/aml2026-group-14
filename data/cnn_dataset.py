from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from constants import LABEL_TO_ID
from data.image_transforms import *
import numpy as np

def get_cnn_transforms(split: str):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Using ImageNet mean and std for normalization
                                std=[0.229, 0.224, 0.225])
    ])

class CNNDataset(Dataset):
    def __init__(self, df, split, aug_cfg=None, random_state=42):
        self.df = df.reset_index(drop=True)
        self.split = split
        self.aug_cfg = aug_cfg
        self.random_state = random_state
        self.transform = get_cnn_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # load image
        arr = load_image_as_rgb_array(row["filepath"], normalize=True)

        # apply augmentation only for train
        if self.split == "train" and self.aug_cfg["enabled"]:
            seed = make_deterministic_image_seed(idx, self.random_state)
            rng = np.random.default_rng(seed)

            if rng.random() < self.aug_cfg["probability"]:
                arr = augment_rgb_array(arr, self.aug_cfg, rng)

        # convert to PIL → transforms
        img = Image.fromarray((arr * 255).astype(np.uint8))

        img = self.transform(img)

        label = LABEL_TO_ID[row["label"]]

        return img, label