from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from constants import LABEL_TO_ID

def get_cnn_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),  # convert to 3-channel
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

class ChestXrayCNNDataset(Dataset):
    def __init__(self, df, split):
        self.df = df.reset_index(drop=True)
        self.transform = get_cnn_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("L")
        label = LABEL_TO_ID[row["label"]]
        img = self.transform(img)
        return img, label