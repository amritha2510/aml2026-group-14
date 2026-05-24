import torch.nn as nn
import torchvision.models as models
from data.cnn_dataset import CNNDataset
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNet121Baseline(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=False):
        super().__init__()

        if pretrained:
            self.model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        else:
            self.model = densenet121(weights=None)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)