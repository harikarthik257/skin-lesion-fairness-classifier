import torch.nn as nn
import timm

CLASS_MAPPING = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AKIEC': 3, 'BKL': 4, 'DF': 5, 'VASC': 6}
INV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
FITZ_MAPPING = {'Light (I-II)': 0, 'Medium (III-IV)': 1, 'Dark (V-VI)': 2}
INV_FITZ_MAPPING = {0: 'Light (I-II)', 1: 'Medium (III-IV)', 2: 'Dark (V-VI)', -1: 'Unknown'}
FITZ_NAMES = {0: 'I-II', 1: 'III-IV', 2: 'V-VI'}


class BaselineModel(nn.Module):
    """Plain EfficientNet-B4 classifier, no fairness intervention."""

    def __init__(self, num_classes=7, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class FairnessModel(nn.Module):
    """EfficientNet-B4 + Dropout head, trained with focal loss and per-Fitzpatrick-group
    reweighting. The Dropout layer also enables MC-Dropout uncertainty estimation at
    inference time."""

    def __init__(self, num_classes=7, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(p=0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def enable_dropout(model):
    """Force Dropout layers back to train mode while everything else (BatchNorm, etc.)
    stays in eval mode -- used for Monte Carlo Dropout inference."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
