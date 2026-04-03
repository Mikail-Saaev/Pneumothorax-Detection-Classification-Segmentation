#!/usr/bin/env python3
"""Model definitions: ResNet18 and DenseNet121 (CheXNet-style) classifiers."""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    DenseNet121_Weights,
)


class ResNet18Classifier(nn.Module):
    """ResNet-18 fine-tuned for binary pneumothorax classification.

    The final fully-connected layer is replaced by a dropout + linear head
    that outputs a single raw logit (use with ``BCEWithLogitsLoss``).

    Args:
        pretrained:    Load ImageNet weights when ``True``.
        dropout_rate:  Dropout probability before the classification head.
    """

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.5) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)


class DenseNet121Classifier(nn.Module):
    """DenseNet-121 classifier inspired by CheXNet.

    Features a deeper 3-layer MLP head (512 → 128 → 1) with progressive
    dropout to reduce overfitting on small medical imaging datasets.

    Args:
        pretrained:    Load ImageNet weights when ``True``.
        dropout_rate:  Base dropout probability; halved / quartered in deeper layers.
    """

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.5) -> None:
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)
        in_features = backbone.classifier.in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 4),
            nn.Linear(128, 1),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)
