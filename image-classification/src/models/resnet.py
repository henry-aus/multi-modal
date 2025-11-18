"""
ResNet-based CNN model for image classification
Uses pre-trained ResNet34 with transfer learning
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class ResNetClassifier(nn.Module):
    """ResNet-based classifier with customizable number of classes"""

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        freeze_backbone: bool = False
    ):
        """
        Initialize ResNet classifier

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pre-trained weights
            dropout_rate: Dropout rate for the classifier head
            freeze_backbone: Whether to freeze backbone parameters
        """
        super(ResNetClassifier, self).__init__()

        # Load pre-trained ResNet34
        self.backbone = models.resnet34(pretrained=pretrained)

        # Get the number of features from the backbone
        num_features = self.backbone.fc.in_features

        # Replace the classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

        # Optionally freeze backbone parameters
        if freeze_backbone:
            self._freeze_backbone()

        self.num_classes = num_classes

    def _freeze_backbone(self):
        """Freeze backbone parameters except for the final classifier"""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:  # Don't freeze the classifier head
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.backbone(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classifier head"""
        # Forward pass through all layers except the final classifier
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def create_model(
    num_classes: int = 10,
    pretrained: bool = True,
    dropout_rate: float = 0.5,
    freeze_backbone: bool = False,
    device: Optional[torch.device] = None
) -> ResNetClassifier:
    """
    Create and initialize a ResNet classifier

    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights
        dropout_rate: Dropout rate for the classifier head
        freeze_backbone: Whether to freeze backbone parameters
        device: Device to move the model to

    Returns:
        Initialized ResNet classifier
    """
    model = ResNetClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone
    )

    if device is not None:
        model = model.to(device)

    return model


def count_parameters(model: nn.Module) -> tuple:
    """
    Count the number of parameters in the model

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params