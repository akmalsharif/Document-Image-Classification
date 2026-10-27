"""ConvNeXt model factory."""

from torch import nn
from torchvision import models


def create_model(num_classes, pretrained=True):
    """Build a ConvNeXt-Tiny classifier with the requested number of classes."""
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)
    input_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(input_features, num_classes)
    return model

