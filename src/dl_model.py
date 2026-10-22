import torch
import torch.nn as nn
from torchvision import models

def build_mobilenet_classifier(num_classes: int = 4, freeze_backbone: bool = True) -> nn.Module:
    """
    Builds a lightweight MobileNetV2 architecture with a frozen ImageNet feature extractor
    and a custom classification head designed for edge produce sorting.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
            
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    return model
