import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
class ResNet50(nn.Module):
    def __init__(self, num_classes=4, weights='ResNet50_Weights', freeze_layers=False):
        super(ResNet50, self).__init__()

        # Load a pretrained ResNet-50 model with specified weights
        if weights == 'ResNet50_Weights':
            resnet50 = models.resnet50(weights=ResNet50_Weights)

        else:
            resnet50 = models.resnet50()

        if freeze_layers:
            # Freeze all layers except the custom classification layers
            for param in resnet50.parameters():
                param.requires_grad = True

        # Remove the final classification layer
        self.resnet50 = nn.Sequential(*list(resnet50.children())[:-2])

        # Add custom classification layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        features = self.resnet50(x)
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
