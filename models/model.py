from torch import nn
from torchvision import models

from config import CONFIG


class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Load pretrained MobileNetV2
        self.base_model = models.mobilenet_v2(weights="IMAGENET1K_V1")

        # Freeze backbone if enabled
        if CONFIG["freeze_backbone"]:
            for param in self.base_model.features.parameters():
                param.requires_grad = False

        # Replace classifier head
        in_features = self.base_model.last_channel

        self.base_model.classifier = nn.Sequential(
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)