import torch
import torch.nn as nn
from torchvision import models


# ==========================================
# NEW MODEL 1: SimpleCNN
# ==========================================
# A lightweight 3-block CNN trained from scratch.
# Used as the baseline — no pretrained weights, no residual connections.
# Fastest to train but lowest accuracy due to limited capacity.
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8, pretrained=False):
        super().__init__()
        # pretrained arg is ignored (no pretrained weights exist for a custom CNN),
        # kept for consistent interface with other models in the comparison loop.
        self.features = nn.Sequential(
            # Block 1: 3 → 32 channels,  64×64 → 32×32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 32 → 64 channels,  32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 64 → 128 channels,  16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # Global Average Pool → 128-dim vector (works for any input resolution)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ==========================================
# NEW MODEL 2: ResNet18Classifier
# ==========================================
# ResNet-18 pretrained on ImageNet, with the final FC layer replaced.
# Deeper residual connections help it learn richer features than SimpleCNN,
# while being lighter than MobileNetV2 on parameter count.
# Supports optional backbone freezing for faster initial training.
class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Replace the final fully-connected layer
        in_features = backbone.fc.in_features          # 512 for ResNet-18
        backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class ViTClassifier(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)
        self.backbone.heads.head = nn.Linear(self.backbone.heads.head.in_features, num_classes)
        
    def forward(self, x): 
        return self.backbone(x)

class TransferMobileNet(nn.Module):
    def __init__(self, num_classes=8, freeze_backbone=True):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features    
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 4) -> None:
        total = len(self.features)
        for i, layer in enumerate(self.features):
            if i >= total - n:
                for param in layer.parameters():
                    param.requires_grad = True
        print(f"  Unfrozen last {n} backbone blocks for fine-tuning.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)