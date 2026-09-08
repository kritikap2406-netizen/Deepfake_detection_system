"""
TrueSight — spatial detection model.

Binary classifier (real=0, fake=1) operating on aligned 224x224 face
crops. Backbone is EfficientNet-B0 via timm with ImageNet-pretrained
weights (transfer learning) and a fresh binary head on top. Xception is
kept as a drop-in backup backbone per the plan — swap via `backbone`.

We output a single logit (not 2-class softmax) and use BCEWithLogitsLoss
/ sigmoid, which pairs naturally with focal loss and with temperature
scaling later in the fusion/calibration stage.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

SUPPORTED_BACKBONES = {
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b2": "efficientnet_b2",
    "xception": "xception",
}


class SpatialClassifier(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone {backbone!r}. Choose from {list(SUPPORTED_BACKBONES)}")

        timm_name = SUPPORTED_BACKBONES[backbone]
        # num_classes=0 -> timm strips its own classifier head and returns
        # pooled features, so we can attach our own binary head.
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),  # single logit: sigmoid -> P(fake)
        )

        self.backbone_name = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) -> logits: (B,)"""
        features = self.backbone(x)
        logits = self.head(features).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) -> P(fake): (B,) in [0, 1]"""
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)


def build_model(backbone: str = "efficientnet_b0", pretrained: bool = True) -> SpatialClassifier:
    return SpatialClassifier(backbone=backbone, pretrained=pretrained)
