"""
TrueSight — noise residual stream classifier.

Same design philosophy as frequency_noise/frequency_model.py: the input
here (3-channel SRM residual stack) is already a heavily processed,
low-level signal-processing representation, not natural image content.
A small from-scratch CNN is appropriate -- no ImageNet pretraining
benefit, no network dependency, matches the plan's "lightweight modules
first" guidance.

Architecture is intentionally identical in shape to FrequencyClassifier
(only in_channels differs) so the two streams train/infer with the same
computational budget and are directly comparable during fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NoiseClassifier(nn.Module):
    def __init__(self, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),  # 224 -> 112
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),           # 112 -> 56
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # 56 -> 28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # 28 -> 14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> logits: (B,)"""
        features = self.features(x)
        logits = self.head(features).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)


def build_model(in_channels: int = 3) -> NoiseClassifier:
    return NoiseClassifier(in_channels=in_channels)
