"""
TrueSight — frequency stream classifier.

Unlike the spatial stream (which benefits from a heavy ImageNet-pretrained
backbone since it's reasoning about natural facial texture), the
frequency stream operates on a much more structured, lower-entropy input
-- log-magnitude spectra. A small CNN trained from scratch is appropriate
here and keeps us aligned with the plan's explicit guidance: "Do not
begin with four huge neural networks. Start with transfer learning and
lightweight modules."

Input: 2-channel image (DCT log-magnitude, FFT log-magnitude), each
already normalized to [0, 1] by frequency_noise/dct.py and fft.py.
Output: single logit, same convention as the spatial stream (sigmoid ->
P(fake)), so losses/metrics/calibration code can be shared.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrequencyClassifier(nn.Module):
    def __init__(self, in_channels: int = 2, input_size: int = 224, dropout: float = 0.3):
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

            nn.AdaptiveAvgPool2d(1),  # -> (B, 128, 1, 1), robust to input_size changes
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, H, W) -> logits: (B,)"""
        features = self.features(x)
        logits = self.head(features).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.forward(x)
        return torch.sigmoid(logits)


def build_model(in_channels: int = 2) -> FrequencyClassifier:
    return FrequencyClassifier(in_channels=in_channels)
