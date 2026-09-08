"""
TrueSight — focal loss.

Deepfake datasets are often imbalanced (more real than fake, or vice
versa depending on the dataset), and many examples are "easy" (obviously
real or obviously fake) while a smaller set are genuinely hard boundary
cases. Plain BCE spends most of its gradient on the easy majority. Focal
loss down-weights easy, well-classified examples so training focuses on
the hard ones — exactly what Phase 4 of the plan calls for.

FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

- gamma controls how aggressively easy examples are down-weighted
  (0 = plain BCE, higher = more aggressive focus on hard examples)
- alpha balances the two classes directly (class weighting)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: (B,) raw scores. targets: (B,) in {0., 1.}."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probas = torch.sigmoid(logits)
        p_t = probas * targets + (1 - probas) * (1 - targets)  # P(correct class)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(labels: list[int]) -> dict[int, float]:
    """Inverse-frequency class weights, used for balanced sampling or as
    an alternative/complement to focal loss's alpha term."""
    n_total = len(labels)
    n_real = sum(1 for l in labels if l == 0)
    n_fake = sum(1 for l in labels if l == 1)

    if n_real == 0 or n_fake == 0:
        raise ValueError("Cannot compute class weights: one class has zero examples.")

    return {
        0: n_total / (2 * n_real),
        1: n_total / (2 * n_fake),
    }
