"""
TrueSight — robustness report generation.

Two genuinely meaningful measurements, both independent of dataset size
or realism (unlike accuracy, which our tiny synthetic stub dataset
cannot honestly support claims about — see the caveat printed in every
report this module produces):

1. Degradation stability: for each degradation type in
   evaluation.degradations.DEGRADATION_SUITE, how much does the SAME
   model's own prediction shift when the SAME input is degraded? This
   only requires the model to be self-consistent, not accurate — a
   meaningful signal even on a handful of samples.

2. Adversarial robustness: FGSM/PGD attack success rate against the
   model's own correctly-self-consistent predictions (see
   evaluation/adversarial.py). Also independent of ground-truth labels
   or dataset realism — it measures how easily the model's own decision
   boundary can be crossed by a bounded perturbation.
"""

from __future__ import annotations

import numpy as np
import torch

from evaluation.adversarial import fgsm_attack, pgd_attack, attack_success_rate
from evaluation.degradations import DEGRADATION_SUITE
from spatial.dataset import normalize_to_tensor
from spatial.model import SpatialClassifier


def evaluate_degradation_stability(
    model: SpatialClassifier,
    face_crops: list[np.ndarray],
) -> dict:
    """face_crops: list of BGR uint8 224x224 face crops (clean, undegraded).
    For each degradation type, computes:
      - mean_abs_score_shift: average |P(fake)_degraded - P(fake)_clean|
      - prediction_flip_rate: fraction of samples where the >=0.5
        decision boundary was crossed by the degradation
    """
    model.eval()
    results = {}

    with torch.no_grad():
        clean_tensors = torch.stack([normalize_to_tensor(c) for c in face_crops])
        clean_probs = torch.sigmoid(model(clean_tensors)).tolist()

    for name, degrade_fn in DEGRADATION_SUITE.items():
        degraded_crops = [degrade_fn(c.copy()) for c in face_crops]
        with torch.no_grad():
            degraded_tensors = torch.stack([normalize_to_tensor(c) for c in degraded_crops])
            degraded_probs = torch.sigmoid(model(degraded_tensors)).tolist()

        shifts = [abs(d - c) for d, c in zip(degraded_probs, clean_probs)]
        flips = [
            (d >= 0.5) != (c >= 0.5)
            for d, c in zip(degraded_probs, clean_probs)
        ]

        results[name] = {
            "mean_abs_score_shift": round(float(np.mean(shifts)), 4),
            "max_abs_score_shift": round(float(np.max(shifts)), 4),
            "prediction_flip_rate": round(float(np.mean(flips)), 4),
        }

    return results


def evaluate_adversarial_robustness(
    model: SpatialClassifier,
    face_crops: list[np.ndarray],
    epsilons: list[float] = (0.01, 0.05, 0.1),
) -> dict:
    """Runs both FGSM and PGD at each epsilon against the model's own
    clean predictions (used as pseudo-labels — see attack_success_rate's
    docstring for why this is the correct methodology for measuring
    decision-boundary fragility rather than accuracy)."""
    model.eval()

    tensors = [normalize_to_tensor(c).unsqueeze(0) for c in face_crops]
    with torch.no_grad():
        pseudo_labels = [(torch.sigmoid(model(t)) >= 0.5).float() for t in tensors]

    results = {"fgsm": {}, "pgd": {}}

    for eps in epsilons:
        results["fgsm"][f"epsilon_{eps}"] = attack_success_rate(
            model, tensors, pseudo_labels, fgsm_attack, epsilon=eps
        )
        results["pgd"][f"epsilon_{eps}"] = attack_success_rate(
            model, tensors, pseudo_labels, pgd_attack, epsilon=eps, alpha=eps / 4, steps=10
        )

    return results


def generate_robustness_report(model: SpatialClassifier, face_crops: list[np.ndarray]) -> dict:
    if len(face_crops) < 10:
        caveat = (
            f"CAVEAT: only {len(face_crops)} face crop(s) evaluated. Degradation "
            f"stability and adversarial flip-rate are meaningful signals about "
            f"THIS model's self-consistency regardless of sample count, but a "
            f"larger, more diverse sample would give a more representative picture."
        )
    else:
        caveat = None

    return {
        "n_samples": len(face_crops),
        "caveat": caveat,
        "degradation_stability": evaluate_degradation_stability(model, face_crops),
        "adversarial_robustness": evaluate_adversarial_robustness(model, face_crops),
    }
