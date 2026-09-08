"""
TrueSight — confidence calibration (temperature scaling).

A raw model probability like 0.94 is not automatically a trustworthy
"94% confidence" — models, especially ones like XGBoost/MLP that can fit
training data very tightly, are often overconfident: their predicted
probabilities cluster near 0 and 1 more than their actual accuracy
justifies. Temperature scaling is the standard, minimal fix: divide the
model's logit by a single learned scalar T > 1 before the sigmoid, which
pulls overconfident probabilities back toward 0.5 without changing which
class is predicted (T only rescales confidence, never flips a
prediction, since it doesn't change which side of 0.5 the probability
falls on).

We work purely in probability space (converting to logit and back)
rather than requiring access to each model's internal raw scores. This
is what lets calibration apply uniformly regardless of which fusion
model won Phase 10 — XGBoost, Logistic Regression, and MLPClassifier all
expose predict_proba, but not all expose a raw decision function in a
consistent way (MLPClassifier in particular has none).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_CALIBRATION_PATH = Path("models/fusion/temperature_scale.json")

_EPS = 1e-6
_T_GRID = np.linspace(0.05, 5.0, 200)


def prob_to_logit(p: np.ndarray) -> np.ndarray:
    p_clipped = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p_clipped / (1.0 - p_clipped))


def logit_to_prob(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale probabilities by dividing their logit by `temperature`.
    T > 1 pulls confidence toward 0.5 (softens overconfidence).
    T < 1 sharpens confidence away from 0.5.
    T == 1 is a no-op (identical to uncalibrated probabilities).
    """
    logits = prob_to_logit(np.asarray(probs, dtype=np.float64))
    return logit_to_prob(logits / temperature)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(np.asarray(probs, dtype=np.float64), _EPS, 1.0 - _EPS)
    labels = np.asarray(labels, dtype=np.float64)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE: split predictions into confidence bins, and within each bin
    compare the bin's average predicted probability against its actual
    accuracy. A well-calibrated model has near-zero gap in every bin.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(probs)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            continue

        avg_confidence = probs[in_bin].mean()
        avg_accuracy = labels[in_bin].mean()
        ece += (n_in_bin / total) * abs(avg_confidence - avg_accuracy)

    return float(ece)


@dataclass
class CalibrationResult:
    temperature: float
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float


def fit_temperature(raw_probs: np.ndarray, labels: np.ndarray) -> CalibrationResult:
    """Grid search over a range of T values, selecting the one that
    minimizes validation NLL. A grid search (rather than gradient-based
    optimization) is used deliberately: it's simple, always converges,
    and is easy to inspect/plot — appropriate for a hackathon prototype
    where debuggability matters more than shaving optimizer iterations.
    """
    raw_probs = np.asarray(raw_probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    nll_before = negative_log_likelihood(raw_probs, labels)
    ece_before = expected_calibration_error(raw_probs, labels)

    best_t = 1.0
    best_nll = nll_before
    for t in _T_GRID:
        calibrated = apply_temperature(raw_probs, t)
        nll = negative_log_likelihood(calibrated, labels)
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    calibrated_probs = apply_temperature(raw_probs, best_t)
    ece_after = expected_calibration_error(calibrated_probs, labels)

    return CalibrationResult(
        temperature=round(best_t, 4),
        nll_before=round(nll_before, 4),
        nll_after=round(best_nll, 4),
        ece_before=round(ece_before, 4),
        ece_after=round(ece_after, 4),
    )


def save_calibration(result: CalibrationResult, path: Path = DEFAULT_CALIBRATION_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "temperature": result.temperature,
            "nll_before": result.nll_before,
            "nll_after": result.nll_after,
            "ece_before": result.ece_before,
            "ece_after": result.ece_after,
        }, f, indent=2)


def load_temperature(path: Path = DEFAULT_CALIBRATION_PATH) -> float:
    """Returns 1.0 (no-op) if no calibration file exists yet, so
    fusion/inference.py can run before Phase 11's training step without
    crashing — it just won't be calibrated until this is run."""
    if not Path(path).exists():
        return 1.0
    with open(path) as f:
        data = json.load(f)
    return data["temperature"]
