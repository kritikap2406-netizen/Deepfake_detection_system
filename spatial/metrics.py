"""
TrueSight — evaluation metrics.

Implements exactly the metrics the plan specifies for spatial model
evaluation: accuracy, precision, recall, F1, ROC-AUC. Kept dependency-
light (sklearn only, no extra framework) since fusion/ and other streams
will reuse this same module for their own evaluation.
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_binary_metrics(labels: list[float], probs: list[float], threshold: float = 0.5) -> dict:
    """labels: ground truth in {0,1}. probs: predicted P(fake) in [0,1]."""
    preds = [1 if p >= threshold else 0 for p in probs]

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

    # ROC-AUC is undefined with only one class present in a batch/split —
    # report None rather than raising, so training doesn't crash on a
    # single-class validation batch (common with small prototype splits).
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = roc_auc_score(labels, probs)

    return metrics
