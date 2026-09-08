"""
TrueSight — cross-dataset evaluation.

Per the plan: train on one dataset, test on a different dataset, repeat
across dataset pairs, compare single-stream vs multi-stream performance,
document generalization. This is the honest, generalization-first
methodology check the plan calls out as important for demonstrating the
system doesn't just memorize a dataset.

SCOPING CAVEAT (read this before trusting any number this module
produces): a fully rigorous version would retrain every stream model
(spatial, frequency, noise) fresh for each held-out dataset fold. That's
expensive even on our tiny stub data and out of scope for a hackathon
prototype's time budget. What this module actually does:

  - SINGLE-STREAM (spatial): genuinely leakage-free leave-one-dataset-out
    — a fresh spatial model is retrained from scratch for each fold,
    never seeing the held-out dataset during training. This number is
    methodologically sound, just based on a tiny/synthetic dataset.

  - MULTI-STREAM (fusion): leave-one-dataset-out at the FUSION layer
    only. The fusion meta-classifier is retrained fresh per fold, never
    seeing held-out-dataset LABELS during its own training — but the
    underlying spatial/frequency/noise SCORES it trains on come from
    Phase 4-6's models, which were trained on ALL four datasets
    (including whatever's held out in a given fold). This means the
    upstream features carry some information leakage from the "held
    out" dataset even though the fusion classifier itself doesn't. The
    fusion-layer generalization test is still meaningful (it measures
    whether the META-classifier overfits to dataset-specific score
    patterns, holding features fixed) but is NOT a true end-to-end
    multi-stream leave-one-out. This is flagged explicitly in every
    report this module returns, not just in this docstring.

Given both of the above, any accuracy numbers below reflect a tiny (12
rows per dataset), synthetic (non-photographic) stub dataset and should
be read as a demonstration of the evaluation METHODOLOGY, not a claim
about real-world generalization.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression

from spatial.dataset import _load_face_crop, normalize_to_tensor
from spatial.augmentation import apply_random_augmentations
from spatial.losses import FocalLoss
from spatial.metrics import compute_binary_metrics
from spatial.model import build_model
from fusion.features import build_feature_matrix, feature_names

METADATA_CSV = Path("datasets/metadata.csv")
FUSION_DATASET_CSV = Path("datasets/fusion_dataset.csv")


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def leave_one_dataset_out_folds(rows: list[dict]) -> list[tuple[str, list[dict], list[dict]]]:
    """Returns [(held_out_dataset_name, train_rows, test_rows), ...] —
    one fold per distinct dataset present in `rows`."""
    dataset_names = sorted({r["dataset"] for r in rows})
    folds = []
    for held_out in dataset_names:
        train_rows = [r for r in rows if r["dataset"] != held_out]
        test_rows = [r for r in rows if r["dataset"] == held_out]
        folds.append((held_out, train_rows, test_rows))
    return folds


class _RowListDataset(Dataset):
    """Lightweight spatial dataset variant that takes an explicit row
    list rather than reading from the fixed person-based train/val/test
    split (preprocessing.dataset_loader.TrueSightDataset) — needed here
    because cross-dataset folds are split by DATASET, not by person."""

    def __init__(self, rows: list[dict], augment: bool):
        self.rows = rows
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        label = int(row["label"] == "fake")
        crop = _load_face_crop(row["file_path"], label, face_size=224)
        if self.augment:
            crop = apply_random_augmentations(crop)
        tensor = normalize_to_tensor(crop)
        return tensor, torch.tensor(float(label))


def train_spatial_fold(train_rows: list[dict], epochs: int = 2, batch_size: int = 4) -> torch.nn.Module:
    """Fresh spatial model, trained ONLY on train_rows (i.e. 3 of the 4
    datasets in a leave-one-out fold). Few epochs deliberately — this is
    a fast per-fold retrain for methodology demonstration, not a full
    Phase-4-scale training run."""
    dataset = _RowListDataset(train_rows, augment=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = build_model(pretrained=False)
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for _ in range(epochs):
        for images, labels in loader:
            logits = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate_spatial_fold(model: torch.nn.Module, test_rows: list[dict]) -> dict:
    dataset = _RowListDataset(test_rows, augment=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.sigmoid(model(images))
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    return compute_binary_metrics(all_labels, all_probs)


def run_single_stream_cross_dataset_evaluation(epochs: int = 2) -> dict:
    """Genuinely leakage-free leave-one-dataset-out for the spatial
    stream alone. See module docstring."""
    rows = load_rows(METADATA_CSV)
    folds = leave_one_dataset_out_folds(rows)

    fold_results = []
    for held_out, train_rows, test_rows in folds:
        model = train_spatial_fold(train_rows, epochs=epochs)
        metrics = evaluate_spatial_fold(model, test_rows)
        fold_results.append({"held_out_dataset": held_out, "n_train": len(train_rows), "n_test": len(test_rows), "metrics": metrics})

    avg_accuracy = float(np.mean([f["metrics"]["accuracy"] for f in fold_results]))
    avg_f1 = float(np.mean([f["metrics"]["f1"] for f in fold_results]))

    return {
        "stream": "spatial (single-stream)",
        "leakage_status": "leakage-free — fresh model retrained per fold, held-out dataset never seen",
        "folds": fold_results,
        "average_accuracy": round(avg_accuracy, 4),
        "average_f1": round(avg_f1, 4),
    }


def train_fusion_fold(train_rows: list[dict]):
    X = build_feature_matrix(train_rows)
    y = np.array([int(r["label"]) for r in train_rows])
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


def evaluate_fusion_fold(model, test_rows: list[dict]) -> dict:
    X = build_feature_matrix(test_rows)
    y = [int(r["label"]) for r in test_rows]
    probs = model.predict_proba(X)[:, 1].tolist()
    return compute_binary_metrics(y, probs)


def run_multi_stream_cross_dataset_evaluation() -> dict:
    """Leave-one-dataset-out at the FUSION layer, using pre-computed
    stream scores from fusion_dataset.csv. See the module docstring's
    caveat: this is NOT a true end-to-end multi-stream leave-one-out,
    since the underlying stream scores came from models trained on all
    four datasets. Flagged explicitly in the returned dict.
    """
    rows = load_rows(FUSION_DATASET_CSV)
    folds = leave_one_dataset_out_folds(rows)

    fold_results = []
    for held_out, train_rows, test_rows in folds:
        model = train_fusion_fold(train_rows)
        metrics = evaluate_fusion_fold(model, test_rows)
        fold_results.append({"held_out_dataset": held_out, "n_train": len(train_rows), "n_test": len(test_rows), "metrics": metrics})

    avg_accuracy = float(np.mean([f["metrics"]["accuracy"] for f in fold_results]))
    avg_f1 = float(np.mean([f["metrics"]["f1"] for f in fold_results]))

    return {
        "stream": "fusion (multi-stream)",
        "leakage_status": (
            "PARTIAL LEAKAGE CAVEAT: fusion classifier itself is retrained fresh per "
            "fold and never sees held-out-dataset labels, but the spatial/frequency/"
            "noise SCORES it trains on come from models (Phase 4-6) trained on ALL "
            "four datasets, including whatever is held out in a given fold. This "
            "measures fusion-layer generalization given fixed upstream features, "
            "NOT a true leakage-free end-to-end multi-stream test."
        ),
        "folds": fold_results,
        "average_accuracy": round(avg_accuracy, 4),
        "average_f1": round(avg_f1, 4),
    }


def generate_cross_dataset_report(spatial_epochs: int = 2) -> dict:
    single_stream = run_single_stream_cross_dataset_evaluation(epochs=spatial_epochs)
    multi_stream = run_multi_stream_cross_dataset_evaluation()

    return {
        "dataset_caveat": (
            "All numbers below are computed on a tiny (12 rows/dataset), synthetic "
            "(non-photographic) stub dataset built for pipeline validation, not a "
            "real deepfake corpus. Treat this as a demonstration of the cross-dataset "
            "evaluation METHODOLOGY, not a real generalization claim."
        ),
        "single_stream": single_stream,
        "multi_stream": multi_stream,
        "comparison_caveat": (
            "single_stream and multi_stream are not perfectly apples-to-apples — see "
            "multi_stream.leakage_status. single_stream is the more trustworthy "
            "leakage-free number of the two."
        ),
    }
