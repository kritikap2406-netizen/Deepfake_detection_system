"""
TrueSight — fusion meta-classifier training and selection.

Phase 10: train XGBoost (primary choice per configs/pipeline.yaml),
compare against Logistic Regression and a small MLP, then select whichever
performs best on the validation split. Reuses spatial/metrics.py's
compute_binary_metrics — the same accuracy/precision/recall/F1/ROC-AUC
used by every other stream — so fusion results are directly comparable.

Usage:
    python -m fusion.meta_classifier
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from fusion.features import build_feature_matrix, feature_names
from spatial.metrics import compute_binary_metrics

FUSION_DATASET_CSV = Path("datasets/fusion_dataset.csv")
MODEL_DIR = Path("models/fusion")
BEST_MODEL_PATH = MODEL_DIR / "best_fusion_model.joblib"
COMPARISON_REPORT_PATH = MODEL_DIR / "fusion_comparison.json"

# Candidate models to compare. Kept small and interpretable per the
# plan's emphasis on comparing simple alternatives before committing.
CANDIDATE_MODELS = {
    "xgboost": lambda: XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        eval_metric="logloss", random_state=42,
    ),
    "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "small_mlp": lambda: MLPClassifier(
        hidden_layer_sizes=(16, 8), max_iter=2000, random_state=42, early_stopping=True,
    ),
}


def load_fusion_rows(csv_path: Path = FUSION_DATASET_CSV) -> dict[str, list[dict]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found — run `python -m fusion.build_dataset` first.")

    with open(csv_path, newline="") as f:
        all_rows = list(csv.DictReader(f))

    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for row in all_rows:
        by_split[row["split"]].append(row)
    return by_split


def rows_to_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = build_feature_matrix(rows)
    y = np.array([int(r["label"]) for r in rows])
    return X, y


def train_and_evaluate(name, build_fn, X_train, y_train, X_val, y_val) -> dict:
    model = build_fn()
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]  # P(fake)
    metrics = compute_binary_metrics(y_val.tolist(), val_probs.tolist())

    return {"name": name, "model": model, "metrics": metrics}


def select_best(results: list[dict]) -> dict:
    """Rank by F1 first (balances precision/recall, appropriate for a
    detector where both false alarms and missed fakes matter), falling
    back to accuracy if F1 ties (common with tiny val sets)."""
    return max(results, key=lambda r: (r["metrics"]["f1"], r["metrics"]["accuracy"]))


def run_comparison() -> dict:
    splits = load_fusion_rows()
    X_train, y_train = rows_to_xy(splits["train"])
    X_val, y_val = rows_to_xy(splits["val"])

    print(f"Fusion training set: {len(y_train)} rows, {len(feature_names())} features")
    print(f"Fusion validation set: {len(y_val)} rows")
    print(f"Feature order: {feature_names()}")
    print("-" * 60)

    results = []
    for name, build_fn in CANDIDATE_MODELS.items():
        result = train_and_evaluate(name, build_fn, X_train, y_train, X_val, y_val)
        results.append(result)
        m = result["metrics"]
        print(
            f"{name:20} | acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
            f"rec={m['recall']:.3f} f1={m['f1']:.3f} auc={m['roc_auc']}"
        )

    best = select_best(results)
    print("-" * 60)
    print(f"Selected: {best['name']} (val_f1={best['metrics']['f1']:.3f})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best["model"], "name": best["name"], "feature_order": feature_names()}, BEST_MODEL_PATH)

    report = {
        "selected": best["name"],
        "comparison": [{"name": r["name"], "metrics": r["metrics"]} for r in results],
        "feature_order": feature_names(),
    }
    with open(COMPARISON_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved best model to {BEST_MODEL_PATH}")
    print(f"Saved comparison report to {COMPARISON_REPORT_PATH}")

    return report


if __name__ == "__main__":
    run_comparison()
