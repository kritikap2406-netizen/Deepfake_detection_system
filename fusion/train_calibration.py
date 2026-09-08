"""
TrueSight — calibration training.

Phase 11: fit temperature scaling on the fusion meta-classifier's
validation-split predictions (never on train — calibration must be
fitted on held-out predictions the model didn't memorize, otherwise it
would just be measuring the model's overfitting rather than its true
confidence gap), compare raw vs calibrated NLL/ECE, and persist the
fitted temperature for fusion/inference.py to use.

Usage:
    python -m fusion.train_calibration
"""

from __future__ import annotations

import joblib
import numpy as np

from fusion.calibration import fit_temperature, save_calibration, DEFAULT_CALIBRATION_PATH
from fusion.meta_classifier import load_fusion_rows, rows_to_xy, BEST_MODEL_PATH


def train_calibration():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No fusion model at {BEST_MODEL_PATH}. Run `python -m fusion.meta_classifier` first."
        )

    saved = joblib.load(BEST_MODEL_PATH)
    model = saved["model"]

    splits = load_fusion_rows()
    X_val, y_val = rows_to_xy(splits["val"])

    if len(y_val) < 30:
        print(
            f"WARNING: validation split has only {len(y_val)} rows. Temperature "
            f"scaling needs a validation set large enough to contain genuine "
            f"misclassifications to correct for — with too few rows (especially "
            f"if the model happens to get all of them right), the fit can degrade "
            f"toward sharpening confidence instead of softening it, which is the "
            f"opposite of calibration's purpose. Treat the fitted temperature "
            f"below as a plumbing check, not a trustworthy value, until this is "
            f"re-run on a real validation split with real datasets."
        )

    raw_probs = model.predict_proba(X_val)[:, 1]

    result = fit_temperature(raw_probs, y_val)

    if result.temperature < 1.0:
        print(
            f"WARNING: fitted temperature ({result.temperature}) is below 1.0, "
            f"meaning it SHARPENS confidence rather than softening it. This is "
            f"the opposite of typical calibration behavior for an "
            f"overconfident model, and usually signals the validation set was "
            f"too small or too easy (see above) rather than a genuinely "
            f"well-calibrated or underconfident model."
        )

    save_calibration(result, DEFAULT_CALIBRATION_PATH)

    print(f"Fitted temperature: {result.temperature}")
    print(f"NLL: {result.nll_before} -> {result.nll_after}")
    print(f"ECE: {result.ece_before} -> {result.ece_after}")
    print(f"Saved calibration to {DEFAULT_CALIBRATION_PATH}")

    return result


if __name__ == "__main__":
    train_calibration()
