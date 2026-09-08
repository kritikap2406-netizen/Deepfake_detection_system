"""
TrueSight — fusion inference.

Loads whichever model Phase 10's comparison selected as best and scores
a set of per-stream evidence into the final fusion_output contract from
configs/pipeline.yaml:

    {"prediction": "REAL"|"FAKE", "raw_score": float, "calibrated_confidence": float}

`calibrated_confidence` now applies Phase 11's fitted temperature scaling
to `raw_score` (see fusion/calibration.py). If calibration hasn't been
run yet (no models/fusion/temperature_scale.json present), temperature
defaults to 1.0 — a no-op — so this module still works before Phase 11's
training step, it just isn't calibrated until that's been run.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from configs_loader import load_config
from fusion.calibration import apply_temperature, load_temperature, DEFAULT_CALIBRATION_PATH
from fusion.features import build_feature_vector

_config = load_config()
FAKE_THRESHOLD = _config["decision"]["fake_threshold"]

DEFAULT_MODEL_PATH = Path("models/fusion/best_fusion_model.joblib")


class FusionInference:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No fusion model at {model_path}. "
                f"Run `python -m fusion.build_dataset` then `python -m fusion.meta_classifier` first."
            )

        saved = joblib.load(model_path)
        self.model = saved["model"]
        self.model_name = saved["name"]
        self.feature_order = saved["feature_order"]
        self.temperature = load_temperature(calibration_path)

    def predict(self, stream_scores: dict) -> dict:
        """stream_scores: dict of any subset of the fusion feature keys
        (spatial_score, frequency_score, noise_score, blink_score,
        rppg_score, temporal_score, metadata_flag). Missing keys default
        to neutral 0.5 — see fusion/features.py.
        """
        vector = build_feature_vector(stream_scores).reshape(1, -1)
        raw_score = float(self.model.predict_proba(vector)[0, 1])  # P(fake)

        calibrated = float(apply_temperature(raw_score, self.temperature))
        prediction = "FAKE" if calibrated >= FAKE_THRESHOLD else "REAL"

        return {
            "prediction": prediction,
            "raw_score": round(raw_score, 4),
            "calibrated_confidence": round(calibrated, 4),
        }
