"""
TrueSight — fusion feature standardization.

Every stream produces its score independently and on its own schedule
(spatial/frequency/noise per-crop, blink/rppg/temporal only for video,
metadata per-file). Before fusion can combine them, they need to be
assembled into one fixed-order feature vector — this is the single place
that ordering is defined, matching configs/pipeline.yaml -> fusion.inputs.

Missing streams (most commonly: blink/rppg/temporal on image-only input,
since those require a video sequence) are filled with a neutral 0.5,
consistent with every stream's own internal "abstain" convention
established in Phases 7-8 (analyze_blinks, analyze_rppg,
aggregate_temporal_score all return 0.5 when they can't judge). This
keeps missing-vs-neutral-evidence handled the same way everywhere in the
system rather than fusion inventing its own convention.
"""

from __future__ import annotations

import numpy as np

from configs_loader import load_config

_config = load_config()
FEATURE_ORDER: list[str] = _config["fusion"]["inputs"]

NEUTRAL_DEFAULT = 0.5


def build_feature_vector(stream_scores: dict) -> np.ndarray:
    """stream_scores: dict that may contain any subset of FEATURE_ORDER
    keys (e.g. {"spatial_score": 0.8, "frequency_score": 0.6, ...}).
    Missing keys are filled with NEUTRAL_DEFAULT.

    Returns a 1D array in the fixed FEATURE_ORDER, ready to feed a
    fusion model.
    """
    return np.array(
        [float(stream_scores.get(key, NEUTRAL_DEFAULT)) for key in FEATURE_ORDER],
        dtype=np.float32,
    )


def build_feature_matrix(rows: list[dict]) -> np.ndarray:
    """Batched version of build_feature_vector for training/evaluation."""
    return np.stack([build_feature_vector(row) for row in rows])


def feature_names() -> list[str]:
    return list(FEATURE_ORDER)
