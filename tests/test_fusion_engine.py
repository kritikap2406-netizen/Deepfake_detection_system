"""
Unit tests for Phase 10 — Fusion Engine.

Run with:
    pytest tests/test_fusion_engine.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from fusion.features import build_feature_vector, build_feature_matrix, feature_names, NEUTRAL_DEFAULT
from fusion.meta_classifier import load_fusion_rows, rows_to_xy, select_best
from fusion.inference import FusionInference

FUSION_DATASET = Path("datasets/fusion_dataset.csv")
BEST_MODEL = Path("models/fusion/best_fusion_model.joblib")


# --- Features ---

def test_feature_names_matches_config_order():
    names = feature_names()
    assert names == [
        "spatial_score", "frequency_score", "noise_score",
        "blink_score", "rppg_score", "temporal_score", "metadata_flag",
    ]


def test_build_feature_vector_full_scores():
    scores = {k: 0.7 for k in feature_names()}
    vec = build_feature_vector(scores)
    assert vec.shape == (7,)
    assert np.allclose(vec, 0.7)


def test_build_feature_vector_fills_missing_with_neutral():
    scores = {"spatial_score": 0.9}
    vec = build_feature_vector(scores)
    assert vec[0] == 0.9
    assert np.all(vec[1:] == NEUTRAL_DEFAULT)


def test_build_feature_vector_ignores_extra_keys():
    scores = {"spatial_score": 0.9, "face_id": 1, "unrelated_key": "x"}
    vec = build_feature_vector(scores)
    assert vec.shape == (7,)
    assert vec[0] == 0.9


def test_build_feature_matrix_stacks_rows():
    rows = [{"spatial_score": 0.1}, {"spatial_score": 0.9}]
    matrix = build_feature_matrix(rows)
    assert matrix.shape == (2, 7)
    assert matrix[0, 0] == 0.1
    assert matrix[1, 0] == 0.9


# --- Fusion dataset (requires build_dataset.py to have run) ---

@pytest.mark.skipif(not FUSION_DATASET.exists(), reason="Run `python -m fusion.build_dataset` first")
def test_fusion_dataset_has_all_splits():
    splits = load_fusion_rows()
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0


@pytest.mark.skipif(not FUSION_DATASET.exists(), reason="Run `python -m fusion.build_dataset` first")
def test_fusion_dataset_rows_to_xy_shapes():
    splits = load_fusion_rows()
    X, y = rows_to_xy(splits["train"])
    assert X.shape[1] == 7
    assert X.shape[0] == len(y)
    assert set(np.unique(y)) <= {0, 1}


@pytest.mark.skipif(not FUSION_DATASET.exists(), reason="Run `python -m fusion.build_dataset` first")
def test_fusion_dataset_video_only_streams_are_neutral():
    """blink/rppg/temporal aren't computed at this prototype stage (see
    fusion/build_dataset.py docstring) — confirm they're consistently
    the documented neutral default, not silently missing or garbage."""
    splits = load_fusion_rows()
    for row in splits["train"]:
        assert float(row["blink_score"]) == NEUTRAL_DEFAULT
        assert float(row["rppg_score"]) == NEUTRAL_DEFAULT
        assert float(row["temporal_score"]) == NEUTRAL_DEFAULT


# --- Model selection logic ---

def test_select_best_picks_highest_f1():
    results = [
        {"name": "a", "metrics": {"f1": 0.6, "accuracy": 0.9}},
        {"name": "b", "metrics": {"f1": 0.8, "accuracy": 0.5}},
        {"name": "c", "metrics": {"f1": 0.3, "accuracy": 0.95}},
    ]
    best = select_best(results)
    assert best["name"] == "b"


def test_select_best_breaks_f1_tie_with_accuracy():
    results = [
        {"name": "a", "metrics": {"f1": 0.7, "accuracy": 0.6}},
        {"name": "b", "metrics": {"f1": 0.7, "accuracy": 0.9}},
    ]
    best = select_best(results)
    assert best["name"] == "b"


# --- Fusion inference (requires meta_classifier.py to have run) ---

def test_fusion_inference_requires_existing_model():
    with pytest.raises(FileNotFoundError):
        FusionInference(model_path=Path("models/fusion/does_not_exist.joblib"))


@pytest.mark.skipif(not BEST_MODEL.exists(), reason="Run `python -m fusion.meta_classifier` first")
def test_fusion_inference_produces_contract_shape():
    engine = FusionInference()
    scores = {"spatial_score": 0.8, "frequency_score": 0.7, "noise_score": 0.6}
    result = engine.predict(scores)

    assert set(result.keys()) == {"prediction", "raw_score", "calibrated_confidence"}
    assert result["prediction"] in {"REAL", "FAKE"}
    assert 0.0 <= result["raw_score"] <= 1.0


@pytest.mark.skipif(not BEST_MODEL.exists(), reason="Run `python -m fusion.meta_classifier` first")
def test_fusion_inference_prediction_matches_threshold():
    engine = FusionInference()
    scores = {"spatial_score": 0.9, "frequency_score": 0.9, "noise_score": 0.9,
              "blink_score": 0.9, "rppg_score": 0.9, "temporal_score": 0.9, "metadata_flag": 0.9}
    result = engine.predict(scores)
    if result["raw_score"] >= 0.5:
        assert result["prediction"] == "FAKE"
    else:
        assert result["prediction"] == "REAL"


@pytest.mark.skipif(not BEST_MODEL.exists(), reason="Run `python -m fusion.meta_classifier` first")
def test_fusion_inference_calibration_is_now_wired_in():
    """Phase 10 originally left calibrated_confidence as a raw_score
    passthrough placeholder (with a test that would fail once Phase 11
    replaced it — see git history / phase 11 test_calibration.py). Phase
    11 has now wired in real temperature scaling via
    fusion/calibration.py, so this just confirms both fields are present
    and valid; the real calibration behavior is tested thoroughly in
    tests/test_calibration.py."""
    engine = FusionInference()
    result = engine.predict({"spatial_score": 0.5})
    assert "raw_score" in result
    assert "calibrated_confidence" in result
    assert 0.0 <= result["calibrated_confidence"] <= 1.0


@pytest.mark.skipif(not BEST_MODEL.exists(), reason="Run `python -m fusion.meta_classifier` first")
def test_fusion_inference_handles_missing_streams_gracefully():
    engine = FusionInference()
    result = engine.predict({"spatial_score": 0.5})  # only one of seven scores available
    assert set(result.keys()) == {"prediction", "raw_score", "calibrated_confidence"}
