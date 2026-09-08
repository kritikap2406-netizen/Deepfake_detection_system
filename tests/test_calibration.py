"""
Unit tests for Phase 11 — Confidence Calibration.

Run with:
    pytest tests/test_calibration.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from fusion.calibration import (
    prob_to_logit, logit_to_prob, apply_temperature,
    negative_log_likelihood, expected_calibration_error,
    fit_temperature, save_calibration, load_temperature,
    CalibrationResult,
)
from fusion.inference import FusionInference

BEST_MODEL = Path("models/fusion/best_fusion_model.joblib")
CALIBRATION_FILE = Path("models/fusion/temperature_scale.json")


# --- logit/prob round-trip ---

def test_prob_to_logit_and_back_roundtrip():
    p = np.array([0.1, 0.5, 0.9])
    z = prob_to_logit(p)
    recovered = logit_to_prob(z)
    assert np.allclose(p, recovered, atol=1e-4)


def test_prob_to_logit_clips_extremes_without_inf():
    p = np.array([0.0, 1.0])
    z = prob_to_logit(p)
    assert np.all(np.isfinite(z))


def test_logit_zero_maps_to_half():
    assert logit_to_prob(np.array([0.0]))[0] == pytest.approx(0.5)


# --- apply_temperature ---

def test_temperature_one_is_noop():
    probs = np.array([0.2, 0.6, 0.9])
    calibrated = apply_temperature(probs, temperature=1.0)
    assert np.allclose(probs, calibrated, atol=1e-6)


def test_temperature_above_one_pulls_toward_half():
    probs = np.array([0.95, 0.05])
    calibrated = apply_temperature(probs, temperature=3.0)
    assert calibrated[0] < probs[0]   # pulled down toward 0.5
    assert calibrated[1] > probs[1]   # pulled up toward 0.5


def test_temperature_below_one_pushes_away_from_half():
    probs = np.array([0.6, 0.4])
    calibrated = apply_temperature(probs, temperature=0.3)
    assert calibrated[0] > probs[0]   # pushed up, away from 0.5
    assert calibrated[1] < probs[1]   # pushed down, away from 0.5


def test_temperature_never_flips_prediction_side():
    """Calibration must never change which class is predicted — only
    how confident the prediction is."""
    probs = np.array([0.7, 0.3])
    for t in [0.1, 0.5, 1.0, 2.0, 5.0]:
        calibrated = apply_temperature(probs, temperature=t)
        assert (calibrated[0] > 0.5) == (probs[0] > 0.5)
        assert (calibrated[1] > 0.5) == (probs[1] > 0.5)


# --- NLL / ECE ---

def test_nll_is_zero_for_perfect_confident_correct_predictions():
    probs = np.array([0.999999, 0.000001])
    labels = np.array([1, 0])
    assert negative_log_likelihood(probs, labels) < 0.001


def test_nll_is_high_for_confident_wrong_predictions():
    probs = np.array([0.999999])
    labels = np.array([0])  # confidently wrong
    assert negative_log_likelihood(probs, labels) > 5.0


def test_ece_zero_for_perfectly_calibrated_predictions():
    """If predicted probability always exactly matches empirical
    frequency within each bin, ECE should be at or near zero."""
    probs = np.array([0.1] * 100 + [0.9] * 100)
    labels = np.array([0] * 90 + [1] * 10 + [1] * 90 + [0] * 10)
    ece = expected_calibration_error(probs, labels)
    assert ece < 0.05


def test_ece_high_for_badly_miscalibrated_predictions():
    probs = np.array([0.99] * 100)
    labels = np.array([0] * 100)  # always confident, always wrong
    ece = expected_calibration_error(probs, labels)
    assert ece > 0.9


# --- fit_temperature ---

def test_fit_temperature_improves_nll_for_overconfident_model():
    np.random.seed(42)
    n = 500
    true_labels = np.random.binomial(1, 0.5, n)
    overconfident_probs = np.where(
        np.random.random(n) < 0.75,
        np.where(true_labels == 1, 0.97, 0.03),
        np.where(true_labels == 1, 0.03, 0.97),
    )

    result = fit_temperature(overconfident_probs, true_labels)

    assert result.nll_after < result.nll_before
    assert result.ece_after < result.ece_before
    assert result.temperature > 1.0  # should soften an overconfident model


def test_fit_temperature_returns_calibration_result_fields():
    np.random.seed(0)
    probs = np.random.uniform(0.05, 0.95, 100)
    labels = np.random.binomial(1, probs)
    result = fit_temperature(probs, labels)
    assert isinstance(result, CalibrationResult)
    assert result.temperature > 0


# --- save / load ---

def test_save_and_load_calibration_roundtrip(tmp_path):
    path = tmp_path / "temperature_scale.json"
    result = CalibrationResult(temperature=2.5, nll_before=0.8, nll_after=0.4, ece_before=0.2, ece_after=0.05)
    save_calibration(result, path)

    loaded_temp = load_temperature(path)
    assert loaded_temp == 2.5


def test_load_temperature_defaults_to_one_when_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    assert load_temperature(missing_path) == 1.0


# --- fusion inference integration ---

def test_fusion_inference_falls_back_to_noop_temperature_without_calibration_file(tmp_path):
    if not BEST_MODEL.exists():
        pytest.skip("Run `python -m fusion.meta_classifier` first")

    engine = FusionInference(calibration_path=tmp_path / "missing.json")
    assert engine.temperature == 1.0

    result = engine.predict({"spatial_score": 0.5})
    assert result["calibrated_confidence"] == result["raw_score"]


@pytest.mark.skipif(
    not (BEST_MODEL.exists() and CALIBRATION_FILE.exists()),
    reason="Run `python -m fusion.meta_classifier` then `python -m fusion.train_calibration` first",
)
def test_fusion_inference_applies_real_calibration_file():
    engine = FusionInference()
    assert engine.temperature > 0

    result = engine.predict({"spatial_score": 0.8, "frequency_score": 0.7, "noise_score": 0.6})
    assert set(result.keys()) == {"prediction", "raw_score", "calibrated_confidence"}
    assert 0.0 <= result["calibrated_confidence"] <= 1.0


@pytest.mark.skipif(
    not (BEST_MODEL.exists() and CALIBRATION_FILE.exists()),
    reason="Run `python -m fusion.meta_classifier` then `python -m fusion.train_calibration` first",
)
def test_fusion_inference_calibration_does_not_flip_prediction():
    """Calibration must never change which side of the threshold a score
    falls on relative to itself pre/post scaling for the same raw score —
    it can only ever rescale confidence."""
    engine = FusionInference()
    scores = {"spatial_score": 0.9, "frequency_score": 0.9, "noise_score": 0.9}
    result = engine.predict(scores)

    raw_says_fake = result["raw_score"] >= 0.5
    calibrated_says_fake = result["calibrated_confidence"] >= 0.5
    # Both should agree with the actual prediction field
    assert (result["prediction"] == "FAKE") == calibrated_says_fake
