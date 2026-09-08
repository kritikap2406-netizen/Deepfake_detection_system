"""
Unit tests for Phase 18 — Robustness Testing.

Run with:
    pytest tests/test_robustness.py -v

Scoping note: this phase covers degradation robustness (compression,
blur, noise, resolution, low-light) and adversarial FGSM/PGD, per the
development plan's Phase 18. Cross-dataset and unseen-manipulation
evaluation are covered more thoroughly in Phase 20's dedicated tests,
rather than duplicated shallowly here.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import pytest

from evaluation.degradations import (
    jpeg_compress, gaussian_blur, gaussian_noise, resize_degrade,
    low_light, social_media_compression, DEGRADATION_SUITE,
)
from evaluation.adversarial import fgsm_attack, pgd_attack, attack_success_rate
from evaluation.robustness_report import (
    evaluate_degradation_stability, evaluate_adversarial_robustness, generate_robustness_report,
)
from spatial.model import build_model

ASSETS = Path("/home/claude/test_assets")
CHECKPOINT = Path("models/spatial/efficientnet_b0.pt")


@pytest.fixture(scope="module")
def sample_image():
    img = cv2.imread(str(ASSETS / "synthetic_face.jpg"))
    return cv2.resize(img, (224, 224))


class TinyLinearModel(nn.Module):
    """Toy model with a real, non-negligible gradient signal — used to
    sanity-check FGSM/PGD attack correctness independent of whether any
    particular trained checkpoint happens to be numerically sensitive or
    insensitive to a given attack (see test_robustness_report tests
    below for that separate, real-checkpoint question)."""
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3, 224, 224) * 0.01)

    def forward(self, x):
        return (x * self.w).sum(dim=(1, 2, 3))


# --- Degradations ---

def test_jpeg_compress_preserves_shape(sample_image):
    out = jpeg_compress(sample_image, quality=20)
    assert out.shape == sample_image.shape


def test_gaussian_blur_preserves_shape(sample_image):
    out = gaussian_blur(sample_image, kernel_size=9)
    assert out.shape == sample_image.shape


def test_gaussian_blur_handles_even_kernel_size(sample_image):
    """cv2.GaussianBlur requires odd kernel sizes — confirm the function
    corrects this rather than crashing."""
    out = gaussian_blur(sample_image, kernel_size=8)
    assert out.shape == sample_image.shape


def test_gaussian_noise_changes_pixels(sample_image):
    out = gaussian_noise(sample_image, std=20.0)
    assert not np.array_equal(out, sample_image)
    assert out.shape == sample_image.shape


def test_resize_degrade_preserves_shape(sample_image):
    out = resize_degrade(sample_image, scale=0.25)
    assert out.shape == sample_image.shape


def test_low_light_darkens_image(sample_image):
    out = low_light(sample_image, brightness_factor=0.3)
    assert out.mean() < sample_image.mean()


def test_social_media_compression_preserves_shape(sample_image):
    out = social_media_compression(sample_image)
    assert out.shape == sample_image.shape


def test_degradation_suite_covers_expected_conditions():
    expected = {
        "jpeg_low_quality", "jpeg_medium_quality", "blur_mild", "blur_heavy",
        "noise_mild", "noise_heavy", "low_resolution", "low_light", "social_media_compression",
    }
    assert set(DEGRADATION_SUITE.keys()) == expected


def test_all_degradations_in_suite_run_without_crashing(sample_image):
    for name, fn in DEGRADATION_SUITE.items():
        out = fn(sample_image.copy())
        assert out.shape == sample_image.shape, f"{name} changed shape"


# --- Adversarial attacks (correctness sanity-checked on a toy model) ---

def test_fgsm_epsilon_zero_is_noop():
    model = TinyLinearModel()
    x = torch.randn(1, 3, 224, 224) * 0.1
    label = torch.tensor([1.0])
    adv = fgsm_attack(model, x, label, epsilon=0.0)
    assert torch.allclose(adv, x)


def test_fgsm_reliably_flips_predictions_on_sensitive_model():
    torch.manual_seed(0)
    model = TinyLinearModel()
    model.eval()

    images, labels = [], []
    for _ in range(15):
        x = torch.randn(1, 3, 224, 224) * 0.1
        with torch.no_grad():
            pred = (torch.sigmoid(model(x)) >= 0.5).float()
        images.append(x)
        labels.append(pred)

    result = attack_success_rate(model, images, labels, fgsm_attack, epsilon=0.05)
    assert result["attack_success_rate"] > 0.8


def test_pgd_reliably_flips_predictions_on_sensitive_model():
    torch.manual_seed(0)
    model = TinyLinearModel()
    model.eval()

    images, labels = [], []
    for _ in range(15):
        x = torch.randn(1, 3, 224, 224) * 0.1
        with torch.no_grad():
            pred = (torch.sigmoid(model(x)) >= 0.5).float()
        images.append(x)
        labels.append(pred)

    result = attack_success_rate(model, images, labels, pgd_attack, epsilon=0.05, alpha=0.0125, steps=10)
    assert result["attack_success_rate"] > 0.8


def test_pgd_stays_within_epsilon_ball():
    torch.manual_seed(0)
    model = TinyLinearModel()
    x = torch.randn(1, 3, 224, 224) * 0.1
    label = torch.tensor([1.0])

    adv = pgd_attack(model, x, label, epsilon=0.1, alpha=0.02, steps=10)
    max_perturbation = (adv - x).abs().max().item()
    assert max_perturbation <= 0.1 + 1e-5  # small tolerance for float rounding


def test_attack_success_rate_skips_already_misclassified_samples():
    """A sample the model already gets wrong shouldn't count toward
    n_correct_clean or be considered an attack 'success' if it flips."""
    model = TinyLinearModel()
    x = torch.randn(1, 3, 224, 224) * 0.1
    wrong_label = torch.tensor([1.0]) if (model(x) < 0).item() else torch.tensor([0.0])
    # Force a mismatch between clean prediction and provided label
    with torch.no_grad():
        clean_pred = (torch.sigmoid(model(x)) >= 0.5).float()
    mismatched_label = 1.0 - clean_pred

    result = attack_success_rate(model, [x], [mismatched_label], fgsm_attack, epsilon=0.05)
    assert result["n_correct_clean"] == 0
    assert result["attack_success_rate"] is None


# --- Robustness report (requires trained checkpoint) ---

@pytest.fixture(scope="module")
def real_model():
    if not CHECKPOINT.exists():
        pytest.skip("Run `python -m spatial.train` first")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(backbone=checkpoint["backbone"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@pytest.fixture(scope="module")
def sample_crops(sample_image):
    rng = np.random.default_rng(0)
    crops = [sample_image]
    for _ in range(5):
        variant = np.clip(sample_image.astype(np.float32) + rng.normal(0, 15, sample_image.shape), 0, 255).astype(np.uint8)
        crops.append(variant)
    return crops


def test_evaluate_degradation_stability_covers_all_conditions(real_model, sample_crops):
    results = evaluate_degradation_stability(real_model, sample_crops)
    assert set(results.keys()) == set(DEGRADATION_SUITE.keys())
    for name, metrics in results.items():
        assert set(metrics.keys()) == {"mean_abs_score_shift", "max_abs_score_shift", "prediction_flip_rate"}
        assert 0.0 <= metrics["mean_abs_score_shift"] <= 1.0
        assert 0.0 <= metrics["prediction_flip_rate"] <= 1.0


def test_evaluate_adversarial_robustness_structure(real_model, sample_crops):
    results = evaluate_adversarial_robustness(real_model, sample_crops, epsilons=[0.05])
    assert "fgsm" in results and "pgd" in results
    assert "epsilon_0.05" in results["fgsm"]
    assert "epsilon_0.05" in results["pgd"]


def test_generate_robustness_report_includes_caveat_for_small_samples(real_model, sample_crops):
    report = generate_robustness_report(real_model, sample_crops)
    assert report["n_samples"] == len(sample_crops)
    assert report["caveat"] is not None  # 6 samples < 10, caveat should fire
    assert "degradation_stability" in report
    assert "adversarial_robustness" in report


def test_generate_robustness_report_no_caveat_for_larger_samples(real_model, sample_image):
    rng = np.random.default_rng(1)
    crops = [
        np.clip(sample_image.astype(np.float32) + rng.normal(0, 15, sample_image.shape), 0, 255).astype(np.uint8)
        for _ in range(12)
    ]
    report = generate_robustness_report(real_model, crops)
    assert report["caveat"] is None
