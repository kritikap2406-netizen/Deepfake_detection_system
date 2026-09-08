"""
Unit tests for Phase 6 — Noise Residual Stream.

Run with:
    pytest tests/test_noise_stream.py -v
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import pytest

from frequency_noise.srm import (
    apply_srm_filter, compute_srm_residuals, normalize_residual,
    prepare_srm_input, SRM_KERNELS,
)
from frequency_noise.noise_model import build_model, NoiseClassifier
from frequency_noise.dataset import NoiseDataset, prepare_noise_input
from frequency_noise.inference import NoiseInference, FrequencyInference
from preprocessing.face_alignment import align_from_bbox
from preprocessing.face_detector import BoundingBox
from preprocessing.frame_extractor import load_single_image

ASSETS = Path("/home/claude/test_assets")
CHECKPOINT = Path("models/noise/noise_model.pt")
FREQ_CHECKPOINT = Path("models/frequency/frequency_model.pt")


# --- SRM filters ---

def test_three_srm_kernels_defined():
    assert len(SRM_KERNELS) == 3
    for k in SRM_KERNELS:
        assert k.shape == (5, 5)


def test_apply_srm_filter_rejects_color_image():
    color = np.zeros((224, 224, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_srm_filter(color, SRM_KERNELS[0])


def test_compute_srm_residuals_shape():
    img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    residuals = compute_srm_residuals(img)
    assert residuals.shape == (3, 224, 224)


def test_srm_residuals_are_near_zero_mean():
    """SRM kernels are predictor filters (their coefficients sum to ~0),
    so on a fairly textured image the residual should be roughly
    zero-centered rather than systematically biased."""
    img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    residuals = compute_srm_residuals(img)
    for r in residuals:
        assert abs(r.mean()) < 20.0  # loose bound, just checks no gross bias


def test_normalize_residual_range():
    residual = np.random.normal(0, 30, (224, 224)).astype(np.float32)
    out = normalize_residual(residual)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_normalize_residual_constant_input_does_not_crash():
    flat = np.zeros((224, 224), dtype=np.float32)
    out = normalize_residual(flat)
    assert not np.isnan(out).any()


def test_prepare_srm_input_shape_and_range():
    img = cv2.imread(str(ASSETS / "synthetic_face.jpg"))
    out = prepare_srm_input(img)
    assert out.shape == (3, 224, 224)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_prepare_srm_input_resizes_nonstandard_input():
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    out = prepare_srm_input(img, size=224)
    assert out.shape == (3, 224, 224)


# --- Model ---

def test_noise_model_forward_shape():
    model = build_model()
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (4,)


def test_noise_model_predict_proba_range():
    model = build_model()
    x = torch.randn(4, 3, 224, 224)
    probs = model.predict_proba(x)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_noise_model_is_lightweight():
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 1_000_000


def test_noise_and_frequency_models_comparable_size():
    """Both streams should carry a similar computational budget so fusion
    isn't implicitly weighted toward whichever stream has a bigger model."""
    from frequency_noise.frequency_model import build_model as build_freq_model
    noise_params = sum(p.numel() for p in build_model().parameters())
    freq_params = sum(p.numel() for p in build_freq_model().parameters())
    ratio = max(noise_params, freq_params) / min(noise_params, freq_params)
    assert ratio < 1.5


# --- Dataset ---

def test_prepare_noise_input_shape():
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    tensor = prepare_noise_input(img)
    assert tensor.shape == (3, 224, 224)


def test_noise_dataset_getitem_shape():
    ds = NoiseDataset(split="test")
    tensor, label, meta = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert label.item() in (0.0, 1.0)
    assert "dataset" in meta


def test_noise_dataset_same_length_as_frequency_dataset():
    """Both streams read the same leakage-safe split, so their datasets
    should agree on split sizes exactly."""
    from frequency_noise.dataset import FrequencyDataset
    noise_ds = NoiseDataset(split="train")
    freq_ds = FrequencyDataset(split="train")
    assert len(noise_ds) == len(freq_ds)


# --- Inference ---

def test_noise_inference_requires_existing_checkpoint():
    with pytest.raises(FileNotFoundError):
        NoiseInference(checkpoint_path=Path("models/noise/does_not_exist.pt"))


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m frequency_noise.train_noise` first")
def test_noise_inference_produces_contract_shape():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    face = align_from_bbox(frame.image, bbox, face_id=5)

    engine = NoiseInference()
    result = engine.score_face(face)

    assert set(result.keys()) == {"face_id", "noise_score"}
    assert result["face_id"] == 5
    assert 0.0 <= result["noise_score"] <= 1.0


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m frequency_noise.train_noise` first")
def test_noise_inference_batch_matches_single():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    face = align_from_bbox(frame.image, bbox, face_id=1)

    engine = NoiseInference()
    single = engine.score_face(face)
    batch = engine.score_faces([face])[0]
    assert single == batch


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m frequency_noise.train_noise` first")
def test_noise_inference_handles_empty_list():
    engine = NoiseInference()
    assert engine.score_faces([]) == []


@pytest.mark.skipif(
    not (CHECKPOINT.exists() and FREQ_CHECKPOINT.exists()),
    reason="Requires both frequency and noise checkpoints",
)
def test_frequency_and_noise_inference_coexist():
    """Both engines share frequency_noise/inference.py -- confirm importing
    and instantiating both together doesn't cause cross-talk (e.g. shared
    mutable state, name clashes from the build_model refactor)."""
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    face = align_from_bbox(frame.image, bbox, face_id=1)

    freq_engine = FrequencyInference()
    noise_engine = NoiseInference()

    freq_result = freq_engine.score_face(face)
    noise_result = noise_engine.score_face(face)

    assert "frequency_score" in freq_result
    assert "noise_score" in noise_result
    assert "frequency_score" not in noise_result
    assert "noise_score" not in freq_result
