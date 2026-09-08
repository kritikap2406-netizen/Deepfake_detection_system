"""
Unit tests for Phase 5 — Frequency Stream.

Run with:
    pytest tests/test_frequency_stream.py -v
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import pytest

from frequency_noise.dct import compute_dct, dct_log_magnitude_normalized, prepare_dct_input
from frequency_noise.fft import compute_fft_magnitude, fft_log_magnitude_normalized, azimuthal_average, prepare_fft_input
from frequency_noise.frequency_model import build_model, FrequencyClassifier
from frequency_noise.dataset import FrequencyDataset, prepare_frequency_input
from frequency_noise.inference import FrequencyInference
from preprocessing.face_alignment import align_from_bbox
from preprocessing.face_detector import BoundingBox
from preprocessing.frame_extractor import load_single_image

ASSETS = Path("/home/claude/test_assets")
CHECKPOINT = Path("models/frequency/frequency_model.pt")


# --- DCT ---

def test_compute_dct_rejects_color_image():
    color = np.zeros((224, 224, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_dct(color)


def test_dct_log_magnitude_normalized_range():
    img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    out = dct_log_magnitude_normalized(img)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.shape == (224, 224)


def test_dct_constant_image_does_not_crash():
    """A flat/constant image has near-zero DCT energy spread -- edge case
    for the min-max normalization divide-by-near-zero guard."""
    flat = np.full((224, 224), 128, dtype=np.uint8)
    out = dct_log_magnitude_normalized(flat)
    assert not np.isnan(out).any()


def test_prepare_dct_input_resizes_nonstandard_input():
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    out = prepare_dct_input(img, size=224)
    assert out.shape == (224, 224)


# --- FFT ---

def test_fft_magnitude_shape():
    img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    mag = compute_fft_magnitude(img)
    assert mag.shape == (224, 224)
    assert (mag >= 0).all()  # magnitude is always non-negative


def test_fft_log_magnitude_normalized_range():
    img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    out = fft_log_magnitude_normalized(img)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_azimuthal_average_shape():
    img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    mag = compute_fft_magnitude(img)
    profile = azimuthal_average(mag, num_bins=50)
    assert profile.shape == (50,)


def test_azimuthal_average_low_freq_dominates_natural_image():
    """Center (low frequency) bin should carry more energy than the
    outermost (high frequency) bin for a typical natural-ish image."""
    img = cv2.imread(str(ASSETS / "synthetic_face.jpg"))
    gray = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2GRAY)
    mag = compute_fft_magnitude(gray)
    profile = azimuthal_average(mag, num_bins=50)
    assert profile[0] > profile[-1]


def test_prepare_fft_input_resizes_nonstandard_input():
    img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    out = prepare_fft_input(img, size=224)
    assert out.shape == (224, 224)


# --- Model ---

def test_frequency_model_forward_shape():
    model = build_model()
    x = torch.randn(4, 2, 224, 224)
    logits = model(x)
    assert logits.shape == (4,)


def test_frequency_model_predict_proba_range():
    model = build_model()
    x = torch.randn(4, 2, 224, 224)
    probs = model.predict_proba(x)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


def test_frequency_model_is_lightweight():
    """Sanity check that this stays a small from-scratch model, not
    something that silently grew to spatial-model size."""
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 1_000_000


def test_frequency_model_handles_variable_input_size():
    """AdaptiveAvgPool2d should make the model robust to non-224 input,
    even though the pipeline standardizes to 224 upstream."""
    model = build_model()
    x = torch.randn(2, 2, 112, 112)
    logits = model(x)
    assert logits.shape == (2,)


# --- Dataset ---

def test_prepare_frequency_input_shape_and_range():
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    tensor = prepare_frequency_input(img)
    assert tensor.shape == (2, 224, 224)
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_frequency_dataset_getitem_shape():
    ds = FrequencyDataset(split="test")
    tensor, label, meta = ds[0]
    assert tensor.shape == (2, 224, 224)
    assert label.item() in (0.0, 1.0)
    assert "dataset" in meta


# --- Inference ---

def test_frequency_inference_requires_existing_checkpoint():
    with pytest.raises(FileNotFoundError):
        FrequencyInference(checkpoint_path=Path("models/frequency/does_not_exist.pt"))


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m frequency_noise.train` first")
def test_frequency_inference_produces_contract_shape():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    face = align_from_bbox(frame.image, bbox, face_id=3)

    engine = FrequencyInference()
    result = engine.score_face(face)

    assert set(result.keys()) == {"face_id", "frequency_score"}
    assert result["face_id"] == 3
    assert 0.0 <= result["frequency_score"] <= 1.0


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m frequency_noise.train` first")
def test_frequency_inference_batch_matches_single():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    face = align_from_bbox(frame.image, bbox, face_id=1)

    engine = FrequencyInference()
    single = engine.score_face(face)
    batch = engine.score_faces([face])[0]
    assert single == batch


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m frequency_noise.train` first")
def test_frequency_inference_handles_empty_list():
    engine = FrequencyInference()
    assert engine.score_faces([]) == []
