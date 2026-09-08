"""
Unit tests for Phase 19 — Test-Time Augmentation.

Run with:
    pytest tests/test_tta.py -v
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from spatial.tta import (
    horizontal_flip, small_crop, compression_variation,
    generate_tta_variants, TTASpatialInference,
)
from spatial.inference import SpatialInference
from preprocessing.face_alignment import align_from_bbox
from preprocessing.face_detector import BoundingBox
from preprocessing.frame_extractor import load_single_image

ASSETS = Path("/home/claude/test_assets")
CHECKPOINT = Path("models/spatial/efficientnet_b0.pt")


@pytest.fixture(scope="module")
def sample_image():
    img = cv2.imread(str(ASSETS / "synthetic_face.jpg"))
    return cv2.resize(img, (224, 224))


# --- Individual transforms ---

def test_horizontal_flip_preserves_shape(sample_image):
    out = horizontal_flip(sample_image)
    assert out.shape == sample_image.shape


def test_horizontal_flip_is_genuinely_mirrored(sample_image):
    flipped = horizontal_flip(sample_image)
    double_flipped = horizontal_flip(flipped)
    assert np.array_equal(double_flipped, sample_image)  # flip twice = identity
    assert not np.array_equal(flipped, sample_image)      # flip once != identity


def test_small_crop_preserves_shape(sample_image):
    out = small_crop(sample_image)
    assert out.shape == sample_image.shape


def test_small_crop_changes_content(sample_image):
    out = small_crop(sample_image, crop_ratio=0.7)
    assert not np.array_equal(out, sample_image)


def test_compression_variation_preserves_shape(sample_image):
    out = compression_variation(sample_image)
    assert out.shape == sample_image.shape


# --- Variant generation ---

def test_generate_tta_variants_returns_all_four_named_variants(sample_image):
    variants = generate_tta_variants(sample_image)
    assert set(variants.keys()) == {"original", "horizontal_flip", "small_crop", "compression_variation"}


def test_generate_tta_variants_original_is_unchanged(sample_image):
    variants = generate_tta_variants(sample_image)
    assert np.array_equal(variants["original"], sample_image)


def test_generate_tta_variants_all_same_shape(sample_image):
    variants = generate_tta_variants(sample_image)
    for name, v in variants.items():
        assert v.shape == sample_image.shape, f"{name} changed shape"


def test_generate_tta_variants_augmented_ones_differ_from_original(sample_image):
    variants = generate_tta_variants(sample_image)
    for name in ("horizontal_flip", "small_crop", "compression_variation"):
        assert not np.array_equal(variants[name], sample_image), f"{name} is a no-op"


# --- TTASpatialInference (requires trained checkpoint) ---

@pytest.fixture(scope="module")
def test_face():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    return align_from_bbox(frame.image, bbox, face_id=1)


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_tta_inference_produces_expected_keys(test_face):
    engine = TTASpatialInference()
    result = engine.score_face_tta(test_face)
    assert set(result.keys()) == {"face_id", "spatial_score", "tta_variants", "tta_std"}


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_tta_inference_variants_cover_all_four(test_face):
    engine = TTASpatialInference()
    result = engine.score_face_tta(test_face)
    assert set(result["tta_variants"].keys()) == {"original", "horizontal_flip", "small_crop", "compression_variation"}


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_tta_aggregated_score_is_mean_of_variants(test_face):
    engine = TTASpatialInference()
    result = engine.score_face_tta(test_face)
    manual_mean = sum(result["tta_variants"].values()) / len(result["tta_variants"])
    assert abs(manual_mean - result["spatial_score"]) < 0.001


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_tta_original_variant_matches_plain_inference_exactly(test_face):
    """The most important correctness check: TTA's 'original' variant
    must be bit-for-bit consistent with plain SpatialInference on the
    same input, proving TTA is a genuine superset rather than an
    accidentally different code path."""
    tta_engine = TTASpatialInference()
    plain_engine = SpatialInference()

    tta_result = tta_engine.score_face_tta(test_face)
    plain_result = plain_engine.score_face(test_face)

    assert tta_result["tta_variants"]["original"] == plain_result["spatial_score"]


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_tta_std_is_nonnegative(test_face):
    engine = TTASpatialInference()
    result = engine.score_face_tta(test_face)
    assert result["tta_std"] >= 0.0


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_tta_identical_inputs_produce_identical_results(test_face):
    engine = TTASpatialInference()
    face2 = align_from_bbox(
        load_single_image(str(ASSETS / "synthetic_face.jpg")).image,
        test_face.source_bbox, face_id=2,
    )
    results = engine.score_faces_tta([test_face, face2])
    assert results[0]["spatial_score"] == results[1]["spatial_score"]
    assert results[0]["face_id"] == 1
    assert results[1]["face_id"] == 2


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_tta_handles_empty_face_list():
    engine = TTASpatialInference()
    assert engine.score_faces_tta([]) == []


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_tta_output_compatible_with_fusion_feature_contract(test_face):
    """fusion/features.py only reads the 'spatial_score' key -- confirm
    TTA's richer output still satisfies that contract as a drop-in."""
    from fusion.features import build_feature_vector

    engine = TTASpatialInference()
    result = engine.score_face_tta(test_face)

    vector = build_feature_vector({"spatial_score": result["spatial_score"]})
    assert vector[0] == result["spatial_score"]
