"""
Unit tests for Phase 12 — Explainable AI.

Run with:
    pytest tests/test_explainability.py -v
"""

from pathlib import Path

import numpy as np
import torch
import pytest

from spatial.model import build_model
from explainability.gradcam import GradCAMExtractor
from explainability.heatmap import cam_to_color_heatmap, overlay_heatmap, save_heatmap
from explainability.explanation import (
    compute_region_intensities, suspicious_regions, generate_textual_explanation,
    FACE_REGIONS, SUSPICIOUS_INTENSITY_THRESHOLD,
)
from explainability.inference import ExplainabilityInference
from preprocessing.face_alignment import align_from_bbox
from preprocessing.face_detector import BoundingBox
from preprocessing.frame_extractor import load_single_image

ASSETS = Path("/home/claude/test_assets")
CHECKPOINT = Path("models/spatial/efficientnet_b0.pt")


@pytest.fixture(scope="module")
def model():
    m = build_model(pretrained=False)
    m.eval()
    return m


# --- Grad-CAM / Grad-CAM++ ---

def test_gradcam_output_shape(model):
    extractor = GradCAMExtractor(model)
    x = torch.randn(1, 3, 224, 224)
    cam = extractor.compute_gradcam(x)
    assert cam.shape == (224, 224)


def test_gradcam_output_bounded(model):
    extractor = GradCAMExtractor(model)
    x = torch.randn(1, 3, 224, 224)
    cam = extractor.compute_gradcam(x)
    assert cam.min().item() >= 0.0
    assert cam.max().item() <= 1.0


def test_gradcam_plusplus_output_shape_and_bounds(model):
    extractor = GradCAMExtractor(model)
    x = torch.randn(1, 3, 224, 224)
    cam = extractor.compute_gradcam_plusplus(x)
    assert cam.shape == (224, 224)
    assert cam.min().item() >= 0.0
    assert cam.max().item() <= 1.0


def test_gradcam_rejects_batch_size_greater_than_one(model):
    extractor = GradCAMExtractor(model)
    x = torch.randn(2, 3, 224, 224)
    with pytest.raises(ValueError):
        extractor.compute_gradcam(x)


def test_gradcam_produces_nonzero_activation_across_multiple_inputs(model):
    """Regression check against a systematic all-zero bug — confirms
    Grad-CAM produces genuine, mostly-nonzero activation on typical
    inputs (an isolated all-zero result on one specific image is a
    legitimate possible outcome, not necessarily a bug, but it should
    not happen on every input)."""
    extractor = GradCAMExtractor(model)
    torch.manual_seed(0)
    nonzero_fracs = []
    for _ in range(5):
        x = torch.randn(1, 3, 224, 224)
        cam = extractor.compute_gradcam(x)
        nonzero_fracs.append((cam > 0).float().mean().item())
    assert max(nonzero_fracs) > 0.5


def test_target_layer_defaults_to_bn2(model):
    extractor = GradCAMExtractor(model)
    assert extractor.target_layer is model.backbone.bn2


# --- Heatmap ---

def test_cam_to_color_heatmap_shape():
    cam = torch.rand(224, 224)
    color = cam_to_color_heatmap(cam)
    assert color.shape == (224, 224, 3)
    assert color.dtype == np.uint8


def test_overlay_heatmap_matches_face_shape():
    face = np.full((224, 224, 3), 150, dtype=np.uint8)
    cam = torch.rand(224, 224)
    overlay = overlay_heatmap(face, cam)
    assert overlay.shape == face.shape


def test_overlay_heatmap_resizes_mismatched_cam():
    face = np.full((224, 224, 3), 150, dtype=np.uint8)
    small_cam = torch.rand(56, 56)  # different resolution than face
    overlay = overlay_heatmap(face, small_cam)
    assert overlay.shape == face.shape


def test_save_heatmap_writes_file(tmp_path):
    face = np.full((224, 224, 3), 150, dtype=np.uint8)
    cam = torch.rand(224, 224)
    path = save_heatmap(face, cam, filename="unit_test.png", output_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".png"


# --- Region localization ---

def test_compute_region_intensities_covers_all_regions():
    cam = np.random.rand(224, 224).astype(np.float32)
    ranked = compute_region_intensities(cam)
    assert len(ranked) == len(FACE_REGIONS)
    assert {r.region for r in ranked} == set(FACE_REGIONS.keys())


def test_compute_region_intensities_sorted_descending():
    cam = np.random.rand(224, 224).astype(np.float32)
    ranked = compute_region_intensities(cam)
    intensities = [r.mean_intensity for r in ranked]
    assert intensities == sorted(intensities, reverse=True)


def test_injected_hotspot_correctly_identified_as_top_region():
    cam = np.zeros((224, 224), dtype=np.float32)
    x0, y0, x1, y1 = FACE_REGIONS["chin"]
    px0, py0, px1, py1 = int(x0 * 224), int(y0 * 224), int(x1 * 224), int(y1 * 224)
    cam[py0:py1, px0:px1] = 1.0

    ranked = compute_region_intensities(cam)
    assert ranked[0].region == "chin"
    assert ranked[0].mean_intensity > 0.9


def test_suspicious_regions_respects_threshold():
    cam = np.zeros((224, 224), dtype=np.float32)
    x0, y0, x1, y1 = FACE_REGIONS["forehead"]
    px0, py0, px1, py1 = int(x0 * 224), int(y0 * 224), int(x1 * 224), int(y1 * 224)
    cam[py0:py1, px0:px1] = SUSPICIOUS_INTENSITY_THRESHOLD - 0.01  # just below threshold

    ranked = compute_region_intensities(cam)
    flagged = suspicious_regions(ranked)
    assert "forehead" not in flagged


def test_suspicious_regions_empty_for_flat_zero_cam():
    cam = np.zeros((224, 224), dtype=np.float32)
    ranked = compute_region_intensities(cam)
    assert suspicious_regions(ranked) == []


# --- Textual explanation ---

def test_explanation_includes_spatial_score_sentence_when_provided():
    cam = np.zeros((224, 224), dtype=np.float32)
    ranked = compute_region_intensities(cam)
    explanation = generate_textual_explanation(ranked, spatial_score=0.9)
    assert any("0.90" in line for line in explanation)


def test_explanation_omits_spatial_sentence_when_not_provided():
    cam = np.zeros((224, 224), dtype=np.float32)
    ranked = compute_region_intensities(cam)
    explanation = generate_textual_explanation(ranked, spatial_score=None)
    assert not any("Spatial analysis" in line for line in explanation)


def test_explanation_mentions_actual_flagged_region_name():
    cam = np.zeros((224, 224), dtype=np.float32)
    x0, y0, x1, y1 = FACE_REGIONS["nose"]
    px0, py0, px1, py1 = int(x0 * 224), int(y0 * 224), int(x1 * 224), int(y1 * 224)
    cam[py0:py1, px0:px1] = 0.9

    ranked = compute_region_intensities(cam)
    explanation = generate_textual_explanation(ranked)
    assert any("nose" in line for line in explanation)


def test_explanation_no_suspicion_message_when_nothing_flagged():
    cam = np.zeros((224, 224), dtype=np.float32)
    ranked = compute_region_intensities(cam)
    explanation = generate_textual_explanation(ranked)
    assert any("No single facial region" in line for line in explanation)


# --- Full inference ---

def test_explainability_inference_requires_existing_checkpoint():
    with pytest.raises(FileNotFoundError):
        ExplainabilityInference(checkpoint_path=Path("models/spatial/does_not_exist.pt"))


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_explainability_inference_produces_expected_keys(tmp_path):
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    face = align_from_bbox(frame.image, bbox, face_id=1)

    engine = ExplainabilityInference()
    result = engine.explain_face(face, spatial_score=0.8)

    assert set(result.keys()) == {"heatmap_path", "suspicious_regions", "explanation"}
    assert Path(result["heatmap_path"]).exists()
    assert isinstance(result["suspicious_regions"], list)
    assert isinstance(result["explanation"], list)
    assert len(result["explanation"]) > 0
