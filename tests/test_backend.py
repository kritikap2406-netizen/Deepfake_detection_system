"""
Unit tests for Phase 14 — Backend (FastAPI).

Run with:
    pytest tests/test_backend.py -v

Uses FastAPI's TestClient (no live server needed) for the HTTP-layer
tests, and unittest.mock.patch to bypass face detection where needed —
the same approach used to manually validate the orchestrator, since our
synthetic test image/video isn't detectable by the Haar cascade fallback
(a known, already-documented limitation from Phase 3 onward), while
every other stage of the pipeline runs for real.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.storage import save_result, load_result
from backend.services.orchestrator import analyze_image, analyze_video, engines_available, _aggregate_overall
from preprocessing.face_alignment import AlignedFace, align_from_bbox
from preprocessing.face_detector import BoundingBox
from preprocessing.frame_extractor import load_single_image
from preprocessing.pipeline import FrameFaces

ASSETS = Path("/home/claude/test_assets")
FUSION_MODEL = Path("models/fusion/best_fusion_model.joblib")

client = TestClient(app)


# --- Storage ---

def test_storage_save_and_load_roundtrip(tmp_path):
    db_path = tmp_path / "test.db"
    save_result("id1", {"prediction": "FAKE"}, db_path=db_path)
    result = load_result("id1", db_path=db_path)
    assert result == {"prediction": "FAKE"}


def test_storage_missing_id_returns_none(tmp_path):
    db_path = tmp_path / "test.db"
    assert load_result("nonexistent", db_path=db_path) is None


def test_storage_overwrites_existing_id(tmp_path):
    db_path = tmp_path / "test.db"
    save_result("id1", {"prediction": "FAKE"}, db_path=db_path)
    save_result("id1", {"prediction": "REAL"}, db_path=db_path)
    assert load_result("id1", db_path=db_path) == {"prediction": "REAL"}


# --- Aggregation policy ---

def test_aggregate_overall_any_fake_face_means_overall_fake():
    verdicts = [
        {"prediction": "REAL", "calibrated_confidence": 0.1},
        {"prediction": "FAKE", "calibrated_confidence": 0.9},
    ]
    prediction, confidence = _aggregate_overall(verdicts)
    assert prediction == "FAKE"
    assert confidence == 0.9


def test_aggregate_overall_all_real_gives_real():
    verdicts = [
        {"prediction": "REAL", "calibrated_confidence": 0.2},
        {"prediction": "REAL", "calibrated_confidence": 0.3},
    ]
    prediction, confidence = _aggregate_overall(verdicts)
    assert prediction == "REAL"
    assert confidence == round(1.0 - 0.3, 4)


def test_aggregate_overall_fake_confidence_is_max_among_fake_faces():
    verdicts = [
        {"prediction": "FAKE", "calibrated_confidence": 0.6},
        {"prediction": "FAKE", "calibrated_confidence": 0.95},
        {"prediction": "REAL", "calibrated_confidence": 0.1},
    ]
    prediction, confidence = _aggregate_overall(verdicts)
    assert prediction == "FAKE"
    assert confidence == 0.95


# --- Orchestrator (requires trained checkpoints) ---

@pytest.mark.skipif(not FUSION_MODEL.exists(), reason="Requires all trained checkpoints")
def test_analyze_image_no_face_detected_shape():
    result = analyze_image(str(ASSETS / "synthetic_face.jpg"))
    assert result["prediction"] == "UNKNOWN"
    assert "message" in result


@pytest.mark.skipif(not FUSION_MODEL.exists(), reason="Requires all trained checkpoints")
def test_analyze_image_success_path_produces_full_contract():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    face = align_from_bbox(frame.image, bbox, face_id=1)

    with patch("backend.services.orchestrator.preprocess_image", return_value=[face]):
        result = analyze_image(str(ASSETS / "synthetic_face.jpg"))

    assert result["prediction"] in {"REAL", "FAKE"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["faces"]) == 1
    face_record = result["faces"][0]
    expected_keys = {
        "face_id", "spatial_score", "frequency_score", "noise_score",
        "blink_score", "rppg_score", "temporal_score", "heatmap", "suspicious_regions",
    }
    assert set(face_record.keys()) == expected_keys
    assert isinstance(result["explanation"], list) and len(result["explanation"]) > 0


@pytest.mark.skipif(not FUSION_MODEL.exists(), reason="Requires all trained checkpoints")
def test_analyze_video_success_path_produces_full_contract():
    frame_faces = []
    for i in range(40):
        crop = np.full((224, 224, 3), 180, dtype=np.uint8)
        face = AlignedFace(face_id=1, image=crop, source_bbox=BoundingBox(0, 0, 224, 224, 1.0), landmarks=None)
        frame_faces.append(FrameFaces(frame_index=i, timestamp_sec=i / 10, faces=[face]))

    with patch("backend.services.orchestrator.preprocess_video", return_value=frame_faces):
        result = analyze_video(str(ASSETS / "synthetic_video.mp4"), sample_rate_fps=10.0)

    assert result["prediction"] in {"REAL", "FAKE"}
    assert len(result["faces"]) == 1
    # video path should populate real (non-neutral-default) physiological/temporal
    # values rather than leaving them at the image-path's 0.5 placeholder
    face_record = result["faces"][0]
    assert isinstance(face_record["blink_score"], float)
    assert isinstance(face_record["temporal_score"], float)


def test_engines_available_returns_bool_dict():
    availability = engines_available()
    assert set(availability.keys()) == {"spatial", "frequency", "noise", "fusion"}
    assert all(isinstance(v, bool) for v in availability.values())


# --- HTTP layer (TestClient) ---

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"ok", "degraded"}
    assert "models_loaded" in data


def test_analyze_image_rejects_wrong_file_type():
    with open(ASSETS / "synthetic_video.mp4", "rb") as f:
        response = client.post("/analyze/image", files={"file": ("test.mp4", f, "video/mp4")})
    assert response.status_code == 400


def test_analyze_video_rejects_wrong_file_type():
    with open(ASSETS / "synthetic_face.jpg", "rb") as f:
        response = client.post("/analyze/video", files={"file": ("test.jpg", f, "image/jpeg")})
    assert response.status_code == 400


def test_analyze_image_rejects_empty_file():
    response = client.post("/analyze/image", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400


@pytest.mark.skipif(not FUSION_MODEL.exists(), reason="Requires all trained checkpoints")
def test_analyze_image_endpoint_no_face_returns_unknown():
    with open(ASSETS / "synthetic_face.jpg", "rb") as f:
        response = client.post("/analyze/image", files={"file": ("test.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "UNKNOWN"


def test_result_endpoint_404_for_unknown_id():
    response = client.get("/result/definitely-does-not-exist")
    assert response.status_code == 404


def test_result_endpoint_returns_previously_saved_result():
    save_result("test-http-roundtrip", {
        "id": "test-http-roundtrip", "media_type": "image", "prediction": "REAL",
        "confidence": 0.6, "faces": [], "explanation": [], "metadata_flag": 0.0,
    })
    response = client.get("/result/test-http-roundtrip")
    assert response.status_code == 200
    assert response.json()["prediction"] == "REAL"


def test_openapi_schema_exposes_all_endpoints():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/health" in paths
    assert "/analyze/image" in paths
    assert "/analyze/video" in paths
    assert "/result/{result_id}" in paths
