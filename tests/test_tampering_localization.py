"""
Unit tests for Phase 13 — Tampering Localization.

Run with:
    pytest tests/test_tampering_localization.py -v

Uses a lightweight stub explainer (matching the FaceExplainer protocol)
for pure association-logic tests, and the real ExplainabilityInference
engine for a full end-to-end integration check — the same two-tier
approach used in Phase 8's temporal_orchestration tests.
"""

from pathlib import Path

import numpy as np
import pytest

from explainability.tampering_localization import localize_face, localize_faces
from explainability.inference import ExplainabilityInference
from preprocessing.face_alignment import AlignedFace
from preprocessing.face_detector import BoundingBox
from preprocessing.frame_extractor import load_single_image
from preprocessing.face_alignment import align_from_bbox

ASSETS = Path("/home/claude/test_assets")
CHECKPOINT = Path("models/spatial/efficientnet_b0.pt")


class StubExplainer:
    """Matches the FaceExplainer protocol without loading a real model —
    lets association logic be tested in isolation from Grad-CAM/model
    correctness, which is already covered by tests/test_explainability.py."""

    def __init__(self):
        self.calls = []

    def explain_face(self, face: AlignedFace, spatial_score) -> dict:
        self.calls.append((face.face_id, spatial_score))
        return {
            "heatmap_path": f"/some/container/path/face_{face.face_id}_heatmap.png",
            "suspicious_regions": ["mouth"] if face.face_id == 1 else [],
            "explanation": [f"stub explanation for face {face.face_id}"],
        }


def _make_face(face_id: int) -> AlignedFace:
    crop = np.zeros((224, 224, 3), dtype=np.uint8)
    return AlignedFace(face_id=face_id, image=crop, source_bbox=BoundingBox(0, 0, 10, 10, 1.0), landmarks=None)


# --- localize_face (single) ---

def test_localize_face_strips_path_to_filename_only():
    face = _make_face(1)
    stub = StubExplainer()
    record = localize_face(face, spatial_score=0.8, explainer=stub)
    assert record["heatmap"] == "face_1_heatmap.png"
    assert "/some/container/path" not in record["heatmap"]


def test_localize_face_passes_spatial_score_through_to_explainer():
    face = _make_face(1)
    stub = StubExplainer()
    localize_face(face, spatial_score=0.42, explainer=stub)
    assert stub.calls == [(1, 0.42)]


def test_localize_face_output_shape():
    face = _make_face(1)
    stub = StubExplainer()
    record = localize_face(face, spatial_score=0.5, explainer=stub)
    assert set(record.keys()) == {"face_id", "heatmap", "suspicious_regions", "explanation"}


# --- localize_faces (batch, multi-face association) ---

def test_localize_faces_correctly_associates_each_face_id():
    faces = [_make_face(1), _make_face(2), _make_face(3)]
    scores = {1: 0.9, 2: 0.1, 3: 0.5}
    stub = StubExplainer()

    results = localize_faces(faces, scores, stub)

    assert [r["face_id"] for r in results] == [1, 2, 3]
    assert stub.calls == [(1, 0.9), (2, 0.1), (3, 0.5)]


def test_localize_faces_each_face_gets_its_own_heatmap_filename():
    faces = [_make_face(1), _make_face(2)]
    scores = {1: 0.5, 2: 0.5}
    stub = StubExplainer()

    results = localize_faces(faces, scores, stub)
    filenames = [r["heatmap"] for r in results]

    assert filenames == ["face_1_heatmap.png", "face_2_heatmap.png"]
    assert len(set(filenames)) == 2  # no accidental collision


def test_localize_faces_missing_spatial_score_falls_back_to_none():
    faces = [_make_face(5)]
    scores = {}  # no entry for face_id 5
    stub = StubExplainer()

    localize_faces(faces, scores, stub)
    assert stub.calls == [(5, None)]


def test_localize_faces_preserves_per_face_suspicious_regions():
    """Confirms face 1's flagged region doesn't leak onto face 2's
    result — a plausible bug if association logic accidentally shared
    state between faces."""
    faces = [_make_face(1), _make_face(2)]
    scores = {1: 0.9, 2: 0.9}
    stub = StubExplainer()

    results = localize_faces(faces, scores, stub)
    by_id = {r["face_id"]: r for r in results}

    assert by_id[1]["suspicious_regions"] == ["mouth"]
    assert by_id[2]["suspicious_regions"] == []


def test_localize_faces_empty_input_returns_empty_list():
    stub = StubExplainer()
    assert localize_faces([], {}, stub) == []


# --- Full integration with the real explainability engine ---

@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_localize_faces_end_to_end_with_real_engine():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    face1 = align_from_bbox(frame.image, bbox, face_id=10)
    face2 = align_from_bbox(frame.image, bbox, face_id=20)

    engine = ExplainabilityInference()
    results = localize_faces([face1, face2], {10: 0.8, 20: 0.2}, engine)

    assert [r["face_id"] for r in results] == [10, 20]
    for r in results:
        assert r["heatmap"].endswith("_heatmap.png")
        assert "/" not in r["heatmap"]  # filename only, no path leakage
        assert isinstance(r["suspicious_regions"], list)
        assert isinstance(r["explanation"], list)
