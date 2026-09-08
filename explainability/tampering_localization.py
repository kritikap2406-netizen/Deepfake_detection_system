"""
TrueSight — tampering localization.

Phase 12 built the mechanics: Grad-CAM++ per face crop, region-intensity
ranking, evidence-grounded text. Phase 13 is the orchestration layer on
top of that: given potentially several detected faces in one image (or
one video frame), associate each face's localization evidence with its
own face_id, and shape the result to match exactly what the final API
response needs (see the development plan's "Final API Response Example"):

    {
        "face_id": 1,
        "heatmap": "face_1_heatmap.png",
        "suspicious_regions": ["left_eye", "cheek"],
        ...
    }

Note `heatmap` here is a filename, not a full filesystem path — the
backend (Phase 14) will serve heatmap images from a static directory, so
the API response should reference them the way a client will actually
request them (e.g. GET /static/heatmaps/face_1_heatmap.png), not leak
the container's local filesystem layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from preprocessing.face_alignment import AlignedFace


class FaceExplainer(Protocol):
    """Structural interface satisfied by explainability.inference.ExplainabilityInference.
    Defined explicitly so tests can inject a lightweight stub instead of
    loading a real model checkpoint — the same pattern used in Phase 8's
    temporal_orchestration.py.
    """
    def explain_face(self, face: AlignedFace, spatial_score: float | None) -> dict: ...


def localize_face(face: AlignedFace, spatial_score: float | None, explainer: FaceExplainer) -> dict:
    """Run explainability for one face and shape it for the API."""
    result = explainer.explain_face(face, spatial_score=spatial_score)

    return {
        "face_id": face.face_id,
        "heatmap": Path(result["heatmap_path"]).name,  # filename only — see module docstring
        "suspicious_regions": result["suspicious_regions"],
        "explanation": result["explanation"],
    }


def localize_faces(
    faces: list[AlignedFace],
    spatial_scores: dict[int, float],
    explainer: FaceExplainer,
) -> list[dict]:
    """Batched version: one localization record per face, correctly
    associated by face_id even if faces arrive in any order or a
    face_id has no corresponding spatial_score entry (falls back to
    None, which explain_face/generate_textual_explanation already
    handles by omitting the spatial-evidence sentence rather than
    guessing a value).

    spatial_scores: {face_id: spatial_score}, typically the output of
    spatial.inference.SpatialInference.score_faces() reshaped into a
    dict keyed by face_id.
    """
    results = []
    for face in faces:
        score = spatial_scores.get(face.face_id)
        results.append(localize_face(face, score, explainer))
    return results
