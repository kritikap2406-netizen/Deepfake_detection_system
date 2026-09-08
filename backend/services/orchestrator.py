"""
TrueSight — pipeline orchestration.

This is what the API routes call: it wires together every phase built so
far (preprocessing -> spatial/frequency/noise -> physiological/temporal
for video -> metadata -> fusion -> calibration -> explainability) into
one end-to-end call per request.

Multi-face aggregation policy (not specified by the development plan,
so documented explicitly here as our own prototype design decision):
each detected face gets its own independent fusion verdict. The
image/video-level overall prediction is FAKE if ANY face is predicted
FAKE (a single manipulated face is enough to flag the whole media item),
and the overall confidence is the highest calibrated_confidence among
FAKE-predicted faces. If every face is predicted REAL, overall
confidence is 1 minus the highest fake-probability seen across faces
(i.e. confidence in the REAL verdict is only as strong as the
least-reassuring face).

Models are loaded once per process (module-level singletons) rather
than per-request, since checkpoint loading is comparatively expensive
and requests should stay fast.
"""

from __future__ import annotations

import uuid

from preprocessing.pipeline import preprocess_image, preprocess_video
from spatial.inference import SpatialInference
from frequency_noise.inference import FrequencyInference, NoiseInference
from physiological.inference import PhysiologicalInference
from physiological.temporal_orchestration import collect_frame_scores, compute_temporal_scores
from metadata.analyzer import analyze_metadata
from fusion.inference import FusionInference
from explainability.inference import ExplainabilityInference
from explainability.tampering_localization import localize_faces

_engines: dict = {}


def _get_engines() -> dict:
    """Lazy singleton loader — models load on first request, not at
    import time, so the API can start even if a checkpoint is briefly
    missing during development, failing only when actually needed."""
    if not _engines:
        _engines["spatial"] = SpatialInference()
        _engines["frequency"] = FrequencyInference()
        _engines["noise"] = NoiseInference()
        _engines["physiological"] = PhysiologicalInference()
        _engines["fusion"] = FusionInference()
        _engines["explainability"] = ExplainabilityInference()
    return _engines


def engines_available() -> dict[str, bool]:
    """Non-throwing check of which model checkpoints exist, for GET /health."""
    from spatial.inference import DEFAULT_CHECKPOINT as spatial_ckpt
    from frequency_noise.inference import DEFAULT_CHECKPOINT as freq_ckpt, DEFAULT_NOISE_CHECKPOINT as noise_ckpt
    from fusion.inference import DEFAULT_MODEL_PATH as fusion_ckpt

    return {
        "spatial": spatial_ckpt.exists(),
        "frequency": freq_ckpt.exists(),
        "noise": noise_ckpt.exists(),
        "fusion": fusion_ckpt.exists(),
    }


def _aggregate_overall(face_verdicts: list[dict]) -> tuple[str, float]:
    """See module docstring for the aggregation policy."""
    fake_faces = [v for v in face_verdicts if v["prediction"] == "FAKE"]

    if fake_faces:
        confidence = max(v["calibrated_confidence"] for v in fake_faces)
        return "FAKE", round(confidence, 4)

    highest_fake_prob = max((v["calibrated_confidence"] for v in face_verdicts), default=0.0)
    return "REAL", round(1.0 - highest_fake_prob, 4)


def analyze_image(file_path: str) -> dict:
    engines = _get_engines()
    result_id = str(uuid.uuid4())

    faces = preprocess_image(file_path)
    if not faces:
        return {
            "id": result_id, "media_type": "image", "prediction": "UNKNOWN",
            "message": "No face could be detected in the provided image.",
        }

    spatial_results = {r["face_id"]: r["spatial_score"] for r in engines["spatial"].score_faces(faces)}
    frequency_results = {r["face_id"]: r["frequency_score"] for r in engines["frequency"].score_faces(faces)}
    noise_results = {r["face_id"]: r["noise_score"] for r in engines["noise"].score_faces(faces)}
    metadata_score = analyze_metadata(file_path, media_type="image").metadata_score

    face_verdicts = []
    face_records = []
    for face in faces:
        stream_scores = {
            "spatial_score": spatial_results.get(face.face_id, 0.5),
            "frequency_score": frequency_results.get(face.face_id, 0.5),
            "noise_score": noise_results.get(face.face_id, 0.5),
            # blink/rppg/temporal are video-only — neutral default for images,
            # consistent with fusion/features.py's convention.
            "metadata_flag": metadata_score,
        }
        verdict = engines["fusion"].predict(stream_scores)
        face_verdicts.append(verdict)

        localized = localize_faces([face], spatial_results, engines["explainability"])[0]
        face_records.append({
            "face_id": face.face_id,
            "spatial_score": stream_scores["spatial_score"],
            "frequency_score": stream_scores["frequency_score"],
            "noise_score": stream_scores["noise_score"],
            "blink_score": 0.5,
            "rppg_score": 0.5,
            "temporal_score": 0.5,
            "heatmap": localized["heatmap"],
            "suspicious_regions": localized["suspicious_regions"],
        })

    overall_prediction, overall_confidence = _aggregate_overall(face_verdicts)
    explanation = _build_explanation(face_records, overall_prediction, metadata_score)

    return {
        "id": result_id,
        "media_type": "image",
        "prediction": overall_prediction,
        "confidence": overall_confidence,
        "faces": face_records,
        "explanation": explanation,
        "metadata_flag": metadata_score,
    }


def analyze_video(file_path: str, sample_rate_fps: float = 2.0, max_frames: int = 60) -> dict:
    engines = _get_engines()
    result_id = str(uuid.uuid4())

    frame_faces = preprocess_video(file_path, sample_rate_fps=sample_rate_fps, max_frames=max_frames)
    if not any(f.faces for f in frame_faces):
        return {
            "id": result_id, "media_type": "video", "prediction": "UNKNOWN",
            "message": "No face could be detected in the provided video.",
        }

    spatial_scores = collect_frame_scores(frame_faces, engines["spatial"].score_faces, "spatial_score")
    frequency_scores = collect_frame_scores(frame_faces, engines["frequency"].score_faces, "frequency_score")
    noise_scores = collect_frame_scores(frame_faces, engines["noise"].score_faces, "noise_score")

    spatial_temporal = compute_temporal_scores(frame_faces, engines["spatial"].score_faces, "spatial_score")
    temporal_by_face = {r["face_id"]: r["temporal_score"] for r in spatial_temporal}

    physio_results = engines["physiological"].score_video(frame_faces, fps=sample_rate_fps)
    physio_by_face = {r["face_id"]: r for r in physio_results}

    metadata_score = analyze_metadata(file_path, media_type="video").metadata_score

    # One representative frame per tracked face (first frame it appears
    # in) for explainability — generating a heatmap per frame per face
    # would be excessive for a prototype; a single representative crop
    # is enough to show the user *why* that face was flagged.
    representative_face_by_id = {}
    for frame in frame_faces:
        for face in frame.faces:
            representative_face_by_id.setdefault(face.face_id, face)

    face_verdicts = []
    face_records = []
    for face_id, face in representative_face_by_id.items():
        mean_spatial = _safe_mean(spatial_scores.get(face_id, []))
        mean_frequency = _safe_mean(frequency_scores.get(face_id, []))
        mean_noise = _safe_mean(noise_scores.get(face_id, []))
        blink_score = physio_by_face.get(face_id, {}).get("blink_score", 0.5)
        rppg_score = physio_by_face.get(face_id, {}).get("rppg_score", 0.5)
        temporal_score = temporal_by_face.get(face_id, 0.5)

        stream_scores = {
            "spatial_score": mean_spatial,
            "frequency_score": mean_frequency,
            "noise_score": mean_noise,
            "blink_score": blink_score,
            "rppg_score": rppg_score,
            "temporal_score": temporal_score,
            "metadata_flag": metadata_score,
        }
        verdict = engines["fusion"].predict(stream_scores)
        face_verdicts.append(verdict)

        localized = localize_faces([face], {face_id: mean_spatial}, engines["explainability"])[0]
        face_records.append({
            "face_id": face_id,
            "spatial_score": round(mean_spatial, 4),
            "frequency_score": round(mean_frequency, 4),
            "noise_score": round(mean_noise, 4),
            "blink_score": blink_score,
            "rppg_score": rppg_score,
            "temporal_score": temporal_score,
            "heatmap": localized["heatmap"],
            "suspicious_regions": localized["suspicious_regions"],
        })

    overall_prediction, overall_confidence = _aggregate_overall(face_verdicts)
    explanation = _build_explanation(face_records, overall_prediction, metadata_score)

    return {
        "id": result_id,
        "media_type": "video",
        "prediction": overall_prediction,
        "confidence": overall_confidence,
        "faces": face_records,
        "explanation": explanation,
        "metadata_flag": metadata_score,
    }


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.5


def _build_explanation(face_records: list[dict], overall_prediction: str, metadata_score: float) -> list[str]:
    """Top-level explanation summary, evidence-grounded per the plan's
    Phase 12 requirement — built from the actual per-face records, not
    a canned sentence independent of them."""
    lines = [f"Overall verdict: {overall_prediction} ({len(face_records)} face(s) analyzed)."]

    for record in face_records:
        if record["suspicious_regions"]:
            regions = ", ".join(r.replace("_", " ") for r in record["suspicious_regions"])
            lines.append(f"Face {record['face_id']}: activation concentrated in {regions}.")

    if metadata_score >= 0.3:
        lines.append(f"Metadata analysis found supporting evidence (score {metadata_score:.2f}).")

    return lines
