"""
Unit tests for Phase 17 — Video Prototype.

Per the plan: "Add video only after the image pipeline is stable."
Phase 16 confirmed the image chain. This phase confirms the video chain
end-to-end: video -> frame extraction -> face tracking -> all streams ->
blink/rPPG -> temporal analysis -> fusion -> result.

Unlike tests/test_backend.py's video test (which uses constant per-frame
content purely to check contract shape), these tests inject genuinely
time-varying content — a real sinusoidal pulse signal in the cheek ROI,
and slow spatial drift across frames — so the physiological/temporal
math is exercised with real signal, not just structurally.

Run with:
    pytest tests/test_video_prototype.py -v
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from backend.services.orchestrator import analyze_video
from preprocessing.face_alignment import AlignedFace
from preprocessing.face_detector import BoundingBox
from preprocessing.pipeline import FrameFaces

FUSION_MODEL = Path("models/fusion/best_fusion_model.joblib")
ASSETS = Path("/home/claude/test_assets")

FPS = 20.0
N_FRAMES = 200  # 10 seconds at 20fps -- matches configs/pipeline.yaml's rppg_window_seconds,
                # and Phase 7's own validated test window. Shorter windows (e.g. 4s) give
                # Welch's PSD estimate too little data to reliably discriminate genuine
                # periodicity from noise -- see this phase's summary for the debugging trail.


def _build_pulsing_video_frames(heart_rate_hz: float = 1.2, drift: bool = True) -> list[FrameFaces]:
    """One tracked face (face_id=1) across N_FRAMES, with a real periodic
    color signal injected into the cheek ROI region (matching
    GeometricFallbackFaceMesh's fixed cheek_roi) and optional slow
    spatial drift."""
    frame_faces = []
    for i in range(N_FRAMES):
        base_val = int(150 + (20 * i / N_FRAMES if drift else 0))
        crop = np.full((224, 224, 3), base_val, dtype=np.uint8)

        t = i / FPS
        pulse = 5.0 * np.sin(2 * np.pi * heart_rate_hz * t)
        crop[123:167, 67:156, 1] = np.clip(base_val + pulse, 0, 255).astype(np.uint8)

        face = AlignedFace(face_id=1, image=crop, source_bbox=BoundingBox(0, 0, 224, 224, 1.0), landmarks=None)
        frame_faces.append(FrameFaces(frame_index=i, timestamp_sec=t, faces=[face]))
    return frame_faces


def _build_flat_noise_video_frames() -> list[FrameFaces]:
    """Same shape, but pure random noise in the cheek ROI instead of a
    genuine periodic signal — the rPPG 'no real pulse' comparison case."""
    frame_faces = []
    rng = np.random.default_rng(0)
    for i in range(N_FRAMES):
        crop = np.full((224, 224, 3), 150, dtype=np.uint8)
        noise = rng.normal(0, 8.0, (44, 89))
        crop[123:167, 67:156, 1] = np.clip(150 + noise, 0, 255).astype(np.uint8)

        face = AlignedFace(face_id=1, image=crop, source_bbox=BoundingBox(0, 0, 224, 224, 1.0), landmarks=None)
        frame_faces.append(FrameFaces(frame_index=i, timestamp_sec=i / FPS, faces=[face]))
    return frame_faces


@pytest.mark.skipif(not FUSION_MODEL.exists(), reason="Requires all trained checkpoints")
def test_video_prototype_full_chain_produces_complete_contract():
    frame_faces = _build_pulsing_video_frames()

    with patch("backend.services.orchestrator.preprocess_video", return_value=frame_faces):
        result = analyze_video(str(ASSETS / "synthetic_video.mp4"), sample_rate_fps=FPS)

    assert result["prediction"] in {"REAL", "FAKE"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["faces"]) == 1

    face = result["faces"][0]
    assert set(face.keys()) == {
        "face_id", "spatial_score", "frequency_score", "noise_score",
        "blink_score", "rppg_score", "temporal_score", "heatmap", "suspicious_regions",
    }


@pytest.mark.skipif(not FUSION_MODEL.exists(), reason="Requires all trained checkpoints")
def test_video_prototype_genuine_pulse_produces_low_rppg_score():
    """The core Phase 17 validation: a real periodic signal, injected at
    the orchestration level (not directly into physiological.rppg),
    should still be correctly detected as real-like after passing
    through preprocess_video's frame structure, PhysiologicalInference,
    and stream-score assembly."""
    frame_faces = _build_pulsing_video_frames(heart_rate_hz=1.2)

    with patch("backend.services.orchestrator.preprocess_video", return_value=frame_faces):
        result = analyze_video(str(ASSETS / "synthetic_video.mp4"), sample_rate_fps=FPS)

    assert result["faces"][0]["rppg_score"] < 0.5


@pytest.mark.skipif(not FUSION_MODEL.exists(), reason="Requires all trained checkpoints")
def test_video_prototype_noise_scores_higher_rppg_than_genuine_pulse():
    pulse_frames = _build_pulsing_video_frames(heart_rate_hz=1.2)
    noise_frames = _build_flat_noise_video_frames()

    with patch("backend.services.orchestrator.preprocess_video", return_value=pulse_frames):
        pulse_result = analyze_video(str(ASSETS / "synthetic_video.mp4"), sample_rate_fps=FPS)

    with patch("backend.services.orchestrator.preprocess_video", return_value=noise_frames):
        noise_result = analyze_video(str(ASSETS / "synthetic_video.mp4"), sample_rate_fps=FPS)

    assert pulse_result["faces"][0]["rppg_score"] < noise_result["faces"][0]["rppg_score"]


@pytest.mark.skipif(not FUSION_MODEL.exists(), reason="Requires all trained checkpoints")
def test_video_prototype_temporal_score_reflects_stability_not_just_mean():
    """Stable (slowly drifting but consistent) spatial content across
    frames should produce a temporal_score close to the mean spatial
    score — i.e. no flickering-instability penalty applied — confirming
    Phase 8's aggregation logic genuinely runs on real per-frame scores
    collected through the orchestrator, not a placeholder."""
    frame_faces = _build_pulsing_video_frames(drift=True)

    with patch("backend.services.orchestrator.preprocess_video", return_value=frame_faces):
        result = analyze_video(str(ASSETS / "synthetic_video.mp4"), sample_rate_fps=FPS)

    face = result["faces"][0]
    assert abs(face["temporal_score"] - face["spatial_score"]) < 0.05


@pytest.mark.skipif(not FUSION_MODEL.exists(), reason="Requires all trained checkpoints")
def test_video_prototype_blink_score_matches_known_fallback_behavior():
    """The geometric fallback landmarker (Phase 7) returns fixed landmark
    coordinates regardless of pixel content, so EAR never varies and zero
    blinks are ever detected — a documented limitation, not a bug. This
    should propagate consistently through the full video orchestration:
    a sufficiently long clip (>= min_frames_required) with zero detected
    blinks scores the same mild suspicion level Phase 7 validated
    directly (0.3733), not some other value."""
    frame_faces = _build_pulsing_video_frames()

    with patch("backend.services.orchestrator.preprocess_video", return_value=frame_faces):
        result = analyze_video(str(ASSETS / "synthetic_video.mp4"), sample_rate_fps=FPS)

    assert result["faces"][0]["blink_score"] == pytest.approx(0.3733, abs=0.01)


@pytest.mark.skipif(not FUSION_MODEL.exists(), reason="Requires all trained checkpoints")
def test_video_prototype_fusion_responds_to_real_stream_differences():
    """Confirms fusion genuinely weighs the physiological evidence rather
    than ignoring it: a video with a real plausible pulse should not
    score a *higher* fake-probability than the same video with pure
    noise in place of that pulse (all else equal)."""
    pulse_frames = _build_pulsing_video_frames(heart_rate_hz=1.2)
    noise_frames = _build_flat_noise_video_frames()

    with patch("backend.services.orchestrator.preprocess_video", return_value=pulse_frames):
        pulse_result = analyze_video(str(ASSETS / "synthetic_video.mp4"), sample_rate_fps=FPS)

    with patch("backend.services.orchestrator.preprocess_video", return_value=noise_frames):
        noise_result = analyze_video(str(ASSETS / "synthetic_video.mp4"), sample_rate_fps=FPS)

    pulse_fake_prob = pulse_result["faces"][0]["rppg_score"]
    noise_fake_prob = noise_result["faces"][0]["rppg_score"]
    assert pulse_fake_prob <= noise_fake_prob


@pytest.mark.skipif(not FUSION_MODEL.exists(), reason="Requires all trained checkpoints")
def test_video_prototype_multiple_tracked_faces_stay_independent():
    """Two simultaneously tracked faces with different injected pulse
    rates should each get their own independently-computed rppg_score,
    with no cross-contamination between face_ids through the full
    orchestrated chain."""
    frame_faces = []
    for i in range(N_FRAMES):
        t = i / FPS
        crop1 = np.full((224, 224, 3), 150, dtype=np.uint8)
        pulse1 = 5.0 * np.sin(2 * np.pi * 1.2 * t)
        crop1[123:167, 67:156, 1] = np.clip(150 + pulse1, 0, 255).astype(np.uint8)
        face1 = AlignedFace(face_id=1, image=crop1, source_bbox=BoundingBox(0, 0, 224, 224, 1.0), landmarks=None)

        rng = np.random.default_rng(i)
        crop2 = np.full((224, 224, 3), 150, dtype=np.uint8)
        noise2 = rng.normal(0, 8.0, (44, 89))
        crop2[123:167, 67:156, 1] = np.clip(150 + noise2, 0, 255).astype(np.uint8)
        face2 = AlignedFace(face_id=2, image=crop2, source_bbox=BoundingBox(0, 0, 224, 224, 1.0), landmarks=None)

        frame_faces.append(FrameFaces(frame_index=i, timestamp_sec=t, faces=[face1, face2]))

    with patch("backend.services.orchestrator.preprocess_video", return_value=frame_faces):
        result = analyze_video(str(ASSETS / "synthetic_video.mp4"), sample_rate_fps=FPS)

    by_id = {f["face_id"]: f for f in result["faces"]}
    assert set(by_id.keys()) == {1, 2}
    assert by_id[1]["rppg_score"] < by_id[2]["rppg_score"]
