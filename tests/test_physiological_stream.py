"""
Unit tests for Phase 7 — Physiological Stream.

Run with:
    pytest tests/test_physiological_stream.py -v

Note: uses the geometric fallback landmarker throughout, since the
MediaPipe FaceLandmarker .task model requires network access not
available in every dev/CI environment (see physiological/landmarks.py
docstring). Blink/rPPG *algorithm correctness* is validated with
controlled synthetic EAR/color sequences constructed directly — this
proves the math works independent of landmark source. Full accuracy on
real video requires the real MediaPipe landmarker.
"""

import numpy as np
import pytest

from physiological.landmarks import FaceMeshLandmarker, GeometricFallbackFaceMesh, FaceMeshResult
from physiological.ear import eye_aspect_ratio, average_ear
from physiological.blink import detect_blinks, analyze_blinks
from physiological.rppg import extract_mean_color_signal, bandpass_filter, analyze_rppg
from physiological.inference import PhysiologicalInference
from preprocessing.face_alignment import AlignedFace
from preprocessing.face_detector import BoundingBox
from preprocessing.pipeline import FrameFaces


# --- Landmarks ---

def test_factory_falls_back_to_geometric_without_model_path():
    landmarker = FaceMeshLandmarker.create()
    assert isinstance(landmarker, GeometricFallbackFaceMesh)


def test_geometric_fallback_returns_six_point_eyes():
    landmarker = GeometricFallbackFaceMesh()
    crop = np.full((224, 224, 3), 180, dtype=np.uint8)
    result = landmarker.get_landmarks(crop)
    assert len(result.left_eye) == 6
    assert len(result.right_eye) == 6


def test_geometric_fallback_cheek_roi_within_bounds():
    landmarker = GeometricFallbackFaceMesh()
    crop = np.full((224, 224, 3), 180, dtype=np.uint8)
    result = landmarker.get_landmarks(crop)
    x, y, w, h = result.cheek_roi
    assert 0 <= x < 224 and 0 <= y < 224
    assert x + w <= 224 and y + h <= 224


# --- EAR ---

def test_ear_rejects_wrong_point_count():
    with pytest.raises(ValueError):
        eye_aspect_ratio([(0, 0), (1, 1)])


def test_ear_decreases_as_eye_closes():
    def make_eye(vertical_gap):
        return [(0, 5), (2, 5 - vertical_gap/2), (4, 5 - vertical_gap/2),
                (6, 5), (4, 5 + vertical_gap/2), (2, 5 + vertical_gap/2)]

    open_ear = eye_aspect_ratio(make_eye(4.0))
    closing_ear = eye_aspect_ratio(make_eye(1.0))
    closed_ear = eye_aspect_ratio(make_eye(0.0))

    assert open_ear > closing_ear > closed_ear
    assert closed_ear == 0.0


def test_average_ear_combines_both_eyes():
    eye = [(0, 5), (2, 3), (4, 3), (6, 5), (4, 7), (2, 7)]
    result = average_ear(eye, eye)
    assert result == eye_aspect_ratio(eye)


# --- Blink detection ---

def test_detect_blinks_finds_dip_below_threshold():
    ear_seq = [0.3, 0.3, 0.08, 0.08, 0.3, 0.3]
    blinks = detect_blinks(ear_seq, threshold=0.21)
    assert blinks == [(2, 3)]


def test_detect_blinks_no_blinks_when_always_open():
    ear_seq = [0.3] * 20
    assert detect_blinks(ear_seq, threshold=0.21) == []


def test_detect_blinks_handles_blink_at_sequence_end():
    ear_seq = [0.3, 0.3, 0.08, 0.08]
    blinks = detect_blinks(ear_seq, threshold=0.21)
    assert blinks == [(2, 3)]


def test_analyze_blinks_abstains_on_short_sequence():
    result = analyze_blinks([0.3] * 5, fps=30)
    assert result.abstained is True
    assert result.blink_score == 0.5


def test_analyze_blinks_natural_pattern_scores_low():
    fps = 30
    n_frames = 300
    ear_seq = [0.30] * n_frames
    for center in [40, 120, 180, 260]:
        for offset in range(-2, 3):
            ear_seq[center + offset] = 0.08
    # slightly vary durations so it's not flagged as mechanically regular
    ear_seq[122] = 0.08
    result = analyze_blinks(ear_seq, fps=fps)
    assert result.abstained is False
    assert result.blink_score < 0.5


def test_analyze_blinks_zero_blinks_is_at_least_mildly_suspicious():
    result = analyze_blinks([0.30] * 300, fps=30)
    assert result.blink_count == 0
    assert result.blink_score > 0.0


def test_analyze_blinks_implausibly_rapid_scores_high():
    fps = 30
    n_frames = 300
    ear_seq = [0.30] * n_frames
    for i in range(0, n_frames, 15):
        for offset in range(3):
            if i + offset < n_frames:
                ear_seq[i + offset] = 0.08
    result = analyze_blinks(ear_seq, fps=fps)
    assert result.blink_score > 0.5


# --- rPPG ---

def test_extract_mean_color_signal_uses_green_channel():
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    crop[:, :, 1] = 200  # green channel
    signal = extract_mean_color_signal([crop])
    assert signal[0] == pytest.approx(200.0)


def test_extract_mean_color_signal_empty_input():
    assert extract_mean_color_signal([]).size == 0


def test_bandpass_filter_removes_dc_offset():
    fps = 30
    signal = np.full(300, 128.0)  # pure DC, no AC component
    filtered = bandpass_filter(signal, fps)
    assert abs(filtered.mean()) < 1.0


def test_analyze_rppg_abstains_on_short_sequence():
    crops = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]
    result = analyze_rppg(crops, fps=30)
    assert result.abstained is True
    assert result.rppg_score == 0.5


def test_analyze_rppg_detects_genuine_periodicity_as_low_score():
    fps = 30
    n_frames = fps * 10
    t = np.arange(n_frames) / fps
    heart_rate_hz = 1.2
    signal = 128 + 4.0 * np.sin(2 * np.pi * heart_rate_hz * t) + np.random.normal(0, 0.3, n_frames)

    crops = []
    for val in signal:
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        crop[:, :, 1] = np.clip(val, 0, 255)
        crops.append(crop)

    result = analyze_rppg(crops, fps=fps)
    assert result.abstained is False
    assert result.periodicity_strength > 0.3
    assert result.rppg_score < 0.5


def test_analyze_rppg_pure_noise_scores_higher_than_genuine_pulse():
    fps = 30
    n_frames = fps * 10
    t = np.arange(n_frames) / fps

    pulse_signal = 128 + 4.0 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.3, n_frames)
    pulse_crops = []
    for val in pulse_signal:
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        crop[:, :, 1] = np.clip(val, 0, 255)
        pulse_crops.append(crop)

    noise_signal = 128 + np.random.normal(0, 3.0, n_frames)
    noise_crops = []
    for val in noise_signal:
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        crop[:, :, 1] = np.clip(val, 0, 255)
        noise_crops.append(crop)

    pulse_result = analyze_rppg(pulse_crops, fps=fps)
    noise_result = analyze_rppg(noise_crops, fps=fps)

    assert pulse_result.rppg_score < noise_result.rppg_score


# --- Full inference ---

def test_physiological_inference_produces_contract_shape():
    fps = 30
    n_frames = 300
    frame_faces = []
    for i in range(n_frames):
        crop = np.full((224, 224, 3), 180, dtype=np.uint8)
        face = AlignedFace(face_id=1, image=crop, source_bbox=BoundingBox(0, 0, 224, 224, 1.0), landmarks=None)
        frame_faces.append(FrameFaces(frame_index=i, timestamp_sec=i / fps, faces=[face]))

    engine = PhysiologicalInference()
    results = engine.score_video(frame_faces, fps=fps)

    assert len(results) == 1
    assert set(results[0].keys()) == {"face_id", "blink_score", "rppg_score"}
    assert results[0]["face_id"] == 1
    assert 0.0 <= results[0]["blink_score"] <= 1.0
    assert 0.0 <= results[0]["rppg_score"] <= 1.0


def test_physiological_inference_handles_no_faces_gracefully():
    engine = PhysiologicalInference()
    results = engine.score_video([], fps=30)
    assert results == []


def test_physiological_inference_handles_multiple_tracked_faces():
    fps = 30
    n_frames = 300
    frame_faces = []
    for i in range(n_frames):
        crop = np.full((224, 224, 3), 180, dtype=np.uint8)
        face1 = AlignedFace(face_id=1, image=crop, source_bbox=BoundingBox(0, 0, 224, 224, 1.0), landmarks=None)
        face2 = AlignedFace(face_id=2, image=crop, source_bbox=BoundingBox(0, 0, 224, 224, 1.0), landmarks=None)
        frame_faces.append(FrameFaces(frame_index=i, timestamp_sec=i / fps, faces=[face1, face2]))

    engine = PhysiologicalInference()
    results = engine.score_video(frame_faces, fps=fps)

    face_ids = {r["face_id"] for r in results}
    assert face_ids == {1, 2}
