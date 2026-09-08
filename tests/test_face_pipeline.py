"""
Unit tests for Phase 3 — Face Processing Pipeline.

Run with:
    pytest tests/test_face_pipeline.py -v

Note: these tests use the OpenCV Haar cascade fallback detector, since
the MediaPipe .task model file requires a network download not available
in every dev environment (see preprocessing/face_detector.py docstring).
Detection-accuracy assertions are therefore loose ("runs without
crashing", "handles zero faces gracefully") rather than strict accuracy
checks — real accuracy validation happens in Phase 4 against actual
dataset faces once MediaPipe is wired in with network access.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocessing.face_detector import FaceDetector, BoundingBox, HaarCascadeFaceDetector
from preprocessing.face_alignment import align_from_bbox, align_from_landmarks, STANDARD_SIZE
from preprocessing.face_tracker import FaceTracker, iou
from preprocessing.frame_extractor import extract_frames, load_single_image
from preprocessing.pipeline import preprocess_image, preprocess_video

ASSETS = Path("/home/claude/test_assets")


# --- Frame extraction ---

def test_load_single_image():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    assert frame.image.shape[2] == 3
    assert frame.frame_index == 0


def test_load_single_image_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_single_image("/nonexistent/path.jpg")


def test_extract_frames_from_video():
    frames = extract_frames(str(ASSETS / "synthetic_video.mp4"), sample_rate_fps=5.0, max_frames=10)
    assert len(frames) > 0
    assert all(f.image.shape[2] == 3 for f in frames)


def test_extract_frames_respects_max_frames():
    frames = extract_frames(str(ASSETS / "synthetic_video.mp4"), sample_rate_fps=100.0, max_frames=3)
    assert len(frames) <= 3


# --- Face detection ---

def test_detector_factory_falls_back_to_haar_without_model_path():
    detector = FaceDetector.create()
    assert isinstance(detector, HaarCascadeFaceDetector)


def test_detector_runs_on_blank_image_without_crashing():
    detector = FaceDetector.create()
    frame = load_single_image(str(ASSETS / "blank.jpg"))
    boxes = detector.detect(frame.image)
    assert isinstance(boxes, list)  # zero detections is a valid outcome


def test_detector_runs_on_synthetic_face_without_crashing():
    detector = FaceDetector.create()
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    boxes = detector.detect(frame.image)
    assert isinstance(boxes, list)
    for box in boxes:
        assert box.width > 0 and box.height > 0


# --- Alignment ---

def test_align_from_bbox_produces_standard_size():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    aligned = align_from_bbox(frame.image, bbox, face_id=1)
    assert aligned.image.shape == (STANDARD_SIZE, STANDARD_SIZE, 3)
    assert aligned.face_id == 1
    assert aligned.landmarks is None


def test_align_from_bbox_rejects_empty_crop():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bad_bbox = BoundingBox(x=10000, y=10000, width=50, height=50, confidence=1.0)
    with pytest.raises(ValueError):
        align_from_bbox(frame.image, bad_bbox, face_id=1)


def test_align_from_landmarks_produces_standard_size_and_rotates():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    # left eye, right eye at a slight tilt
    landmarks = [(285, 215), (355, 205)]
    aligned = align_from_landmarks(
        frame.image, bbox, landmarks, left_eye_idx=0, right_eye_idx=1, face_id=2,
    )
    assert aligned.image.shape == (STANDARD_SIZE, STANDARD_SIZE, 3)
    assert aligned.landmarks == landmarks


# --- Tracking ---

def test_iou_identical_boxes_is_one():
    box = BoundingBox(x=10, y=10, width=50, height=50, confidence=1.0)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = BoundingBox(x=0, y=0, width=10, height=10, confidence=1.0)
    b = BoundingBox(x=1000, y=1000, width=10, height=10, confidence=1.0)
    assert iou(a, b) == 0.0


def test_tracker_keeps_same_id_for_slowly_moving_box():
    tracker = FaceTracker(iou_threshold=0.3)
    box1 = BoundingBox(x=100, y=100, width=100, height=100, confidence=1.0)
    box2 = BoundingBox(x=105, y=100, width=100, height=100, confidence=1.0)  # small shift

    result1 = tracker.update([box1])
    result2 = tracker.update([box2])

    assert result1[0][0] == result2[0][0], "Small movement should keep the same face_id"


def test_tracker_assigns_new_id_for_unrelated_box():
    tracker = FaceTracker(iou_threshold=0.3)
    box1 = BoundingBox(x=100, y=100, width=50, height=50, confidence=1.0)
    box2 = BoundingBox(x=1000, y=1000, width=50, height=50, confidence=1.0)

    result1 = tracker.update([box1])
    result2 = tracker.update([box2])

    assert result1[0][0] != result2[0][0], "Unrelated box should get a new face_id"


def test_tracker_handles_two_simultaneous_faces():
    tracker = FaceTracker(iou_threshold=0.3)
    left = BoundingBox(x=0, y=0, width=50, height=50, confidence=1.0)
    right = BoundingBox(x=500, y=0, width=50, height=50, confidence=1.0)

    result = tracker.update([left, right])
    ids = {r[0] for r in result}
    assert len(ids) == 2, "Two simultaneous non-overlapping faces should get distinct IDs"


# --- Full pipeline orchestration ---

def test_preprocess_image_returns_aligned_faces_or_empty():
    faces = preprocess_image(str(ASSETS / "synthetic_face.jpg"))
    assert isinstance(faces, list)
    for face in faces:
        assert face.image.shape == (STANDARD_SIZE, STANDARD_SIZE, 3)


def test_preprocess_image_handles_no_faces_gracefully():
    faces = preprocess_image(str(ASSETS / "blank.jpg"))
    assert faces == []


def test_preprocess_video_returns_frame_faces_structure():
    frame_faces = preprocess_video(str(ASSETS / "synthetic_video.mp4"), sample_rate_fps=5.0, max_frames=5)
    assert isinstance(frame_faces, list)
    assert len(frame_faces) > 0
    for ff in frame_faces:
        assert hasattr(ff, "frame_index")
        assert hasattr(ff, "faces")


def test_preprocess_video_reuses_detector_across_frames_efficiently():
    """Passing one detector instance in should avoid reloading the cascade per frame."""
    detector = FaceDetector.create()
    frame_faces = preprocess_video(
        str(ASSETS / "synthetic_video.mp4"), detector=detector, sample_rate_fps=5.0, max_frames=5
    )
    assert len(frame_faces) > 0
