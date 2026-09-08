"""
TrueSight — face alignment.

Every detection stream (spatial, frequency, noise, physiological) must
receive faces in the exact same standardized form: a 224x224 crop, eyes
roughly horizontal, face centered. If each stream aligned faces
differently, their scores wouldn't be comparable and fusion would be
learning noise instead of signal.

Two alignment modes:
  - Landmark-based (uses MediaPipe FaceLandmarker eye landmarks to rotate
    the face so the eye line is horizontal before cropping). This is the
    production path and is what physiological/blink.py and rppg.py also
    need — they require accurate eye/cheek landmark positions.
  - Bounding-box-only fallback (no rotation, just crop + resize) — used
    when only a BoundingBox is available (e.g. Haar cascade fallback).
    Good enough for spatial/frequency/noise streams; NOT sufficient for
    blink/rPPG, which is called out explicitly at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from preprocessing.face_detector import BoundingBox

STANDARD_SIZE = 224


@dataclass
class AlignedFace:
    face_id: int
    image: np.ndarray            # STANDARD_SIZE x STANDARD_SIZE x 3, BGR
    source_bbox: BoundingBox
    landmarks: list[tuple[float, float]] | None  # None if bbox-only fallback


def align_from_bbox(
    image: np.ndarray,
    bbox: BoundingBox,
    face_id: int,
    margin_ratio: float = 0.25,
) -> AlignedFace:
    """Crop + resize only. No rotation correction. Fallback path."""
    h, w = image.shape[:2]

    margin_x = int(bbox.width * margin_ratio)
    margin_y = int(bbox.height * margin_ratio)

    x0 = max(0, bbox.x - margin_x)
    y0 = max(0, bbox.y - margin_y)
    x1 = min(w, bbox.x + bbox.width + margin_x)
    y1 = min(h, bbox.y + bbox.height + margin_y)

    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(f"Empty crop for bbox {bbox} on image of shape {image.shape}")

    resized = cv2.resize(crop, (STANDARD_SIZE, STANDARD_SIZE), interpolation=cv2.INTER_AREA)
    return AlignedFace(face_id=face_id, image=resized, source_bbox=bbox, landmarks=None)


def align_from_landmarks(
    image: np.ndarray,
    bbox: BoundingBox,
    landmarks: list[tuple[float, float]],
    left_eye_idx: int,
    right_eye_idx: int,
    face_id: int,
    margin_ratio: float = 0.25,
) -> AlignedFace:
    """Rotate so the eye line is horizontal, then crop + resize.

    `landmarks` are absolute pixel coordinates (already scaled to image
    size), as produced by preprocessing/pipeline.py from MediaPipe
    FaceLandmarker output.
    """
    left_eye = np.array(landmarks[left_eye_idx])
    right_eye = np.array(landmarks[right_eye_idx])

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = tuple(((left_eye + right_eye) / 2).tolist())
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    h, w = image.shape[:2]
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))

    # bbox corners must be rotated the same way before cropping
    corners = np.array([
        [bbox.x, bbox.y],
        [bbox.x + bbox.width, bbox.y + bbox.height],
    ])
    ones = np.ones((corners.shape[0], 1))
    rotated_corners = (rotation_matrix @ np.hstack([corners, ones]).T).T

    x0, y0 = rotated_corners.min(axis=0)
    x1, y1 = rotated_corners.max(axis=0)

    margin_x = (x1 - x0) * margin_ratio
    margin_y = (y1 - y0) * margin_ratio
    x0, y0 = int(max(0, x0 - margin_x)), int(max(0, y0 - margin_y))
    x1, y1 = int(min(w, x1 + margin_x)), int(min(h, y1 + margin_y))

    crop = rotated[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(f"Empty aligned crop for face_id={face_id}")

    resized = cv2.resize(crop, (STANDARD_SIZE, STANDARD_SIZE), interpolation=cv2.INTER_AREA)
    return AlignedFace(face_id=face_id, image=resized, source_bbox=bbox, landmarks=landmarks)
