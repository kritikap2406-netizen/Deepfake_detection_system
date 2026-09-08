"""
TrueSight — face landmarks for physiological analysis.

Blink detection (ear.py/blink.py) needs precise eye-corner and eyelid
points. rPPG (rppg.py) needs a stable cheek/forehead region to sample
skin color from. Both need this on every frame of a video, which is why
this stays a separate, lightweight module rather than re-running a heavy
detector.

Primary backend: MediaPipe FaceLandmarker (Tasks API), 468-point face
mesh. Requires a downloaded .task model file (same network constraint as
preprocessing/face_detector.py — see that file's docstring for the fetch
command; the face_landmarker model is fetched the same way from
MediaPipe's model storage, landmarker variant).

Fallback: a fixed geometric estimate of eye/cheek regions based on
standardized face proportions. This is NOT eye-tracking — it assumes the
input is already an aligned, centered 224x224 face crop (which
preprocessing.pipeline guarantees) and places regions at the average
relative position eyes/cheeks occupy in such a crop. It exists purely so
ear.py/rppg.py/blink.py are runnable and testable without network access
to MediaPipe's landmark model. It will NOT produce accurate blink/rPPG
signal on real video — that requires the real landmarker.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Standard MediaPipe Face Mesh landmark indices for the 6-point
# eye-aspect-ratio calculation (widely documented indices, not
# project-specific). Order matches the classic EAR formulation:
# [left_corner, top_1, top_2, right_corner, bottom_2, bottom_1]
MEDIAPIPE_LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
MEDIAPIPE_RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# A small set of cheek/forehead indices for rPPG ROI sampling.
MEDIAPIPE_LEFT_CHEEK_IDX = [50, 101, 118, 205]
MEDIAPIPE_RIGHT_CHEEK_IDX = [280, 330, 347, 425]
MEDIAPIPE_FOREHEAD_IDX = [67, 109, 10, 338, 297]


@dataclass
class FaceMeshResult:
    left_eye: list[tuple[float, float]]   # 6 points, pixel coords in the crop
    right_eye: list[tuple[float, float]]  # 6 points
    cheek_roi: tuple[int, int, int, int]  # (x, y, w, h) within the crop, for rPPG sampling


class FaceMeshLandmarker:
    def get_landmarks(self, face_crop_bgr: np.ndarray) -> FaceMeshResult | None:
        """Returns None if no face mesh could be computed for this crop."""
        raise NotImplementedError

    @staticmethod
    def create(backend: str = "auto", mediapipe_model_path: str | None = None) -> "FaceMeshLandmarker":
        if backend == "mediapipe":
            return MediaPipeFaceMesh(mediapipe_model_path)
        if backend == "geometric":
            return GeometricFallbackFaceMesh()

        if mediapipe_model_path and Path(mediapipe_model_path).exists():
            return MediaPipeFaceMesh(mediapipe_model_path)
        print(
            "[FaceMeshLandmarker] MediaPipe FaceLandmarker .task model not found — "
            "falling back to geometric region estimate (dev/test only, "
            "see preprocessing/../physiological/landmarks.py docstring). "
            "Blink/rPPG signal from this fallback is NOT meaningful, only structurally testable."
        )
        return GeometricFallbackFaceMesh()


class MediaPipeFaceMesh(FaceMeshLandmarker):
    """Production landmarker — MediaPipe Tasks API FaceLandmarker."""

    def __init__(self, model_path: str):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, core

        base_options = core.base_options.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp

    def get_landmarks(self, face_crop_bgr: np.ndarray) -> FaceMeshResult | None:
        import cv2

        h, w = face_crop_bgr.shape[:2]
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        points = result.face_landmarks[0]  # first (only) face

        def to_pixel(idx: int) -> tuple[float, float]:
            lm = points[idx]
            return (lm.x * w, lm.y * h)

        left_eye = [to_pixel(i) for i in MEDIAPIPE_LEFT_EYE_IDX]
        right_eye = [to_pixel(i) for i in MEDIAPIPE_RIGHT_EYE_IDX]

        cheek_pts = [to_pixel(i) for i in MEDIAPIPE_LEFT_CHEEK_IDX + MEDIAPIPE_RIGHT_CHEEK_IDX]
        xs = [p[0] for p in cheek_pts]
        ys = [p[1] for p in cheek_pts]
        cheek_roi = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))

        return FaceMeshResult(left_eye=left_eye, right_eye=right_eye, cheek_roi=cheek_roi)


class GeometricFallbackFaceMesh(FaceMeshLandmarker):
    """Offline fallback for a standardized, aligned 224x224 face crop.

    Positions are rough averages for where eyes/cheeks sit in a
    face-centered crop with ~25% margin (matching
    preprocessing.face_alignment's default margin_ratio). Dev/test only.
    """

    def get_landmarks(self, face_crop_bgr: np.ndarray) -> FaceMeshResult | None:
        h, w = face_crop_bgr.shape[:2]

        def eye_hexagon(center_x: float, center_y: float, eye_w: float, eye_h: float) -> list[tuple[float, float]]:
            # 6 points roughly tracing an eye outline: [left, top1, top2, right, bottom2, bottom1]
            return [
                (center_x - eye_w / 2, center_y),
                (center_x - eye_w / 4, center_y - eye_h / 2),
                (center_x + eye_w / 4, center_y - eye_h / 2),
                (center_x + eye_w / 2, center_y),
                (center_x + eye_w / 4, center_y + eye_h / 2),
                (center_x - eye_w / 4, center_y + eye_h / 2),
            ]

        eye_y = h * 0.40
        eye_w, eye_h = w * 0.15, h * 0.06

        left_eye = eye_hexagon(w * 0.35, eye_y, eye_w, eye_h)
        right_eye = eye_hexagon(w * 0.65, eye_y, eye_w, eye_h)

        cheek_x = int(w * 0.30)
        cheek_y = int(h * 0.55)
        cheek_w = int(w * 0.40)
        cheek_h = int(h * 0.20)
        cheek_roi = (cheek_x, cheek_y, cheek_w, cheek_h)

        return FaceMeshResult(left_eye=left_eye, right_eye=right_eye, cheek_roi=cheek_roi)
