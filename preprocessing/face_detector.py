"""
TrueSight — face detection.

Primary detector: MediaPipe FaceDetector (Tasks API), as specified in the
development plan. This requires a downloaded .task model file
(configs/pipeline.yaml -> preprocessing.mediapipe_model_path). Fetch it
with:

    curl -L -o models/mediapipe/blaze_face_short_range.tflite \\
      https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

Fallback detector: OpenCV's bundled Haar cascade. This ships with
opencv-python, needs no download, and works fully offline — useful for
local dev/CI environments without network access to Google's model
storage. It is NOT a replacement for MediaPipe in the real system: it's
less accurate and has no landmark output, which face_alignment.py needs.
It exists purely so the pipeline is runnable and testable everywhere.

Both detectors implement the same output contract: a list of BoundingBox.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    confidence: float

    def as_slice(self, image_shape: tuple[int, int]) -> tuple[slice, slice]:
        """Clamp box to image bounds and return (row_slice, col_slice)."""
        h, w = image_shape[:2]
        x0 = max(0, self.x)
        y0 = max(0, self.y)
        x1 = min(w, self.x + self.width)
        y1 = min(h, self.y + self.height)
        return slice(y0, y1), slice(x0, x1)


class FaceDetector:
    """Common interface. Use `create()` to get the best available backend."""

    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        raise NotImplementedError

    @staticmethod
    def create(
        backend: str = "auto",
        mediapipe_model_path: str | None = None,
        min_detection_confidence: float = 0.6,
    ) -> "FaceDetector":
        if backend == "mediapipe":
            return MediaPipeFaceDetector(mediapipe_model_path, min_detection_confidence)
        if backend == "haar":
            return HaarCascadeFaceDetector(min_detection_confidence)

        # auto: prefer MediaPipe if the model file is present, else fall back
        if mediapipe_model_path and Path(mediapipe_model_path).exists():
            return MediaPipeFaceDetector(mediapipe_model_path, min_detection_confidence)
        print(
            "[FaceDetector] MediaPipe .task model not found — "
            "falling back to OpenCV Haar cascade (dev/test only, "
            "see preprocessing/face_detector.py docstring)."
        )
        return HaarCascadeFaceDetector(min_detection_confidence)


class MediaPipeFaceDetector(FaceDetector):
    """Production detector — MediaPipe Tasks API."""

    def __init__(self, model_path: str, min_detection_confidence: float):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, core

        base_options = core.base_options.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        self._mp = mp

    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        boxes = []
        for detection in result.detections:
            bbox = detection.bounding_box
            score = detection.categories[0].score if detection.categories else 0.0
            boxes.append(
                BoundingBox(
                    x=bbox.origin_x,
                    y=bbox.origin_y,
                    width=bbox.width,
                    height=bbox.height,
                    confidence=float(score),
                )
            )
        return boxes


class HaarCascadeFaceDetector(FaceDetector):
    """Offline fallback — OpenCV bundled Haar cascade. Dev/test only."""

    def __init__(self, min_detection_confidence: float):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        # Haar doesn't output a confidence score directly; we treat every
        # detection as fixed-confidence since it already passed the
        # cascade's internal thresholding.
        self._fixed_confidence = min_detection_confidence

    def detect(self, image: np.ndarray) -> list[BoundingBox]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        return [
            BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h),
                        confidence=self._fixed_confidence)
            for (x, y, w, h) in detections
        ]
