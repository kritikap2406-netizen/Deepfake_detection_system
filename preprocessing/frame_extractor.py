"""
TrueSight — frame extraction.

For video input, we don't process every single frame (too slow, too
redundant). We sample at a fixed rate and cap the total number of frames,
which is enough for spatial/frequency/noise scoring and for the temporal
and physiological streams to have a usable sequence.

For image input, this module is skipped entirely — the pipeline treats a
single image as a "video" of one frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ExtractedFrame:
    frame_index: int          # index within the original video
    timestamp_sec: float
    image: np.ndarray         # BGR, as read by OpenCV


def extract_frames(
    video_path: str,
    sample_rate_fps: float = 2.0,
    max_frames: int = 60,
) -> list[ExtractedFrame]:
    """Sample frames from a video at approximately `sample_rate_fps`.

    Raises FileNotFoundError if the path doesn't exist, and ValueError if
    OpenCV cannot open it (corrupt file, unsupported codec, etc).
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(source_fps / sample_rate_fps))

    frames: list[ExtractedFrame] = []
    frame_index = 0

    while len(frames) < max_frames:
        ok, image = cap.read()
        if not ok:
            break

        if frame_index % frame_interval == 0:
            frames.append(
                ExtractedFrame(
                    frame_index=frame_index,
                    timestamp_sec=frame_index / source_fps,
                    image=image,
                )
            )

        frame_index += 1

    cap.release()
    return frames


def load_single_image(image_path: str) -> ExtractedFrame:
    """Treat a still image as a one-frame 'video' for pipeline uniformity."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    return ExtractedFrame(frame_index=0, timestamp_sec=0.0, image=image)
