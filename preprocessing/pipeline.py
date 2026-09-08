"""
TrueSight — common preprocessing pipeline.

This is THE entry point every detection stream should call. It hides
frame extraction, face detection, tracking, and alignment behind one
function so no stream silently builds its own slightly-different
preprocessing path (the exact failure mode the development plan warns
against in Phase 3).

    from preprocessing.pipeline import preprocess_image, preprocess_video

    faces = preprocess_image("photo.jpg")
    # -> list[AlignedFace], one per detected face

    video_faces = preprocess_video("clip.mp4")
    # -> list[FrameFaces], one per sampled frame, each holding AlignedFace
    #    objects with face_id consistent across frames
"""

from __future__ import annotations

from dataclasses import dataclass

from preprocessing.face_alignment import AlignedFace, align_from_bbox
from preprocessing.face_detector import FaceDetector
from preprocessing.face_tracker import FaceTracker
from preprocessing.frame_extractor import ExtractedFrame, extract_frames, load_single_image


@dataclass
class FrameFaces:
    frame_index: int
    timestamp_sec: float
    faces: list[AlignedFace]


def preprocess_image(
    image_path: str,
    detector: FaceDetector | None = None,
) -> list[AlignedFace]:
    """Detect + align all faces in a single still image.

    Uses bbox-only alignment (see face_alignment.py) — for images we don't
    yet wire in the landmark-based rotation step; that arrives with the
    physiological stream (Phase 7), which is the first consumer that
    actually needs precise eye landmarks.
    """
    detector = detector or FaceDetector.create()
    frame = load_single_image(image_path)

    boxes = detector.detect(frame.image)
    return [
        align_from_bbox(frame.image, bbox, face_id=i + 1)
        for i, bbox in enumerate(boxes)
    ]


def preprocess_video(
    video_path: str,
    detector: FaceDetector | None = None,
    sample_rate_fps: float = 2.0,
    max_frames: int = 60,
) -> list[FrameFaces]:
    """Detect + track + align faces across sampled video frames.

    face_id is consistent across frames for the same tracked person,
    which downstream physiological/temporal modules depend on.
    """
    detector = detector or FaceDetector.create()
    tracker = FaceTracker()

    frames = extract_frames(video_path, sample_rate_fps=sample_rate_fps, max_frames=max_frames)

    results: list[FrameFaces] = []
    for frame in frames:
        boxes = detector.detect(frame.image)
        tracked = tracker.update(boxes)  # list[(face_id, bbox)]

        aligned = [
            align_from_bbox(frame.image, bbox, face_id=face_id)
            for face_id, bbox in tracked
        ]
        results.append(
            FrameFaces(
                frame_index=frame.frame_index,
                timestamp_sec=frame.timestamp_sec,
                faces=aligned,
            )
        )

    return results
