"""
TrueSight — metadata analysis.

Critical rule from the development plan: "Do not make metadata a
standalone fake/real decision; use it as supporting evidence." Everything
in this module produces a `metadata_score` that is one input among many
to the fusion layer (Phase 10) — it must never be treated as, or exposed
as, a real/fake verdict on its own. A photo with no EXIF data is not
proof of anything; it's a single soft signal.

We extract:
  - Images: EXIF (camera make/model, software tag, original datetime)
  - Videos: codec (fourcc), fps, resolution, frame count/duration
  - Both: file size, file modification time

...and convert observations into named, human-readable flags rather than
an opaque number, so the reasoning stays inspectable and each flag can be
weighted independently. Known image/video-editing and AI-generation
software tags are treated as a strong signal; missing metadata is
treated as only a mild signal, since plenty of legitimately-shared real
media also lacks EXIF (re-uploads, screenshots, messaging-app
recompression all strip it).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from PIL import Image
from PIL.ExifTags import TAGS

# Software tags associated with editing or AI-generation tools. This is a
# supporting heuristic, not a definitive fake-detector — matching one of
# these only contributes a partial flag, since editing software also
# shows up on legitimately touched-up real photos.
SUSPICIOUS_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "faceapp", "deepfacelab", "faceswap",
    "stable diffusion", "midjourney", "dall-e", "dalle",
    "runway", "synthesia", "deepfake",
]

# Flag weights: how much each observation contributes to metadata_score.
# Kept small and additive (never any single flag near 1.0 alone) so
# metadata evidence stays proportionate — a supporting signal, not a verdict.
FLAG_WEIGHTS = {
    "no_exif_data": 0.15,
    "suspicious_software_tag": 0.45,
    "missing_creation_datetime": 0.10,
    "unusual_video_codec": 0.15,
    "unusual_frame_rate": 0.10,
    "unusual_resolution": 0.10,
}

COMMON_VIDEO_FOURCCS = {"avc1", "h264", "hvc1", "hev1", "mp4v", "xvid"}
COMMON_FPS_VALUES = {23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0}
FPS_TOLERANCE = 0.5


@dataclass
class MetadataAnalysis:
    flags: list[str] = field(default_factory=list)
    metadata_score: float = 0.0    # supporting evidence only — see module docstring
    raw: dict = field(default_factory=dict)


def _decode_fourcc(fourcc_float: float) -> str:
    fourcc_int = int(fourcc_float)
    chars = [chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(chars).strip().lower()


def extract_image_metadata(image_path: str) -> dict:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    raw = {
        "file_size_bytes": path.stat().st_size,
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "software": None,
        "datetime_original": None,
    }

    try:
        with Image.open(path) as img:
            raw["width"], raw["height"] = img.size
            exif = img.getexif()
            if exif:
                raw["has_exif"] = True
                tag_map = {TAGS.get(k, k): v for k, v in exif.items()}
                raw["camera_make"] = tag_map.get("Make")
                raw["camera_model"] = tag_map.get("Model")
                raw["software"] = tag_map.get("Software")
                raw["datetime_original"] = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
    except Exception as exc:
        raw["read_error"] = str(exc)

    return raw


def extract_video_metadata(video_path: str) -> dict:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    raw = {"file_size_bytes": path.stat().st_size}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raw["read_error"] = "OpenCV could not open video"
        return raw

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc_raw = cap.get(cv2.CAP_PROP_FOURCC)
    cap.release()

    raw["fps"] = round(fps, 3) if fps else None
    raw["frame_count"] = int(frame_count) if frame_count else None
    raw["width"] = int(width) if width else None
    raw["height"] = int(height) if height else None
    raw["duration_sec"] = round(frame_count / fps, 2) if fps and frame_count else None
    raw["codec"] = _decode_fourcc(fourcc_raw) if fourcc_raw else None

    return raw


def _flag_image(raw: dict) -> list[str]:
    flags = []

    if not raw.get("has_exif"):
        flags.append("no_exif_data")
    else:
        software = (raw.get("software") or "").lower()
        if any(keyword in software for keyword in SUSPICIOUS_SOFTWARE_KEYWORDS):
            flags.append("suspicious_software_tag")
        if not raw.get("datetime_original"):
            flags.append("missing_creation_datetime")

    return flags


def _flag_video(raw: dict) -> list[str]:
    flags = []

    codec = raw.get("codec")
    if codec and codec not in COMMON_VIDEO_FOURCCS:
        flags.append("unusual_video_codec")

    fps = raw.get("fps")
    if fps and not any(abs(fps - common) <= FPS_TOLERANCE for common in COMMON_FPS_VALUES):
        flags.append("unusual_frame_rate")

    width, height = raw.get("width"), raw.get("height")
    if width and height:
        # Flag extreme/non-standard aspect ratios or resolutions well
        # outside typical camera/phone/social-media output ranges.
        if width < 64 or height < 64 or width > 7680 or height > 4320:
            flags.append("unusual_resolution")

    return flags


def _score_flags(flags: list[str]) -> float:
    score = sum(FLAG_WEIGHTS.get(f, 0.0) for f in flags)
    return round(min(1.0, score), 4)


def analyze_metadata(file_path: str, media_type: str) -> MetadataAnalysis:
    """media_type: 'image' or 'video'. Returns supporting evidence only —
    see module docstring. Never expose this as a standalone verdict."""
    if media_type not in {"image", "video"}:
        raise ValueError(f"media_type must be 'image' or 'video', got {media_type!r}")

    if media_type == "image":
        raw = extract_image_metadata(file_path)
        flags = _flag_image(raw)
    else:
        raw = extract_video_metadata(file_path)
        flags = _flag_video(raw)

    score = _score_flags(flags)
    return MetadataAnalysis(flags=flags, metadata_score=score, raw=raw)
