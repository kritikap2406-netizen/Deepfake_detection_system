"""
Unit tests for Phase 9 — Metadata Analysis.

Run with:
    pytest tests/test_metadata_analysis.py -v
"""

from pathlib import Path

import pytest
from PIL import Image
from PIL.ExifTags import TAGS

from metadata.analyzer import (
    analyze_metadata, extract_image_metadata, extract_video_metadata,
    _decode_fourcc, _score_flags, FLAG_WEIGHTS,
)

ASSETS = Path("/home/claude/test_assets")


@pytest.fixture(scope="module")
def suspicious_exif_image(tmp_path_factory):
    path = tmp_path_factory.mktemp("meta") / "suspicious.jpg"
    img = Image.new("RGB", (224, 224), color=(180, 170, 150))
    exif = Image.Exif()
    name_to_id = {v: k for k, v in TAGS.items()}
    exif[name_to_id["Software"]] = "DeepFaceLab 2.0"
    img.save(path, exif=exif)
    return str(path)


@pytest.fixture(scope="module")
def clean_exif_image(tmp_path_factory):
    path = tmp_path_factory.mktemp("meta") / "clean.jpg"
    img = Image.new("RGB", (224, 224), color=(180, 170, 150))
    exif = Image.Exif()
    name_to_id = {v: k for k, v in TAGS.items()}
    exif[name_to_id["Make"]] = "Apple"
    exif[name_to_id["Model"]] = "iPhone 15 Pro"
    exif[name_to_id["DateTimeOriginal"]] = "2026:01:15 10:30:00"
    img.save(path, exif=exif)
    return str(path)


@pytest.fixture(scope="module")
def no_exif_image(tmp_path_factory):
    path = tmp_path_factory.mktemp("meta") / "no_exif.jpg"
    img = Image.new("RGB", (224, 224), color=(180, 170, 150))
    img.save(path)  # PIL doesn't attach EXIF unless explicitly given
    return str(path)


# --- Image metadata extraction ---

def test_extract_image_metadata_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_image_metadata("/nonexistent/image.jpg")


def test_extract_image_metadata_reads_dimensions(no_exif_image):
    raw = extract_image_metadata(no_exif_image)
    assert raw["width"] == 224
    assert raw["height"] == 224


def test_extract_image_metadata_detects_no_exif(no_exif_image):
    raw = extract_image_metadata(no_exif_image)
    assert raw["has_exif"] is False


def test_extract_image_metadata_reads_software_tag(suspicious_exif_image):
    raw = extract_image_metadata(suspicious_exif_image)
    assert raw["has_exif"] is True
    assert raw["software"] == "DeepFaceLab 2.0"


def test_extract_image_metadata_reads_camera_info(clean_exif_image):
    raw = extract_image_metadata(clean_exif_image)
    assert raw["camera_make"] == "Apple"
    assert raw["camera_model"] == "iPhone 15 Pro"


# --- Video metadata extraction ---

def test_extract_video_metadata_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_video_metadata("/nonexistent/video.mp4")


def test_extract_video_metadata_reads_basic_fields():
    raw = extract_video_metadata(str(ASSETS / "synthetic_video.mp4"))
    assert raw["width"] == 640
    assert raw["height"] == 480
    assert raw["fps"] is not None
    assert raw["frame_count"] is not None


def test_decode_fourcc_roundtrip():
    import cv2
    fourcc_float = cv2.VideoWriter_fourcc(*"mp4v")
    decoded = _decode_fourcc(fourcc_float)
    assert decoded == "mp4v"


# --- Flagging heuristics ---

def test_no_exif_image_gets_mild_flag_only(no_exif_image):
    result = analyze_metadata(no_exif_image, media_type="image")
    assert result.flags == ["no_exif_data"]
    assert result.metadata_score == FLAG_WEIGHTS["no_exif_data"]


def test_suspicious_software_tag_flagged(suspicious_exif_image):
    result = analyze_metadata(suspicious_exif_image, media_type="image")
    assert "suspicious_software_tag" in result.flags


def test_clean_camera_exif_gets_no_flags(clean_exif_image):
    result = analyze_metadata(clean_exif_image, media_type="image")
    assert result.flags == []
    assert result.metadata_score == 0.0


def test_video_metadata_produces_valid_analysis():
    result = analyze_metadata(str(ASSETS / "synthetic_video.mp4"), media_type="video")
    assert isinstance(result.flags, list)
    assert 0.0 <= result.metadata_score <= 1.0


def test_analyze_metadata_rejects_invalid_media_type():
    with pytest.raises(ValueError):
        analyze_metadata(str(ASSETS / "synthetic_face.jpg"), media_type="audio")


# --- Score computation ---

def test_score_flags_empty_list_is_zero():
    assert _score_flags([]) == 0.0


def test_score_flags_never_exceeds_one():
    all_flags = list(FLAG_WEIGHTS.keys())
    assert _score_flags(all_flags) <= 1.0


def test_score_flags_no_single_flag_dominates():
    """Per the plan's critical rule, metadata is supporting evidence
    only -- no single heuristic should alone push the score anywhere
    near verdict territory (e.g. >= 0.5)."""
    for flag, weight in FLAG_WEIGHTS.items():
        assert weight < 0.5, f"{flag} weight {weight} is too strong for supporting-evidence-only design"


def test_score_flags_is_additive_for_multiple_flags():
    single = _score_flags(["no_exif_data"])
    combined = _score_flags(["no_exif_data", "missing_creation_datetime"])
    assert combined > single
