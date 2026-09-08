"""
Unit tests for Phase 8 — Temporal Analysis.

Run with:
    pytest tests/test_temporal_analysis.py -v
"""

import itertools

import numpy as np

from physiological.temporal import (
    moving_average, score_variance, aggregate_temporal_score,
    aggregate_video_scores, MIN_FRAMES_REQUIRED,
)
from physiological.temporal_orchestration import collect_frame_scores, compute_temporal_scores
from preprocessing.face_alignment import AlignedFace
from preprocessing.face_detector import BoundingBox
from preprocessing.pipeline import FrameFaces


# --- moving_average ---

def test_moving_average_empty_input():
    assert moving_average([]) == []


def test_moving_average_constant_sequence_unchanged():
    scores = [0.5] * 10
    smoothed = moving_average(scores, window=3)
    assert all(abs(s - 0.5) < 1e-9 for s in smoothed)


def test_moving_average_smooths_a_spike():
    scores = [0.1, 0.1, 0.9, 0.1, 0.1]
    smoothed = moving_average(scores, window=3)
    # the spike at index 2 should be pulled down by neighboring low values
    assert smoothed[2] < scores[2]


def test_moving_average_output_length_matches_input():
    scores = [0.3] * 15
    smoothed = moving_average(scores, window=5)
    assert len(smoothed) == len(scores)


# --- score_variance ---

def test_score_variance_zero_for_constant_sequence():
    assert score_variance([0.5] * 10) == 0.0


def test_score_variance_positive_for_varying_sequence():
    assert score_variance([0.1, 0.9, 0.1, 0.9]) > 0.0


def test_score_variance_single_value_is_zero():
    assert score_variance([0.5]) == 0.0


# --- aggregate_temporal_score ---

def test_aggregate_abstains_below_min_frames():
    scores = [0.5] * (MIN_FRAMES_REQUIRED - 1)
    result = aggregate_temporal_score(scores)
    assert result.abstained is True
    assert result.temporal_score == 0.5


def test_aggregate_consistent_high_stays_near_mean():
    scores = [0.85] * (MIN_FRAMES_REQUIRED + 10)
    result = aggregate_temporal_score(scores)
    assert result.abstained is False
    assert abs(result.temporal_score - 0.85) < 0.01
    assert result.std_dev == 0.0


def test_aggregate_consistent_low_stays_near_mean():
    scores = [0.10] * (MIN_FRAMES_REQUIRED + 10)
    result = aggregate_temporal_score(scores)
    assert abs(result.temporal_score - 0.10) < 0.01


def test_aggregate_flickering_scores_higher_than_its_own_mean():
    """Same mean as a stable-0.5 sequence, but wild frame-to-frame swings
    should push the aggregated score above the raw mean."""
    scores = list(itertools.islice(itertools.cycle([0.1, 0.9]), MIN_FRAMES_REQUIRED + 10))
    result = aggregate_temporal_score(scores)
    assert result.temporal_score > result.mean_score


def test_aggregate_flickering_scores_higher_than_equally_stable_sequence():
    n = MIN_FRAMES_REQUIRED + 10
    flickering = list(itertools.islice(itertools.cycle([0.1, 0.9]), n))
    stable_same_mean = [0.5] * n

    flick_result = aggregate_temporal_score(flickering)
    stable_result = aggregate_temporal_score(stable_same_mean)

    assert flick_result.temporal_score > stable_result.temporal_score


def test_aggregate_temporal_score_bounded_zero_to_one():
    scores = [1.0] * (MIN_FRAMES_REQUIRED + 5)
    result = aggregate_temporal_score(scores)
    assert 0.0 <= result.temporal_score <= 1.0


# --- aggregate_video_scores (contract) ---

def test_aggregate_video_scores_contract_shape():
    face_scores = {
        1: [0.8] * (MIN_FRAMES_REQUIRED + 5),
        2: [0.2] * (MIN_FRAMES_REQUIRED + 5),
    }
    results = aggregate_video_scores(face_scores)
    assert len(results) == 2
    for r in results:
        assert set(r.keys()) == {"face_id", "temporal_score"}


def test_aggregate_video_scores_sorted_by_face_id():
    face_scores = {
        3: [0.5] * (MIN_FRAMES_REQUIRED + 1),
        1: [0.5] * (MIN_FRAMES_REQUIRED + 1),
        2: [0.5] * (MIN_FRAMES_REQUIRED + 1),
    }
    results = aggregate_video_scores(face_scores)
    assert [r["face_id"] for r in results] == [1, 2, 3]


# --- orchestration ---

def test_collect_frame_scores_groups_by_face_id():
    def fake_score_fn(faces):
        return [{"face_id": f.face_id, "spatial_score": 0.5} for f in faces]

    crop = np.zeros((224, 224, 3), dtype=np.uint8)
    frame_faces = [
        FrameFaces(frame_index=0, timestamp_sec=0.0, faces=[
            AlignedFace(face_id=1, image=crop, source_bbox=BoundingBox(0, 0, 10, 10, 1.0), landmarks=None),
            AlignedFace(face_id=2, image=crop, source_bbox=BoundingBox(0, 0, 10, 10, 1.0), landmarks=None),
        ]),
        FrameFaces(frame_index=1, timestamp_sec=0.1, faces=[
            AlignedFace(face_id=1, image=crop, source_bbox=BoundingBox(0, 0, 10, 10, 1.0), landmarks=None),
        ]),
    ]

    scores = collect_frame_scores(frame_faces, fake_score_fn, "spatial_score")
    assert scores[1] == [0.5, 0.5]
    assert scores[2] == [0.5]


def test_collect_frame_scores_skips_empty_frames():
    def fake_score_fn(faces):
        return [{"face_id": f.face_id, "spatial_score": 0.5} for f in faces]

    frame_faces = [FrameFaces(frame_index=0, timestamp_sec=0.0, faces=[])]
    scores = collect_frame_scores(frame_faces, fake_score_fn, "spatial_score")
    assert scores == {}


def test_compute_temporal_scores_end_to_end():
    def fake_score_fn(faces):
        return [{"face_id": f.face_id, "spatial_score": 0.7} for f in faces]

    crop = np.zeros((224, 224, 3), dtype=np.uint8)
    frame_faces = [
        FrameFaces(frame_index=i, timestamp_sec=i / 10, faces=[
            AlignedFace(face_id=1, image=crop, source_bbox=BoundingBox(0, 0, 10, 10, 1.0), landmarks=None),
        ])
        for i in range(MIN_FRAMES_REQUIRED + 5)
    ]

    results = compute_temporal_scores(frame_faces, fake_score_fn, "spatial_score")
    assert len(results) == 1
    assert results[0]["face_id"] == 1
    assert abs(results[0]["temporal_score"] - 0.7) < 0.01
