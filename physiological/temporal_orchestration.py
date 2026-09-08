"""
TrueSight — temporal orchestration helper.

Bridges preprocessing.pipeline.preprocess_video's per-frame tracked faces
with any single-crop stream scorer (spatial, frequency, or noise
inference) and physiological.temporal's aggregation. This is what the
backend (Phase 14) will actually call: run a stream's inference across
every sampled frame, collect scores by face_id, then aggregate.

Kept separate from temporal.py itself so temporal.py stays a pure,
dependency-light aggregation module (easy to unit test with plain lists)
while this file handles the messier job of iterating frames and faces.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from preprocessing.pipeline import FrameFaces
from physiological.temporal import aggregate_video_scores


def collect_frame_scores(
    frame_faces: list[FrameFaces],
    score_fn: Callable[[list], list[dict]],
    score_key: str,
) -> dict[int, list[float]]:
    """score_fn: a stream's batched inference function, e.g.
    SpatialInference.score_faces — takes list[AlignedFace], returns
    list[{"face_id": int, <score_key>: float}].

    Returns {face_id: [score at each frame the face appeared in]}, in
    frame order, ready for physiological.temporal.aggregate_video_scores.
    """
    scores_by_face: dict[int, list[float]] = defaultdict(list)

    for frame in frame_faces:
        if not frame.faces:
            continue
        frame_results = score_fn(frame.faces)
        for result in frame_results:
            scores_by_face[result["face_id"]].append(result[score_key])

    return dict(scores_by_face)


def compute_temporal_scores(
    frame_faces: list[FrameFaces],
    score_fn: Callable[[list], list[dict]],
    score_key: str,
) -> list[dict]:
    """End-to-end: per-frame scoring -> per-face score sequences -> temporal aggregation."""
    scores_by_face = collect_frame_scores(frame_faces, score_fn, score_key)
    return aggregate_video_scores(scores_by_face)
