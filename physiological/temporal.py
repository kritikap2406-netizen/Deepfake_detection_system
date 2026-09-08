"""
TrueSight — temporal analysis.

A video isn't just a bag of independently-scored frames — how evidence
*changes* across frames matters too. Two videos can have the same
average spatial_score but tell very different stories: one where every
frame is consistently suspicious (steady manipulation) vs one where
suspicion flickers wildly frame to frame (often a sign of inconsistent
generation, or a detector picking up compression artifacts rather than
real manipulation evidence).

Per the plan, we deliberately keep this to simple, inspectable
statistics — moving averages and variance/consistency measures — rather
than a temporal deep-learning model (LSTM/Temporal Transformer are
explicitly called out as a later, optional upgrade once this baseline is
proven reliable).

This module is agnostic to *which* stream produced the per-frame scores:
it can aggregate a sequence of spatial_scores, frequency_scores,
noise_scores, or a blended per-frame evidence value. The orchestrating
caller (backend, in Phase 14) decides what per-frame signal to feed in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from configs_loader import load_config

_config = load_config()
MIN_FRAMES_REQUIRED = _config["physiological"]["min_frames_required"]

# How strongly frame-to-frame instability boosts the aggregated score
# above the plain average. Kept as a named constant (not buried in the
# formula) so it's easy to tune during hackathon demo calibration.
CONSISTENCY_PENALTY_WEIGHT = 0.4

DEFAULT_MOVING_AVERAGE_WINDOW = 5


@dataclass
class TemporalAnalysis:
    frame_count: int
    mean_score: float
    smoothed_scores: list[float]
    std_dev: float
    temporal_score: float   # 0 (consistent, real-like) - 1 (unstable/high evidence, fake-like)
    abstained: bool


def moving_average(scores: list[float], window: int = DEFAULT_MOVING_AVERAGE_WINDOW) -> list[float]:
    """Simple centered-ish moving average (trailing window, since we don't
    want to use future frames not yet available in a streaming context).
    Smooths out single-frame noise so the variance calculation reflects
    genuine trend instability rather than per-frame detector jitter.
    """
    if not scores:
        return []
    window = max(1, min(window, len(scores)))

    smoothed = []
    for i in range(len(scores)):
        start = max(0, i - window + 1)
        smoothed.append(float(np.mean(scores[start:i + 1])))
    return smoothed


def score_variance(scores: list[float]) -> float:
    """Standard deviation of the raw (unsmoothed) per-frame scores —
    the consistency/instability measure."""
    if len(scores) < 2:
        return 0.0
    return float(np.std(scores))


def aggregate_temporal_score(
    scores: list[float],
    window: int = DEFAULT_MOVING_AVERAGE_WINDOW,
) -> TemporalAnalysis:
    """Combine a sequence of per-frame [0,1] evidence scores into one
    video-level temporal_score.

    Logic: start from the mean of the smoothed sequence (central
    tendency, robust to single-frame spikes), then add a penalty
    proportional to how unstable the raw scores are frame-to-frame.
    High instability is itself suspicious evidence — a real, unedited
    video's per-frame manipulation-likelihood tends to be fairly stable,
    while flickering evidence often indicates inconsistent or partial
    manipulation, or an unstable detection process worth flagging.
    """
    n_frames = len(scores)

    if n_frames < MIN_FRAMES_REQUIRED:
        return TemporalAnalysis(
            frame_count=n_frames, mean_score=0.5 if n_frames == 0 else float(np.mean(scores)),
            smoothed_scores=[], std_dev=0.0, temporal_score=0.5, abstained=True,
        )

    smoothed = moving_average(scores, window=window)
    mean_score = float(np.mean(smoothed))
    std_dev = score_variance(scores)

    # std_dev is naturally bounded by [0, 0.5] for scores in [0,1]; scale
    # into a comparable [0,1] penalty range before weighting.
    normalized_instability = min(1.0, std_dev / 0.5)
    temporal_score = mean_score + CONSISTENCY_PENALTY_WEIGHT * normalized_instability
    temporal_score = round(min(1.0, temporal_score), 4)

    return TemporalAnalysis(
        frame_count=n_frames,
        mean_score=round(mean_score, 4),
        smoothed_scores=[round(s, 4) for s in smoothed],
        std_dev=round(std_dev, 4),
        temporal_score=temporal_score,
        abstained=False,
    )


def aggregate_video_scores(face_frame_scores: dict[int, list[float]]) -> list[dict]:
    """face_frame_scores: {face_id: [per-frame evidence scores across the
    sampled video]}. Returns the fixed contract from configs/pipeline.yaml:

        [{"face_id": int, "temporal_score": float}, ...]
    """
    results = []
    for face_id in sorted(face_frame_scores.keys()):
        analysis = aggregate_temporal_score(face_frame_scores[face_id])
        results.append({"face_id": face_id, "temporal_score": analysis.temporal_score})
    return results
