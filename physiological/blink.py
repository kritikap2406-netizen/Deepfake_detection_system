"""
TrueSight — blink detection.

Consumes a sequence of per-frame EAR values (one per tracked face across
a video) and detects blink events: EAR dips below a threshold for at
least one frame, then recovers. From detected blinks we derive rate and
duration statistics and compare them against natural human ranges to
produce a plausibility score.

Natural human blink reference ranges (widely cited in ophthalmology /
human-factors literature):
  - Rate: roughly 12-20 blinks per minute at rest (0.2-0.33 Hz), though
    this varies a lot with context (screen use, lighting, fatigue).
  - Duration: roughly 100-400ms per blink.

Deepfakes historically under-blink (a well-known early deepfake
artifact) or, in some generation methods, blink at mechanically regular
intervals. Either extreme — near-zero blink rate or suspiciously uniform
timing — is treated as fake-leaning evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from configs_loader import load_config

_config = load_config()
EAR_BLINK_THRESHOLD = _config["physiological"]["ear_blink_threshold"]
MIN_FRAMES_REQUIRED = _config["physiological"]["min_frames_required"]

# Natural blink-rate plausibility band, in blinks per minute.
PLAUSIBLE_BLINK_RATE_RANGE = (8.0, 30.0)
PLAUSIBLE_DURATION_RANGE_SEC = (0.05, 0.5)


@dataclass
class BlinkAnalysis:
    blink_count: int
    blink_rate_per_min: float
    avg_duration_sec: float | None
    duration_std_sec: float | None   # low variance + regular spacing => suspicious
    blink_score: float               # 0 (plausible/real-like) - 1 (implausible/fake-like)
    abstained: bool                  # True if too few frames to judge


def detect_blinks(ear_sequence: list[float], threshold: float = EAR_BLINK_THRESHOLD) -> list[tuple[int, int]]:
    """Returns list of (start_frame, end_frame) index pairs, one per detected blink."""
    blinks = []
    in_blink = False
    start = None

    for i, ear in enumerate(ear_sequence):
        below = ear < threshold
        if below and not in_blink:
            in_blink = True
            start = i
        elif not below and in_blink:
            in_blink = False
            blinks.append((start, i - 1))

    if in_blink:  # sequence ended mid-blink
        blinks.append((start, len(ear_sequence) - 1))

    return blinks


def analyze_blinks(ear_sequence: list[float], fps: float) -> BlinkAnalysis:
    n_frames = len(ear_sequence)

    if n_frames < MIN_FRAMES_REQUIRED:
        return BlinkAnalysis(
            blink_count=0, blink_rate_per_min=0.0, avg_duration_sec=None,
            duration_std_sec=None, blink_score=0.5, abstained=True,
        )

    blinks = detect_blinks(ear_sequence)
    duration_sec = n_frames / fps
    blink_rate_per_min = (len(blinks) / duration_sec) * 60.0 if duration_sec > 0 else 0.0

    if blinks:
        durations = [((end - start + 1) / fps) for start, end in blinks]
        avg_duration = float(np.mean(durations))
        duration_std = float(np.std(durations)) if len(durations) > 1 else 0.0
    else:
        avg_duration = None
        duration_std = None

    blink_score = _score_plausibility(blink_rate_per_min, avg_duration, duration_std, len(blinks))

    return BlinkAnalysis(
        blink_count=len(blinks),
        blink_rate_per_min=blink_rate_per_min,
        avg_duration_sec=avg_duration,
        duration_std_sec=duration_std,
        blink_score=blink_score,
        abstained=False,
    )


def _score_plausibility(
    rate_per_min: float,
    avg_duration: float | None,
    duration_std: float | None,
    blink_count: int,
) -> float:
    """Combine rate + duration evidence into a single 0 (real-like) - 1
    (fake-like) score. Kept as simple, inspectable rules rather than a
    learned model — this is a signal-processing module per the plan, and
    a hand-tunable scoring function is easier to debug/demo than an
    opaque one for a hackathon prototype.
    """
    lo, hi = PLAUSIBLE_BLINK_RATE_RANGE
    if lo <= rate_per_min <= hi:
        rate_penalty = 0.0
    else:
        distance = min(abs(rate_per_min - lo), abs(rate_per_min - hi))
        rate_penalty = min(1.0, distance / hi)  # scale penalty by how far outside the band

    if avg_duration is None:
        duration_penalty = 0.6  # zero blinks at all over a long enough clip is itself suspicious
    else:
        d_lo, d_hi = PLAUSIBLE_DURATION_RANGE_SEC
        if d_lo <= avg_duration <= d_hi:
            duration_penalty = 0.0
        else:
            duration_penalty = 0.5

    # Suspiciously mechanical regularity (near-zero variance with >=3 blinks)
    # is itself a red flag independent of rate/duration being individually fine.
    regularity_penalty = 0.0
    if blink_count >= 3 and duration_std is not None and duration_std < 0.01:
        regularity_penalty = 0.3

    score = min(1.0, rate_penalty * 0.5 + duration_penalty * 0.4 + regularity_penalty)
    return round(score, 4)
