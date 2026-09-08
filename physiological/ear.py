"""
TrueSight — Eye Aspect Ratio (EAR).

The standard formula from Soukupova & Cech's eye-blink-detection work,
widely used across face-analysis literature:

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

where p1..p6 trace the eye outline: p1/p4 are the horizontal corners,
p2/p3 are the top eyelid, p5/p6 are the bottom eyelid. EAR stays roughly
constant while the eye is open and drops sharply toward zero when the eye
closes, since the vertical eyelid distance shrinks while the horizontal
corner distance doesn't.
"""

from __future__ import annotations

import numpy as np


def _euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def eye_aspect_ratio(eye_points: list[tuple[float, float]]) -> float:
    """eye_points: 6 points in order [corner_L, top1, top2, corner_R, bottom2, bottom1]
    (matches the point order produced by physiological/landmarks.py for
    both the MediaPipe and geometric-fallback backends).
    """
    if len(eye_points) != 6:
        raise ValueError(f"eye_aspect_ratio expects exactly 6 points, got {len(eye_points)}")

    p1, p2, p3, p4, p5, p6 = eye_points

    vertical_1 = _euclidean(p2, p6)
    vertical_2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)

    if horizontal < 1e-6:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def average_ear(left_eye: list[tuple[float, float]], right_eye: list[tuple[float, float]]) -> float:
    """Average EAR across both eyes — more robust to single-eye occlusion
    or noisy landmark estimates than using one eye alone."""
    return (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
