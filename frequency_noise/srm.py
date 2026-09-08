"""
TrueSight — SRM (Spatial Rich Model) noise residual pipeline.

Real camera sensors leave a characteristic noise fingerprint (photo-
response non-uniformity, demosaicing artifacts, sensor read noise).
Manipulation -- blending a generated face into a frame, re-encoding,
smoothing -- disturbs this fingerprint in ways that are usually invisible
to the eye but show up clearly once the "obvious" image content (edges,
texture, color) is suppressed by a high-pass filter.

SRM filters, from steganalysis / image-forensics literature, are
predictor kernels: each one predicts a pixel's value from its neighbors,
and the *residual* (actual - predicted) is what's left over after
removing predictable local structure. That residual is where sensor
noise and manipulation traces live.

We use three standard fixed (non-trainable) 5x5 SRM kernels, a small,
well-established subset of the classic "rich model" filter bank:
  - KV kernel: a strong 2nd-order predictor, the single most widely used
    SRM filter in forensic CNN literature.
  - A horizontal 2nd-order edge predictor.
  - A compact Laplacian-style edge predictor.

Being fixed math (not learned weights), these are cheap, require no
training, and give the noise classifier a clean, interpretable input.
"""

from __future__ import annotations

import cv2
import numpy as np

# Kirchner/Fridrich "KV" kernel -- the most common single SRM filter.
_KV_KERNEL = np.array([
    [-1,  2,  -2,  2, -1],
    [ 2, -6,   8, -6,  2],
    [-2,  8, -12,  8, -2],
    [ 2, -6,   8, -6,  2],
    [-1,  2,  -2,  2, -1],
], dtype=np.float32) / 12.0

# Horizontal 2nd-order edge predictor.
_HORIZONTAL_KERNEL = np.array([
    [0, 0,  0, 0, 0],
    [0, 0,  0, 0, 0],
    [0, 1, -2, 1, 0],
    [0, 0,  0, 0, 0],
    [0, 0,  0, 0, 0],
], dtype=np.float32) / 4.0

# Compact Laplacian-style edge predictor.
_LAPLACIAN_KERNEL = np.array([
    [0,  0,  0,  0, 0],
    [0, -1,  2, -1, 0],
    [0,  2, -4,  2, 0],
    [0, -1,  2, -1, 0],
    [0,  0,  0,  0, 0],
], dtype=np.float32) / 2.0

SRM_KERNELS = [_KV_KERNEL, _HORIZONTAL_KERNEL, _LAPLACIAN_KERNEL]


def apply_srm_filter(gray_image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply one SRM kernel to a single-channel image -> raw residual."""
    if gray_image.ndim != 2:
        raise ValueError(f"apply_srm_filter expects a single-channel image, got shape {gray_image.shape}")
    return cv2.filter2D(gray_image.astype(np.float32), ddepth=-1, kernel=kernel)


def compute_srm_residuals(gray_image: np.ndarray) -> np.ndarray:
    """Apply all SRM kernels -> stacked raw residuals, shape (3, H, W)."""
    residuals = [apply_srm_filter(gray_image, k) for k in SRM_KERNELS]
    return np.stack(residuals, axis=0)


def normalize_residual(residual: np.ndarray, clip_std: float = 3.0) -> np.ndarray:
    """SRM residuals are near-zero-mean with heavy tails (most pixels are
    predictable, a few are strongly not). Clip to a few standard
    deviations before min-max scaling so outlier pixels don't wash out
    the rest of the signal into a flat mid-gray.
    """
    mean, std = residual.mean(), residual.std()
    if std < 1e-8:
        return np.zeros_like(residual)

    clipped = np.clip(residual, mean - clip_std * std, mean + clip_std * std)
    min_val, max_val = clipped.min(), clipped.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(residual)
    return (clipped - min_val) / (max_val - min_val)


def prepare_srm_input(face_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """Full pipeline: BGR face crop -> grayscale -> resized -> 3-channel
    normalized SRM residual stack, ready for the noise classifier.
    Output shape: (3, size, size).
    """
    if face_bgr.shape[:2] != (size, size):
        face_bgr = cv2.resize(face_bgr, (size, size))
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

    raw_residuals = compute_srm_residuals(gray)
    normalized = np.stack([normalize_residual(r) for r in raw_residuals], axis=0)
    return normalized.astype(np.float32)
