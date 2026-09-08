"""
TrueSight — DCT (Discrete Cosine Transform) pipeline.

Deepfake generation (GANs, diffusion upsampling, autoencoder decoders)
tends to leave periodic artifacts in the frequency domain that aren't
visible to the naked eye in the spatial domain. DCT represents the image
as a sum of cosine waves at different frequencies -- exactly the
representation JPEG compression uses -- which makes it a natural place to
look for these generation artifacts.

We start with DCT (per the plan) because unlike FFT it produces a single
real-valued matrix (no complex numbers, no phase component to discard),
which makes it simpler to inspect, visualize, and reason about first.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_dct(gray_image: np.ndarray) -> np.ndarray:
    """2D DCT of a single-channel float32 image. Input values should be
    in [0, 255]; internally converted to float32 as OpenCV's dct requires.
    """
    if gray_image.ndim != 2:
        raise ValueError(f"compute_dct expects a single-channel image, got shape {gray_image.shape}")

    float_img = gray_image.astype(np.float32)
    return cv2.dct(float_img)


def dct_log_magnitude(gray_image: np.ndarray) -> np.ndarray:
    """DCT coefficients span a huge dynamic range (low frequencies dominate
    massively over high frequencies), so raw coefficients are nearly
    invisible/unusable directly. Log-scaling compresses that range into
    something a CNN (and a human) can actually make use of.
    """
    dct_coeffs = compute_dct(gray_image)
    return np.log(np.abs(dct_coeffs) + 1e-6)


def dct_log_magnitude_normalized(gray_image: np.ndarray) -> np.ndarray:
    """Log-magnitude DCT, min-max normalized to [0, 1] for use as a model
    input channel alongside other representations (e.g. FFT)."""
    log_mag = dct_log_magnitude(gray_image)
    min_val, max_val = log_mag.min(), log_mag.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(log_mag)
    return (log_mag - min_val) / (max_val - min_val)


def prepare_dct_input(face_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """Full pipeline: BGR face crop -> grayscale -> resized -> normalized
    log-magnitude DCT, ready to feed into the frequency model.
    """
    if face_bgr.shape[:2] != (size, size):
        face_bgr = cv2.resize(face_bgr, (size, size))
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return dct_log_magnitude_normalized(gray)
