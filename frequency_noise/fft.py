"""
TrueSight — FFT (Fast Fourier Transform) pipeline.

Complements the DCT stream. FFT gives us the full frequency spectrum
including a natural low-frequency-centered visualization (via fftshift),
which is the standard way GAN-fingerprint / spectral-artifact papers
present frequency evidence. We discard phase and keep only the magnitude
spectrum, since phase mostly encodes spatial position information, while
magnitude captures the periodic-texture artifacts we care about.
"""

from __future__ import annotations

import cv2
import numpy as np


def compute_fft_magnitude(gray_image: np.ndarray) -> np.ndarray:
    """2D FFT of a single-channel image, shifted so low frequencies are
    centered (standard convention for visualizing/analyzing spectra)."""
    if gray_image.ndim != 2:
        raise ValueError(f"compute_fft_magnitude expects a single-channel image, got shape {gray_image.shape}")

    float_img = gray_image.astype(np.float32)
    fft = np.fft.fft2(float_img)
    fft_shifted = np.fft.fftshift(fft)
    return np.abs(fft_shifted)


def fft_log_magnitude_normalized(gray_image: np.ndarray) -> np.ndarray:
    """Log-scaled, min-max normalized FFT magnitude spectrum -> [0, 1]."""
    magnitude = compute_fft_magnitude(gray_image)
    log_mag = np.log(magnitude + 1e-6)

    min_val, max_val = log_mag.min(), log_mag.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(log_mag)
    return (log_mag - min_val) / (max_val - min_val)


def azimuthal_average(spectrum: np.ndarray, num_bins: int = 50) -> np.ndarray:
    """Radially average the spectrum around its center into `num_bins`
    values. This 1D radial power profile is a well-known compact feature
    for detecting GAN upsampling artifacts (which show up as periodic
    bumps at specific radii) and is cheap to feed into a lightweight
    classifier alongside the 2D representation.
    """
    h, w = spectrum.shape
    center_y, center_x = h // 2, w // 2

    y_indices, x_indices = np.indices((h, w))
    radii = np.sqrt((y_indices - center_y) ** 2 + (x_indices - center_x) ** 2)

    max_radius = radii.max()
    bin_edges = np.linspace(0, max_radius, num_bins + 1)

    profile = np.zeros(num_bins, dtype=np.float32)
    for i in range(num_bins):
        mask = (radii >= bin_edges[i]) & (radii < bin_edges[i + 1])
        if mask.any():
            profile[i] = spectrum[mask].mean()

    return profile


def prepare_fft_input(face_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    """Full pipeline: BGR face crop -> grayscale -> resized -> normalized
    log-magnitude FFT spectrum, ready to feed into the frequency model.
    """
    if face_bgr.shape[:2] != (size, size):
        face_bgr = cv2.resize(face_bgr, (size, size))
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return fft_log_magnitude_normalized(gray)
