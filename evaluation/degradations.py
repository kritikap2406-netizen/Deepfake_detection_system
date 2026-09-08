"""
TrueSight — robustness degradation functions.

Phase 4's spatial/augmentation.py applies *randomized* degradations for
training-time data augmentation. Robustness *testing* needs the
opposite: fixed, reproducible severities so results are comparable
across runs and across models. This module wraps the same underlying
transforms (JPEG, blur, noise, resize) at controlled severity levels,
and adds two the training augmentation didn't need: low-light
simulation and social-media-style recompression.
"""

from __future__ import annotations

import cv2
import numpy as np


def jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return image
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def gaussian_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return image
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def gaussian_noise(image: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def resize_degrade(image: np.ndarray, scale: float) -> np.ndarray:
    """Downscale then upscale back — simulates a low-resolution source
    (e.g. an old webcam, a heavily downscaled upload) rather than
    genuinely reducing output resolution, so downstream shape contracts
    stay intact."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def low_light(image: np.ndarray, brightness_factor: float, noise_std: float = 8.0) -> np.ndarray:
    """Simulates a dim lighting condition: scales pixel values down
    (less light reaching the sensor) and adds proportionally more sensor
    noise (real camera sensors get noisier at low light / high ISO, not
    just darker) — a plain brightness multiply alone wouldn't capture
    that second, equally important effect.
    """
    darkened = image.astype(np.float32) * brightness_factor
    noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
    result = darkened + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def social_media_compression(image: np.ndarray) -> np.ndarray:
    """Simulates the compounding degradation a video/image undergoes
    when uploaded to and re-served by a social platform: downscale,
    aggressive JPEG re-encode, then a second lighter re-encode (many
    platforms transcode on upload and again on each re-share/re-download).
    """
    degraded = resize_degrade(image, scale=0.5)
    degraded = jpeg_compress(degraded, quality=40)
    degraded = jpeg_compress(degraded, quality=60)
    return degraded


# Named severity presets so robustness tests can iterate over a
# consistent, documented set of conditions rather than ad-hoc parameters
# scattered across test files.
DEGRADATION_SUITE: dict[str, callable] = {
    "jpeg_low_quality": lambda img: jpeg_compress(img, quality=20),
    "jpeg_medium_quality": lambda img: jpeg_compress(img, quality=50),
    "blur_mild": lambda img: gaussian_blur(img, kernel_size=3),
    "blur_heavy": lambda img: gaussian_blur(img, kernel_size=9),
    "noise_mild": lambda img: gaussian_noise(img, std=5.0),
    "noise_heavy": lambda img: gaussian_noise(img, std=20.0),
    "low_resolution": lambda img: resize_degrade(img, scale=0.25),
    "low_light": lambda img: low_light(img, brightness_factor=0.35),
    "social_media_compression": social_media_compression,
}
