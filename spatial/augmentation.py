"""
TrueSight — spatial stream augmentation.

The plan explicitly calls for JPEG/blur/color/resize/noise augmentation.
The point isn't cosmetic variety — it's robustness: deepfakes encountered
in the wild have been through social-media compression, resizing, and
recompression, and a model trained only on pristine dataset frames will
collapse under that distribution shift (this is exactly what Phase 18 —
Robustness Testing — later measures).

We implement augmentation as plain functions over numpy/OpenCV images
(not decoded tensors) so it can run identically whether the caller is
a PyTorch DataLoader, a quick visualization script, or a notebook.
"""

from __future__ import annotations

import random

import cv2
import numpy as np


def random_jpeg_compression(image: np.ndarray, quality_range: tuple[int, int] = (30, 90)) -> np.ndarray:
    quality = random.randint(*quality_range)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return image
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def random_blur(image: np.ndarray, max_kernel: int = 5) -> np.ndarray:
    k = random.choice([k for k in range(1, max_kernel + 1, 2)])  # odd kernel sizes only
    if k == 1:
        return image
    return cv2.GaussianBlur(image, (k, k), 0)


def random_color_jitter(
    image: np.ndarray,
    brightness_range: tuple[float, float] = (0.8, 1.2),
    contrast_range: tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)

    img = image.astype(np.float32)
    mean = img.mean()
    img = (img - mean) * contrast + mean          # contrast around image mean
    img = img * brightness                         # brightness scale
    return np.clip(img, 0, 255).astype(np.uint8)


def random_resize_degrade(image: np.ndarray, min_scale: float = 0.4, max_scale: float = 1.0) -> np.ndarray:
    """Downscale then upscale back — simulates low-resolution source video."""
    h, w = image.shape[:2]
    scale = random.uniform(min_scale, max_scale)
    if scale >= 0.99:
        return image
    small = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def random_gaussian_noise(image: np.ndarray, std_range: tuple[float, float] = (2.0, 12.0)) -> np.ndarray:
    std = random.uniform(*std_range)
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


AUGMENTATIONS = [
    random_jpeg_compression,
    random_blur,
    random_color_jitter,
    random_resize_degrade,
    random_gaussian_noise,
]


def apply_random_augmentations(image: np.ndarray, max_ops: int = 2, apply_prob: float = 0.5) -> np.ndarray:
    """Apply between 0 and `max_ops` random augmentations, each independently gated.

    Applying all five every time would push training distribution too far
    from realistic degradation; instead each call picks a random subset,
    which better matches "some fraction of real-world videos are heavily
    compressed, others aren't."
    """
    if random.random() > apply_prob:
        return image

    chosen = random.sample(AUGMENTATIONS, k=random.randint(1, max_ops))
    for aug_fn in chosen:
        image = aug_fn(image)
    return image
