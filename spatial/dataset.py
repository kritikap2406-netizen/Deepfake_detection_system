"""
TrueSight — spatial stream dataset adapter.

Bridges preprocessing.dataset_loader.TrueSightDataset (file paths + labels
from the leakage-safe split) to a real torch.utils.data.Dataset that
yields normalized 224x224 tensors + binary labels.

Augmentation (Phase 4's JPEG/blur/color/resize/noise) is applied only to
the train split — val/test must reflect realistic, unmodified evaluation
conditions per standard practice.

NOTE: at this prototype stage, dataset media files are placeholders (see
Phase 2), so `_load_face_crop` falls back to a deterministic synthetic
crop when the real file can't be decoded. This keeps the training loop
fully runnable and testable now; swap in real face detection (Phase 3's
pipeline.preprocess_image/video) once actual video/image bytes are
present in datasets/.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.dataset_loader import TrueSightDataset
from spatial.augmentation import apply_random_augmentations

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_face_crop(file_path: str, label: int, face_size: int = 224) -> np.ndarray:
    """Load an aligned face crop for `file_path`.

    Real path (once datasets/ has actual media): decode the file, run it
    through preprocessing.pipeline, take the first detected face.

    Prototype fallback: our stub dataset files are empty placeholders, so
    we deterministically synthesize a crop from a hash of the path. This
    is clearly NOT real training signal — it exists purely so the training
    loop, loss curve, and checkpointing logic can be exercised end-to-end
    before real datasets are dropped in.
    """
    import cv2

    try:
        img = cv2.imread(file_path)
        if img is not None:
            return cv2.resize(img, (face_size, face_size))
    except Exception:
        pass

    rng = np.random.default_rng(abs(hash(file_path)) % (2**32))
    base_color = (200, 170, 150) if label == 0 else (150, 170, 200)
    crop = np.full((face_size, face_size, 3), base_color, dtype=np.uint8)
    noise = rng.normal(0, 15, crop.shape)
    crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return crop


def normalize_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC -> normalized float32 CHW tensor (ImageNet stats)."""
    import cv2

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(chw).float()


class SpatialDataset(Dataset):
    def __init__(self, split: str, face_size: int = 224, augment: bool | None = None):
        self.base = TrueSightDataset(split=split)
        self.face_size = face_size
        # augment defaults to True only for train unless explicitly overridden
        self.augment = augment if augment is not None else (split == "train")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        file_path, label, meta = self.base[index]

        crop = _load_face_crop(file_path, label, self.face_size)
        if self.augment:
            crop = apply_random_augmentations(crop)

        tensor = normalize_to_tensor(crop)
        return tensor, torch.tensor(float(label)), meta
