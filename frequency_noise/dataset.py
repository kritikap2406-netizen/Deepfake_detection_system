"""
TrueSight — frequency stream dataset adapter.

Same leakage-safe split (via preprocessing.dataset_loader) as the spatial
stream, but the model input here is a 2-channel [DCT, FFT] log-magnitude
stack instead of raw RGB pixels.

Prototype note: same fallback as spatial/dataset.py — stub dataset files
are empty placeholders, so we synthesize a deterministic crop when the
real file can't be decoded. See spatial/dataset.py docstring for the full
rationale; swap in real face crops once actual media is in datasets/.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.dataset_loader import TrueSightDataset
from spatial.dataset import _load_face_crop  # reuse the same fallback/loading logic
from frequency_noise.dct import prepare_dct_input
from frequency_noise.fft import prepare_fft_input
from frequency_noise.srm import prepare_srm_input


def prepare_frequency_input(face_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    """BGR face crop -> stacked [DCT, FFT] tensor, shape (2, size, size)."""
    dct_channel = prepare_dct_input(face_bgr, size=size)
    fft_channel = prepare_fft_input(face_bgr, size=size)
    stacked = np.stack([dct_channel, fft_channel], axis=0)  # (2, H, W)
    return torch.from_numpy(stacked).float()


def prepare_noise_input(face_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    """BGR face crop -> 3-channel SRM residual tensor, shape (3, size, size)."""
    residuals = prepare_srm_input(face_bgr, size=size)  # already (3, H, W) float32
    return torch.from_numpy(residuals).float()


class FrequencyDataset(Dataset):
    def __init__(self, split: str, face_size: int = 224):
        self.base = TrueSightDataset(split=split)
        self.face_size = face_size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        file_path, label, meta = self.base[index]

        crop = _load_face_crop(file_path, label, self.face_size)
        tensor = prepare_frequency_input(crop, self.face_size)

        return tensor, torch.tensor(float(label)), meta


class NoiseDataset(Dataset):
    """Same leakage-safe split as FrequencyDataset, but yields a 3-channel
    SRM residual stack instead of [DCT, FFT]. Kept in this same file
    since both streams share dataset-loading/fallback logic and live in
    the same frequency_noise/ module."""

    def __init__(self, split: str, face_size: int = 224):
        self.base = TrueSightDataset(split=split)
        self.face_size = face_size

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        file_path, label, meta = self.base[index]

        crop = _load_face_crop(file_path, label, self.face_size)
        tensor = prepare_noise_input(crop, self.face_size)

        return tensor, torch.tensor(float(label)), meta
