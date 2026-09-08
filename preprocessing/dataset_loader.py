"""
TrueSight — shared dataset loader.

This is the single reusable Dataset class that every stream (spatial,
frequency, noise, physiological) should use to read training data. It
reads from datasets/metadata_split.csv, which already has leakage-safe
train/val/test assignments (see split_dataset.py).

Individual streams differ in what *transform* they apply to the loaded
face image (raw pixels for spatial, FFT/DCT for frequency, SRM residual
for noise) — not in how they read the dataset. Keeping this shared avoids
each of the 6 team members quietly building a slightly different loader.

Usage:
    from preprocessing.dataset_loader import TrueSightDataset

    train_set = TrueSightDataset(split="train")
    image_or_path, label, meta = train_set[0]
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Optional

METADATA_SPLIT_CSV = Path("datasets/metadata_split.csv")

LABEL_TO_INT = {"real": 0, "fake": 1}


class TrueSightDataset:
    """Minimal, dependency-light Dataset. Swap in torch.utils.data.Dataset
    once the team is ready to wire up actual training loops — the indexing
    interface below (`__len__`, `__getitem__`) is already compatible.
    """

    def __init__(
        self,
        split: str,
        metadata_csv: Path = METADATA_SPLIT_CSV,
        transform: Optional[Callable] = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.split = split
        self.transform = transform
        self.rows = self._load_rows(metadata_csv, split)

        if not self.rows:
            raise ValueError(
                f"No rows found for split={split!r}. "
                f"Did you run build_metadata.py and split_dataset.py?"
            )

    @staticmethod
    def _load_rows(metadata_csv: Path, split: str) -> list[dict]:
        if not metadata_csv.exists():
            raise FileNotFoundError(
                f"{metadata_csv} not found — run "
                f"`python -m preprocessing.build_metadata` then "
                f"`python -m preprocessing.split_dataset` first."
            )
        with open(metadata_csv, newline="") as f:
            all_rows = list(csv.DictReader(f))
        return [row for row in all_rows if row["split"] == split]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        file_path = row["file_path"]
        label = LABEL_TO_INT[row["label"]]

        # NOTE: prototype stage — we return the file_path itself rather
        # than decoded pixels, since the stub dataset contains empty
        # placeholder files, not real video/image bytes. Once real media
        # is dropped into datasets/, this is where decoding + the
        # preprocessing.pipeline face-alignment step gets called.
        sample = file_path
        if self.transform is not None:
            sample = self.transform(sample)

        meta = {
            "dataset": row["dataset"],
            "manipulation_type": row["manipulation_type"],
            "video_id": row["video_id"],
            "person_id": row["person_id"],
        }
        return sample, label, meta

    def class_balance(self) -> dict:
        counts = {"real": 0, "fake": 0}
        for row in self.rows:
            counts[row["label"]] += 1
        return counts


def _demo():
    for split in ("train", "val", "test"):
        ds = TrueSightDataset(split=split)
        print(f"{split:5} — {len(ds)} samples, balance: {ds.class_balance()}")
        sample, label, meta = ds[0]
        print(f"      example: {sample} label={label} meta={meta}")


if __name__ == "__main__":
    _demo()
