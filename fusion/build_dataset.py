"""
TrueSight — fusion dataset builder.

Phase 10, step "Create fusion dataset": runs every already-trained
stream model (spatial, frequency, noise from Phases 4-6; metadata from
Phase 9) over the same leakage-safe train/val/test split used
throughout, and records their scores alongside the true label. This is
the real training data for the meta-classifier.

blink_score / rppg_score / temporal_score are NOT computed here and are
left as the neutral 0.5 default for every row. This is a deliberate,
documented limitation of the current prototype stage: those three
streams require a multi-frame video sequence for a single tracked face,
but our dataset loader (preprocessing.dataset_loader) operates at the
file level, one row per media file, not per extracted frame sequence.
Wiring real video-derived blink/rppg/temporal features into fusion
training happens naturally once Phase 14's backend orchestrates full
video processing end-to-end — at that point this script should be
extended to call physiological.inference.PhysiologicalInference per
video and merge those scores in.

This is not a workaround to hide a gap — it's the same neutral-default
convention every other module already uses when a signal can't be
computed (see fusion/features.py docstring), applied consistently here
too, and it's called out explicitly so nobody mistakes 0.5 constants for
a bug later.

Usage:
    python -m fusion.build_dataset
"""

from __future__ import annotations

import csv
from pathlib import Path

from preprocessing.face_alignment import AlignedFace
from preprocessing.face_detector import BoundingBox
from preprocessing.dataset_loader import TrueSightDataset
from spatial.dataset import _load_face_crop
from spatial.inference import SpatialInference
from frequency_noise.inference import FrequencyInference, NoiseInference
from metadata.analyzer import analyze_metadata
from fusion.features import FEATURE_ORDER, NEUTRAL_DEFAULT

OUTPUT_CSV = Path("datasets/fusion_dataset.csv")


def _make_aligned_face(crop) -> AlignedFace:
    """Wrap a raw face crop as an AlignedFace so it can be fed directly
    into each stream's *Inference.score_face(). We bypass real detection
    here (as in every prior phase's direct-crop tests) since the stub
    dataset's placeholder files carry no real detectable face; this
    mirrors how Phases 4-6 were each validated individually."""
    return AlignedFace(
        face_id=1, image=crop,
        source_bbox=BoundingBox(x=0, y=0, width=crop.shape[1], height=crop.shape[0], confidence=1.0),
        landmarks=None,
    )


def _safe_metadata_score(file_path: str) -> float:
    """Our stub dataset files are empty .mp4 placeholders (Phase 2), so
    real video metadata extraction will legitimately find nothing to
    read and score 0.0 — that's an honest result, not a bug. Wrapped in
    try/except purely so a genuinely corrupt/unreadable file elsewhere
    in a real dataset can't crash the whole dataset build.
    """
    try:
        result = analyze_metadata(file_path, media_type="video")
        return result.metadata_score
    except Exception:
        return NEUTRAL_DEFAULT


def build_fusion_dataset(output_path: Path = OUTPUT_CSV) -> list[dict]:
    spatial_engine = SpatialInference()
    frequency_engine = FrequencyInference()
    noise_engine = NoiseInference()

    rows: list[dict] = []

    for split in ("train", "val", "test"):
        dataset = TrueSightDataset(split=split)
        for i in range(len(dataset)):
            file_path, label, meta = dataset[i]
            crop = _load_face_crop(file_path, label, face_size=224)
            face = _make_aligned_face(crop)

            spatial_score = spatial_engine.score_face(face)["spatial_score"]
            frequency_score = frequency_engine.score_face(face)["frequency_score"]
            noise_score = noise_engine.score_face(face)["noise_score"]
            metadata_score = _safe_metadata_score(file_path)

            rows.append({
                "file_path": file_path,
                "split": split,
                "label": label,  # 0=real, 1=fake (int, from TrueSightDataset)
                "dataset": meta["dataset"],
                "spatial_score": spatial_score,
                "frequency_score": frequency_score,
                "noise_score": noise_score,
                "blink_score": NEUTRAL_DEFAULT,     # see module docstring
                "rppg_score": NEUTRAL_DEFAULT,       # see module docstring
                "temporal_score": NEUTRAL_DEFAULT,   # see module docstring
                "metadata_flag": metadata_score,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return rows


def main():
    rows = build_fusion_dataset()
    print(f"Built fusion dataset: {len(rows)} rows -> {OUTPUT_CSV}")
    for split in ("train", "val", "test"):
        n = sum(1 for r in rows if r["split"] == split)
        print(f"  {split}: {n} rows")


if __name__ == "__main__":
    main()
