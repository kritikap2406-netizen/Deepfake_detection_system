"""
TrueSight — dataset metadata builder.

Scans the datasets/ folder structure:

    datasets/<DatasetName>/<real|fake>/<video files>

and produces a single metadata CSV with one row per media file:

    file_path, dataset, label, manipulation_type, video_id, person_id, split

`split` is left empty here — it is assigned later by split.py, which is
responsible for avoiding video/identity leakage across train/val/test.

This module does NOT guess manipulation_type or person_id beyond what can
be parsed from the filename/folder; unknown values are recorded as
"unknown" rather than invented, so downstream code can filter honestly.

Usage:
    python -m preprocessing.build_metadata
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path

DATASETS_ROOT = Path("datasets")
OUTPUT_CSV = DATASETS_ROOT / "metadata.csv"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS

# Matches filenames like: person001_vida.mp4 -> person_id=001, video_id=a
FILENAME_PATTERN = re.compile(r"person(?P<person_id>\w+?)_vid(?P<video_id>\w+)\.")


@dataclass
class MediaRecord:
    file_path: str
    dataset: str
    label: str              # "real" | "fake"
    manipulation_type: str  # dataset name for fakes, "none" for real, "unknown" if unclear
    video_id: str
    person_id: str
    split: str = ""          # assigned later by split.py


def parse_identity(filename: str) -> tuple[str, str]:
    """Extract (person_id, video_id) from filename, or ('unknown', filename) if unparsable."""
    match = FILENAME_PATTERN.search(filename)
    if match:
        return match.group("person_id"), match.group("video_id")
    return "unknown", Path(filename).stem


def scan_dataset(dataset_dir: Path) -> list[MediaRecord]:
    records: list[MediaRecord] = []
    dataset_name = dataset_dir.name

    for label in ("real", "fake"):
        label_dir = dataset_dir / label
        if not label_dir.exists():
            continue

        for file_path in sorted(label_dir.iterdir()):
            if file_path.suffix.lower() not in MEDIA_EXTENSIONS:
                continue

            person_id, video_id = parse_identity(file_path.name)
            manipulation_type = "none" if label == "real" else dataset_name

            records.append(
                MediaRecord(
                    file_path=str(file_path),
                    dataset=dataset_name,
                    label=label,
                    manipulation_type=manipulation_type,
                    video_id=video_id,
                    person_id=person_id,
                )
            )

    return records


def build_metadata(datasets_root: Path = DATASETS_ROOT) -> list[MediaRecord]:
    all_records: list[MediaRecord] = []

    if not datasets_root.exists():
        raise FileNotFoundError(f"Datasets root not found: {datasets_root}")

    for dataset_dir in sorted(datasets_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        all_records.extend(scan_dataset(dataset_dir))

    return all_records


def write_csv(records: list[MediaRecord], output_path: Path = OUTPUT_CSV) -> None:
    if not records:
        raise ValueError("No records to write — did you scan any datasets?")

    fieldnames = list(asdict(records[0]).keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def main() -> None:
    records = build_metadata()
    write_csv(records)

    print(f"Scanned {len(records)} media files across "
          f"{len({r.dataset for r in records})} datasets.")
    print(f"Metadata written to: {OUTPUT_CSV}")

    real_count = sum(1 for r in records if r.label == "real")
    fake_count = sum(1 for r in records if r.label == "fake")
    print(f"  real: {real_count}  fake: {fake_count}")


if __name__ == "__main__":
    main()
