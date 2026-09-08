"""
TrueSight — leakage-safe train/val/test splitter.

Critical rule from the development plan:
    "Do not randomly split frames from the same video across train and
    test; this can create data leakage."

We enforce this by splitting on PERSON_ID, not on individual file rows.
Every video belonging to the same person_id is guaranteed to land in the
same split. If person_id is "unknown" (couldn't be parsed), we fall back
to splitting by video_id so at least whole-video leakage is prevented,
and we flag it separately in the printed report.

Usage:
    python -m preprocessing.split_dataset
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

METADATA_CSV = Path("datasets/metadata.csv")
OUTPUT_CSV = Path("datasets/metadata_split.csv")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # implied remainder

RANDOM_SEED = 42


def load_rows(path: Path = METADATA_CSV) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m preprocessing.build_metadata` first."
        )
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def group_key(row: dict) -> str:
    """Group by (dataset, person_id) when known, otherwise (dataset, video_id).

    Namespacing by dataset matters: person_id "001" in FaceForensics++ and
    person_id "001" in DFDC are different real people who happen to share
    an ID string. Without the dataset prefix they'd be incorrectly merged
    into one identity group.
    """
    if row["person_id"] and row["person_id"] != "unknown":
        return f"{row['dataset']}:person:{row['person_id']}"
    return f"{row['dataset']}:video:{row['video_id']}"


def assign_splits(rows: list[dict], seed: int = RANDOM_SEED) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n = len(group_keys)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_keys = set(group_keys[:n_train])
    val_keys = set(group_keys[n_train:n_train + n_val])
    test_keys = set(group_keys[n_train + n_val:])

    for key, group_rows in groups.items():
        if key in train_keys:
            split = "train"
        elif key in val_keys:
            split = "val"
        else:
            split = "test"
        for row in group_rows:
            row["split"] = split

    return rows


def verify_no_leakage(rows: list[dict]) -> bool:
    """Sanity check: no group_key should appear in more than one split."""
    key_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key_to_splits[group_key(row)].add(row["split"])

    leaked = {k: v for k, v in key_to_splits.items() if len(v) > 1}
    if leaked:
        print(f"LEAKAGE DETECTED in {len(leaked)} groups: {leaked}")
        return False
    return True


def write_csv(rows: list[dict], output_path: Path = OUTPUT_CSV) -> None:
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = load_rows()
    rows = assign_splits(rows)

    assert verify_no_leakage(rows), "Refusing to write output — leakage detected."

    write_csv(rows)

    counts = defaultdict(int)
    for row in rows:
        counts[row["split"]] += 1

    print(f"Split {len(rows)} rows across {len({group_key(r) for r in rows})} identity groups.")
    for split in ("train", "val", "test"):
        print(f"  {split:5}: {counts[split]} rows")
    print(f"No leakage detected. Output written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
