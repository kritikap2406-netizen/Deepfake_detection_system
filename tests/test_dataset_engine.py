"""
Unit tests for Phase 2 — Dataset Engine.

Run with:
    pytest tests/test_dataset_engine.py -v
"""

from pathlib import Path

import pytest

from preprocessing.build_metadata import build_metadata, scan_dataset
from preprocessing.split_dataset import load_rows, assign_splits, verify_no_leakage, group_key
from preprocessing.dataset_loader import TrueSightDataset


DATASETS_ROOT = Path("datasets")


def test_metadata_scan_finds_files():
    records = build_metadata(DATASETS_ROOT)
    assert len(records) > 0, "Expected at least one media file in datasets/"


def test_metadata_labels_are_valid():
    records = build_metadata(DATASETS_ROOT)
    for r in records:
        assert r.label in {"real", "fake"}


def test_real_records_have_no_manipulation_type():
    records = build_metadata(DATASETS_ROOT)
    for r in records:
        if r.label == "real":
            assert r.manipulation_type == "none"


def test_split_produces_no_leakage():
    rows = load_rows()
    rows = assign_splits(rows, seed=1)
    assert verify_no_leakage(rows)


def test_split_covers_all_rows():
    rows = load_rows()
    rows = assign_splits(rows, seed=1)
    for row in rows:
        assert row["split"] in {"train", "val", "test"}


def test_group_key_namespaces_by_dataset():
    row_a = {"dataset": "DFDC", "person_id": "001", "video_id": "a"}
    row_b = {"dataset": "CelebDF", "person_id": "001", "video_id": "a"}
    assert group_key(row_a) != group_key(row_b), (
        "Same person_id string in different datasets must not collapse "
        "into the same identity group."
    )


def test_dataset_loader_requires_valid_split():
    with pytest.raises(ValueError):
        TrueSightDataset(split="not_a_real_split")


def test_dataset_loader_returns_expected_shape():
    ds = TrueSightDataset(split="train")
    sample, label, meta = ds[0]
    assert isinstance(sample, str)  # file_path at this prototype stage
    assert label in (0, 1)
    assert {"dataset", "manipulation_type", "video_id", "person_id"} <= meta.keys()
