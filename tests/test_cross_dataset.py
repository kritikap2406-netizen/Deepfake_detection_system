"""
Unit tests for Phase 20 — Cross-Dataset Evaluation.

Run with:
    pytest tests/test_cross_dataset.py -v

Note: the full training-based evaluation tests are slow (retrain a
spatial model per fold) and are marked accordingly. Fold-construction
and leakage-safety tests run fast and require no training.
"""

from pathlib import Path

import pytest

from evaluation.cross_dataset import (
    load_rows, leave_one_dataset_out_folds,
    run_single_stream_cross_dataset_evaluation,
    run_multi_stream_cross_dataset_evaluation,
    generate_cross_dataset_report,
    METADATA_CSV, FUSION_DATASET_CSV,
)

FUSION_MODEL = Path("models/fusion/best_fusion_model.joblib")


# --- Fold construction (fast, no training) ---

def test_load_rows_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_rows(Path("/nonexistent/file.csv"))


@pytest.mark.skipif(not METADATA_CSV.exists(), reason="Run `python -m preprocessing.build_metadata` first")
def test_folds_cover_every_dataset_exactly_once_as_holdout():
    rows = load_rows(METADATA_CSV)
    folds = leave_one_dataset_out_folds(rows)

    dataset_names = sorted({r["dataset"] for r in rows})
    held_out_names = sorted(f[0] for f in folds)
    assert held_out_names == dataset_names


@pytest.mark.skipif(not METADATA_CSV.exists(), reason="Run `python -m preprocessing.build_metadata` first")
def test_folds_have_no_leakage():
    """The held-out dataset must never appear in its own fold's train
    rows, and test rows must contain ONLY the held-out dataset."""
    rows = load_rows(METADATA_CSV)
    folds = leave_one_dataset_out_folds(rows)

    for held_out, train_rows, test_rows in folds:
        train_datasets = {r["dataset"] for r in train_rows}
        test_datasets = {r["dataset"] for r in test_rows}

        assert held_out not in train_datasets, f"LEAKAGE: {held_out} appears in its own train fold"
        assert test_datasets == {held_out}, f"test fold for {held_out} contains other datasets"


@pytest.mark.skipif(not METADATA_CSV.exists(), reason="Run `python -m preprocessing.build_metadata` first")
def test_folds_train_and_test_partition_all_rows():
    rows = load_rows(METADATA_CSV)
    folds = leave_one_dataset_out_folds(rows)

    for held_out, train_rows, test_rows in folds:
        assert len(train_rows) + len(test_rows) == len(rows)


@pytest.mark.skipif(not FUSION_DATASET_CSV.exists(), reason="Run `python -m fusion.build_dataset` first")
def test_fusion_dataset_folds_also_leakage_free():
    rows = load_rows(FUSION_DATASET_CSV)
    folds = leave_one_dataset_out_folds(rows)

    for held_out, train_rows, test_rows in folds:
        train_datasets = {r["dataset"] for r in train_rows}
        assert held_out not in train_datasets


# --- Multi-stream (fusion) evaluation — fast, no deep model retraining ---

@pytest.mark.skipif(not FUSION_DATASET_CSV.exists(), reason="Run `python -m fusion.build_dataset` first")
def test_multi_stream_evaluation_produces_one_fold_per_dataset():
    result = run_multi_stream_cross_dataset_evaluation()
    rows = load_rows(FUSION_DATASET_CSV)
    n_datasets = len({r["dataset"] for r in rows})
    assert len(result["folds"]) == n_datasets


@pytest.mark.skipif(not FUSION_DATASET_CSV.exists(), reason="Run `python -m fusion.build_dataset` first")
def test_multi_stream_evaluation_includes_leakage_caveat():
    result = run_multi_stream_cross_dataset_evaluation()
    assert "PARTIAL LEAKAGE CAVEAT" in result["leakage_status"]


@pytest.mark.skipif(not FUSION_DATASET_CSV.exists(), reason="Run `python -m fusion.build_dataset` first")
def test_multi_stream_evaluation_metrics_bounded():
    result = run_multi_stream_cross_dataset_evaluation()
    assert 0.0 <= result["average_accuracy"] <= 1.0
    assert 0.0 <= result["average_f1"] <= 1.0
    for fold in result["folds"]:
        assert 0.0 <= fold["metrics"]["accuracy"] <= 1.0


# --- Single-stream (spatial) evaluation — slow, retrains a model per fold ---

@pytest.mark.slow
@pytest.mark.skipif(not METADATA_CSV.exists(), reason="Run `python -m preprocessing.build_metadata` first")
def test_single_stream_evaluation_produces_one_fold_per_dataset():
    result = run_single_stream_cross_dataset_evaluation(epochs=1)
    rows = load_rows(METADATA_CSV)
    n_datasets = len({r["dataset"] for r in rows})
    assert len(result["folds"]) == n_datasets


@pytest.mark.slow
@pytest.mark.skipif(not METADATA_CSV.exists(), reason="Run `python -m preprocessing.build_metadata` first")
def test_single_stream_evaluation_declares_leakage_free_status():
    result = run_single_stream_cross_dataset_evaluation(epochs=1)
    assert "leakage-free" in result["leakage_status"]


@pytest.mark.slow
@pytest.mark.skipif(
    not (METADATA_CSV.exists() and FUSION_DATASET_CSV.exists()),
    reason="Requires both metadata.csv and fusion_dataset.csv",
)
def test_generate_cross_dataset_report_includes_both_streams_and_caveats():
    report = generate_cross_dataset_report(spatial_epochs=1)

    assert "dataset_caveat" in report
    assert "comparison_caveat" in report
    assert report["single_stream"]["stream"] == "spatial (single-stream)"
    assert report["multi_stream"]["stream"] == "fusion (multi-stream)"
