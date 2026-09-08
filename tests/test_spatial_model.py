"""
Unit tests for Phase 4 — Spatial Detection Model.

Run with:
    pytest tests/test_spatial_model.py -v

Note: pretrained=True is not tested here since it requires downloading
ImageNet weights from HuggingFace Hub, which needs network access not
guaranteed in every dev/CI environment. Structural correctness (shapes,
loss behavior, contract compliance) is fully covered without it.
"""

from pathlib import Path

import numpy as np
import torch
import pytest

from spatial.model import build_model, SpatialClassifier
from spatial.augmentation import apply_random_augmentations, AUGMENTATIONS
from spatial.losses import FocalLoss, compute_class_weights
from spatial.dataset import SpatialDataset, normalize_to_tensor
from spatial.metrics import compute_binary_metrics
from spatial.inference import SpatialInference
from preprocessing.face_alignment import align_from_bbox
from preprocessing.face_detector import BoundingBox
from preprocessing.frame_extractor import load_single_image

ASSETS = Path("/home/claude/test_assets")
CHECKPOINT = Path("models/spatial/efficientnet_b0.pt")


# --- Model ---

def test_model_builds_without_pretrained_weights():
    model = build_model(backbone="efficientnet_b0", pretrained=False)
    assert isinstance(model, SpatialClassifier)


def test_model_rejects_unknown_backbone():
    with pytest.raises(ValueError):
        build_model(backbone="not_a_real_backbone", pretrained=False)


def test_model_forward_shape():
    model = build_model(pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (4,)


def test_model_predict_proba_in_valid_range():
    model = build_model(pretrained=False)
    x = torch.randn(4, 3, 224, 224)
    probs = model.predict_proba(x)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


# --- Augmentation ---

def test_each_augmentation_preserves_shape_and_dtype():
    img = np.full((224, 224, 3), 150, dtype=np.uint8)
    for fn in AUGMENTATIONS:
        out = fn(img.copy())
        assert out.shape == img.shape
        assert out.dtype == np.uint8


def test_combined_augmentation_preserves_shape():
    img = np.full((224, 224, 3), 150, dtype=np.uint8)
    out = apply_random_augmentations(img.copy())
    assert out.shape == img.shape


# --- Losses ---

def test_focal_loss_downweights_easy_examples():
    fl = FocalLoss()
    easy = fl(torch.tensor([5.0]), torch.tensor([1.0]))   # confident + correct
    hard = fl(torch.tensor([0.0]), torch.tensor([1.0]))   # unsure
    assert easy.item() < hard.item()


def test_focal_loss_is_nonnegative():
    fl = FocalLoss()
    logits = torch.randn(10)
    targets = torch.randint(0, 2, (10,)).float()
    loss = fl(logits, targets)
    assert loss.item() >= 0.0


def test_class_weights_favor_minority_class():
    weights = compute_class_weights([0, 0, 0, 0, 1, 1])  # 4 real, 2 fake
    assert weights[1] > weights[0]


def test_class_weights_raises_on_missing_class():
    with pytest.raises(ValueError):
        compute_class_weights([0, 0, 0])  # no fake examples at all


# --- Dataset ---

def test_normalize_to_tensor_shape_and_type():
    img = np.full((224, 224, 3), 128, dtype=np.uint8)
    tensor = normalize_to_tensor(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_spatial_dataset_train_applies_augmentation_flag():
    ds = SpatialDataset(split="train")
    assert ds.augment is True


def test_spatial_dataset_val_does_not_augment_by_default():
    ds = SpatialDataset(split="val")
    assert ds.augment is False


def test_spatial_dataset_getitem_shape():
    ds = SpatialDataset(split="test")
    tensor, label, meta = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert label.item() in (0.0, 1.0)
    assert "dataset" in meta


# --- Metrics ---

def test_metrics_perfect_predictions():
    labels = [0, 0, 1, 1]
    probs = [0.1, 0.2, 0.9, 0.8]
    m = compute_binary_metrics(labels, probs)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_metrics_handles_single_class_gracefully():
    labels = [1, 1, 1]
    probs = [0.6, 0.7, 0.8]
    m = compute_binary_metrics(labels, probs)
    assert m["roc_auc"] is None  # undefined with one class, must not crash


# --- Inference (requires a trained checkpoint from spatial/train.py) ---

def test_inference_requires_existing_checkpoint():
    with pytest.raises(FileNotFoundError):
        SpatialInference(checkpoint_path=Path("models/spatial/does_not_exist.pt"))


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_inference_produces_contract_shape():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    face = align_from_bbox(frame.image, bbox, face_id=7)

    engine = SpatialInference()
    result = engine.score_face(face)

    assert set(result.keys()) == {"face_id", "spatial_score"}
    assert result["face_id"] == 7
    assert 0.0 <= result["spatial_score"] <= 1.0


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_inference_batch_matches_single_scoring():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    face = align_from_bbox(frame.image, bbox, face_id=1)

    engine = SpatialInference()
    single = engine.score_face(face)
    batch = engine.score_faces([face])[0]

    assert single == batch


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_inference_handles_empty_face_list():
    engine = SpatialInference()
    assert engine.score_faces([]) == []
