"""
TrueSight — test-time augmentation (TTA).

Per the plan: original input, horizontal flip, small crop, compression
variation — run multiple predictions, aggregate evidence. Applied here
to the spatial stream specifically, since TTA classically targets image
classifiers and spatial is the stream operating on raw face pixels.

Why these four variants: a genuine manipulation artifact should survive
small, meaning-preserving transforms of the same face (it's a property
of the face itself, not of the exact crop/encoding), whereas noise in a
single prediction might not. Averaging across variants should be more
stable than trusting any one crop, and the spread across variants is
itself a usable uncertainty signal — a face where all four variants
agree closely is a more confident evidence source than one where they
disagree wildly, even if the mean score is the same in both cases.

Per the plan's explicit sequencing: "Use TTA after the basic detector
works; do not introduce it before establishing a clean baseline" — this
builds on Phase 4's already-trained, already-tested spatial model.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from preprocessing.face_alignment import AlignedFace
from spatial.dataset import normalize_to_tensor
from spatial.inference import SpatialInference, DEFAULT_CHECKPOINT


def horizontal_flip(face_bgr: np.ndarray) -> np.ndarray:
    return cv2.flip(face_bgr, 1)


def small_crop(face_bgr: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
    """Center-crop to `crop_ratio` of the original size, then resize back
    — a small viewpoint/framing perturbation rather than a different
    transform family (distinct from resize_degrade in evaluation/degradations.py,
    which simulates low resolution; this simulates slightly different framing)."""
    h, w = face_bgr.shape[:2]
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    cropped = face_bgr[y0:y0 + crop_h, x0:x0 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def compression_variation(face_bgr: np.ndarray, quality: int = 70) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", face_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return face_bgr
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def generate_tta_variants(face_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Returns the four variants named in the plan, keyed by name so
    per-variant scores stay interpretable rather than an anonymous list."""
    return {
        "original": face_bgr,
        "horizontal_flip": horizontal_flip(face_bgr),
        "small_crop": small_crop(face_bgr),
        "compression_variation": compression_variation(face_bgr),
    }


class TTASpatialInference:
    """Wraps SpatialInference to add test-time augmentation. Output
    remains a superset of the plain face_id/spatial_score contract
    (fusion/features.py only reads spatial_score, so this is a drop-in
    richer replacement, not a breaking change) plus per-variant detail
    and a spread measure for transparency.
    """

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT):
        self.base_engine = SpatialInference(checkpoint_path=checkpoint_path)

    def score_face_tta(self, face: AlignedFace) -> dict:
        variants = generate_tta_variants(face.image)

        tensors = torch.stack([normalize_to_tensor(img) for img in variants.values()])
        with torch.no_grad():
            probs = self.base_engine.model.predict_proba(tensors.to(self.base_engine.device)).cpu().tolist()

        variant_scores = {name: round(p, 4) for name, p in zip(variants.keys(), probs)}
        aggregated = float(np.mean(probs))
        spread = float(np.std(probs))

        return {
            "face_id": face.face_id,
            "spatial_score": round(aggregated, 4),
            "tta_variants": variant_scores,
            "tta_std": round(spread, 4),
        }

    def score_faces_tta(self, faces: list[AlignedFace]) -> list[dict]:
        return [self.score_face_tta(face) for face in faces]
