"""
TrueSight — spatial stream inference.

Consumes AlignedFace objects from preprocessing.pipeline and produces the
exact contract specified in configs/pipeline.yaml:

    {"face_id": int, "spatial_score": float}   # 0.0 (real) - 1.0 (fake)

This is the module the fusion stage (Phase 10) and backend (Phase 14)
actually import. Nothing outside spatial/ should touch SpatialClassifier
or the raw checkpoint directly — this file is the sanctioned entry point.
"""

from __future__ import annotations

from pathlib import Path

import torch

from preprocessing.face_alignment import AlignedFace
from spatial.dataset import normalize_to_tensor
from spatial.model import SpatialClassifier, build_model

DEFAULT_CHECKPOINT = Path("models/spatial/efficientnet_b0.pt")


class SpatialInference:
    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"No spatial checkpoint at {checkpoint_path}. "
                f"Run `python -m spatial.train` first."
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        backbone = checkpoint["backbone"]

        self.model: SpatialClassifier = build_model(backbone=backbone, pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_face(self, face: AlignedFace) -> dict:
        """Score a single AlignedFace -> {"face_id": int, "spatial_score": float}"""
        tensor = normalize_to_tensor(face.image).unsqueeze(0).to(self.device)
        proba = self.model.predict_proba(tensor).item()
        return {"face_id": face.face_id, "spatial_score": round(proba, 4)}

    @torch.no_grad()
    def score_faces(self, faces: list[AlignedFace]) -> list[dict]:
        """Batched scoring for efficiency when multiple faces are present."""
        if not faces:
            return []

        tensors = torch.stack([normalize_to_tensor(f.image) for f in faces]).to(self.device)
        probas = self.model.predict_proba(tensors).cpu().tolist()

        return [
            {"face_id": face.face_id, "spatial_score": round(p, 4)}
            for face, p in zip(faces, probas)
        ]
