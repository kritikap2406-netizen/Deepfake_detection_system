"""
TrueSight — frequency stream inference.

Consumes AlignedFace objects from preprocessing.pipeline and produces the
exact contract specified in configs/pipeline.yaml:

    {"face_id": int, "frequency_score": float}   # 0.0 (real) - 1.0 (fake)
"""

from __future__ import annotations

from pathlib import Path

import torch

from preprocessing.face_alignment import AlignedFace
from frequency_noise.dataset import prepare_frequency_input, prepare_noise_input
from frequency_noise.frequency_model import FrequencyClassifier
from frequency_noise.frequency_model import build_model as build_frequency_model
from frequency_noise.noise_model import NoiseClassifier
from frequency_noise.noise_model import build_model as build_noise_model

DEFAULT_CHECKPOINT = Path("models/frequency/frequency_model.pt")
DEFAULT_NOISE_CHECKPOINT = Path("models/noise/noise_model.pt")


class FrequencyInference:
    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"No frequency checkpoint at {checkpoint_path}. "
                f"Run `python -m frequency_noise.train` first."
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model: FrequencyClassifier = build_frequency_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_face(self, face: AlignedFace) -> dict:
        tensor = prepare_frequency_input(face.image).unsqueeze(0).to(self.device)
        proba = self.model.predict_proba(tensor).item()
        return {"face_id": face.face_id, "frequency_score": round(proba, 4)}

    @torch.no_grad()
    def score_faces(self, faces: list[AlignedFace]) -> list[dict]:
        if not faces:
            return []

        tensors = torch.stack([prepare_frequency_input(f.image) for f in faces]).to(self.device)
        probas = self.model.predict_proba(tensors).cpu().tolist()

        return [
            {"face_id": face.face_id, "frequency_score": round(p, 4)}
            for face, p in zip(faces, probas)
        ]


class NoiseInference:
    def __init__(self, checkpoint_path: Path = DEFAULT_NOISE_CHECKPOINT, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"No noise checkpoint at {checkpoint_path}. "
                f"Run `python -m frequency_noise.train_noise` first."
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model: NoiseClassifier = build_noise_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_face(self, face: AlignedFace) -> dict:
        tensor = prepare_noise_input(face.image).unsqueeze(0).to(self.device)
        proba = self.model.predict_proba(tensor).item()
        return {"face_id": face.face_id, "noise_score": round(proba, 4)}

    @torch.no_grad()
    def score_faces(self, faces: list[AlignedFace]) -> list[dict]:
        if not faces:
            return []

        tensors = torch.stack([prepare_noise_input(f.image) for f in faces]).to(self.device)
        probas = self.model.predict_proba(tensors).cpu().tolist()

        return [
            {"face_id": face.face_id, "noise_score": round(p, 4)}
            for face, p in zip(faces, probas)
        ]
