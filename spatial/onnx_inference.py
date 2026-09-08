"""
TrueSight — ONNX Runtime inference.

Drop-in alternative to spatial/inference.py's SpatialInference, using
ONNX Runtime instead of PyTorch for the forward pass. Same output
contract ({"face_id": int, "spatial_score": float}), so fusion and the
backend orchestrator can use either interchangeably.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from preprocessing.face_alignment import AlignedFace
from spatial.dataset import normalize_to_tensor
from spatial.export_onnx import DEFAULT_ONNX_PATH


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class SpatialONNXInference:
    def __init__(self, onnx_path: Path = DEFAULT_ONNX_PATH):
        if not Path(onnx_path).exists():
            raise FileNotFoundError(
                f"No ONNX model at {onnx_path}. Run `python -m spatial.export_onnx` first."
            )
        self.session = ort.InferenceSession(str(onnx_path))
        self.input_name = self.session.get_inputs()[0].name

    def score_face(self, face: AlignedFace) -> dict:
        tensor = normalize_to_tensor(face.image).unsqueeze(0).numpy()
        logit = self.session.run(None, {self.input_name: tensor})[0]
        proba = float(_sigmoid(logit)[0])
        return {"face_id": face.face_id, "spatial_score": round(proba, 4)}

    def score_faces(self, faces: list[AlignedFace]) -> list[dict]:
        if not faces:
            return []
        tensors = np.stack([normalize_to_tensor(f.image).numpy() for f in faces])
        logits = self.session.run(None, {self.input_name: tensors})[0]
        probas = _sigmoid(logits)
        return [
            {"face_id": face.face_id, "spatial_score": round(float(p), 4)}
            for face, p in zip(faces, probas)
        ]
