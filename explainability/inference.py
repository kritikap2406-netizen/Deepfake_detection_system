"""
TrueSight — explainability inference.

The sanctioned entry point tying together Phase 12's three pieces:
Grad-CAM++ computation, heatmap overlay/saving, and evidence-based
textual explanation. This is what the backend (Phase 14) calls once per
face to populate the final API response's "heatmap", "suspicious_regions",
and "explanation" fields.

Grad-CAM++ is used as the default (not plain Grad-CAM) per
configs/pipeline.yaml's explainability.method setting, since it
generally produces tighter, more precisely localized activation — better
suited to per-region suspicious-area reporting.
"""

from __future__ import annotations

from pathlib import Path

import torch

from configs_loader import load_config
from explainability.explanation import (
    compute_region_intensities, suspicious_regions, generate_textual_explanation,
)
from explainability.gradcam import GradCAMExtractor
from explainability.heatmap import save_heatmap
from preprocessing.face_alignment import AlignedFace
from spatial.dataset import normalize_to_tensor
from spatial.model import SpatialClassifier, build_model

_config = load_config()
DEFAULT_CHECKPOINT = Path(_config["paths"]["spatial_weights"])


class ExplainabilityInference:
    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"No spatial checkpoint at {checkpoint_path}. Run `python -m spatial.train` first."
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model: SpatialClassifier = build_model(backbone=checkpoint["backbone"], pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.extractor = GradCAMExtractor(self.model)

    def explain_face(self, face: AlignedFace, spatial_score: float | None = None) -> dict:
        """Returns everything the final API response needs for one face:

            {
                "heatmap_path": str,
                "suspicious_regions": [str, ...],
                "explanation": [str, ...],
            }
        """
        tensor = normalize_to_tensor(face.image).unsqueeze(0)
        cam = self.extractor.compute_gradcam_plusplus(tensor)

        ranked = compute_region_intensities(cam)
        flagged = suspicious_regions(ranked)
        explanation = generate_textual_explanation(ranked, spatial_score=spatial_score)

        heatmap_path = save_heatmap(face.image, cam, filename=f"face_{face.face_id}_heatmap.png")

        return {
            "heatmap_path": str(heatmap_path),
            "suspicious_regions": flagged,
            "explanation": explanation,
        }
