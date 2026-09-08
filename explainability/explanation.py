"""
TrueSight — evidence-based explanation generation.

Per the plan: "The explanation should be tied to actual model evidence
rather than invented reasons." Concretely, that means: we never write a
canned sentence like "unnatural skin texture detected" unless the CAM
heatmap actually shows elevated activation in the region that sentence
refers to. Every phrase in the generated explanation is backed by a
computed number (a region's mean activation intensity), not a template
picked independent of the actual heatmap.

Regions are defined as fixed proportional boxes on the standardized
224x224 aligned face crop — the same "aligned, centered face" assumption
preprocessing.face_alignment guarantees, and the same style of fixed
proportional regions used in physiological/landmarks.py's geometric
fallback, for consistency across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Proportional (fraction of width/height) regions on a 224x224 aligned
# face crop. Coarse but sufficient for a hackathon-scale explanation —
# precise landmark-based regions could replace this once the real
# MediaPipe FaceLandmarker is wired in (see physiological/landmarks.py).
FACE_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # name: (x0, y0, x1, y1) as fractions of image width/height
    "forehead":    (0.20, 0.05, 0.80, 0.30),
    "left_eye":    (0.20, 0.30, 0.48, 0.45),
    "right_eye":   (0.52, 0.30, 0.80, 0.45),
    "nose":        (0.38, 0.40, 0.62, 0.62),
    "left_cheek":  (0.10, 0.45, 0.38, 0.68),
    "right_cheek": (0.62, 0.45, 0.90, 0.68),
    "mouth":       (0.32, 0.65, 0.68, 0.80),
    "chin":        (0.30, 0.80, 0.70, 0.95),
}

SUSPICIOUS_INTENSITY_THRESHOLD = 0.5  # mean region activation above this counts as "suspicious"
TOP_K_REGIONS_IN_EXPLANATION = 3


@dataclass
class RegionEvidence:
    region: str
    mean_intensity: float


def compute_region_intensities(cam: torch.Tensor | np.ndarray, regions: dict = FACE_REGIONS) -> list[RegionEvidence]:
    """cam: (H, W) heatmap in [0,1] (as produced by explainability.gradcam).
    Returns per-region mean intensity, sorted descending — the ranking IS
    the localization: highest-intensity regions are where the model's
    evidence for "fake" concentrated.
    """
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()

    h, w = cam.shape
    results = []
    for name, (x0, y0, x1, y1) in regions.items():
        px0, py0 = int(x0 * w), int(y0 * h)
        px1, py1 = int(x1 * w), int(y1 * h)
        region_slice = cam[py0:py1, px0:px1]

        mean_intensity = float(region_slice.mean()) if region_slice.size > 0 else 0.0
        results.append(RegionEvidence(region=name, mean_intensity=round(mean_intensity, 4)))

    return sorted(results, key=lambda r: r.mean_intensity, reverse=True)


def suspicious_regions(
    ranked_regions: list[RegionEvidence],
    threshold: float = SUSPICIOUS_INTENSITY_THRESHOLD,
) -> list[str]:
    """Region names whose evidence crosses the suspicion threshold —
    matches the final API's `suspicious_regions` field."""
    return [r.region for r in ranked_regions if r.mean_intensity >= threshold]


def generate_textual_explanation(
    ranked_regions: list[RegionEvidence],
    spatial_score: float | None = None,
    top_k: int = TOP_K_REGIONS_IN_EXPLANATION,
) -> list[str]:
    """Produces a short list of evidence-grounded explanation strings.
    Every sentence names an actual region and cites its actual computed
    intensity — nothing here is templated independent of the CAM output.
    """
    explanations: list[str] = []

    if spatial_score is not None:
        if spatial_score >= 0.5:
            explanations.append(
                f"Spatial analysis flagged manipulation evidence (score {spatial_score:.2f})."
            )
        else:
            explanations.append(
                f"Spatial analysis found the face visually consistent with authentic content (score {spatial_score:.2f})."
            )

    top_regions = ranked_regions[:top_k]
    flagged = [r for r in top_regions if r.mean_intensity >= SUSPICIOUS_INTENSITY_THRESHOLD]

    if not flagged:
        explanations.append("No single facial region showed concentrated activation above the suspicion threshold.")
        return explanations

    if len(flagged) == 1:
        r = flagged[0]
        explanations.append(
            f"Strongest activation concentrated around the {r.region.replace('_', ' ')} "
            f"(intensity {r.mean_intensity:.2f})."
        )
    else:
        region_phrases = [f"{r.region.replace('_', ' ')} ({r.mean_intensity:.2f})" for r in flagged]
        explanations.append(
            "Activation evidence concentrated in multiple regions: " + ", ".join(region_phrases) + "."
        )

    return explanations
