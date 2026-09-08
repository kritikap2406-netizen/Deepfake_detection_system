"""
TrueSight — heatmap generation and overlay.

Converts a raw [0,1] Grad-CAM/Grad-CAM++ activation map into a
human-readable colored overlay on the original face crop, and saves it
to disk for the API/frontend to serve.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from configs_loader import load_config

_config = load_config()
HEATMAP_OUTPUT_DIR = Path(_config["paths"]["heatmap_output_dir"])


def cam_to_color_heatmap(cam: torch.Tensor | np.ndarray) -> np.ndarray:
    """[0,1] CAM -> BGR uint8 color-mapped heatmap (JET: blue=low, red=high activation)."""
    if isinstance(cam, torch.Tensor):
        cam = cam.detach().cpu().numpy()

    cam_uint8 = (cam * 255).astype(np.uint8)
    return cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)


def overlay_heatmap(face_bgr: np.ndarray, cam: torch.Tensor | np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend the color-mapped heatmap onto the original face crop.
    alpha: heatmap opacity (0 = invisible, 1 = heatmap only, no face visible).
    """
    color_heatmap = cam_to_color_heatmap(cam)
    if color_heatmap.shape[:2] != face_bgr.shape[:2]:
        color_heatmap = cv2.resize(color_heatmap, (face_bgr.shape[1], face_bgr.shape[0]))

    return cv2.addWeighted(face_bgr, 1 - alpha, color_heatmap, alpha, 0)


def save_heatmap(
    face_bgr: np.ndarray,
    cam: torch.Tensor | np.ndarray,
    filename: str,
    output_dir: Path = HEATMAP_OUTPUT_DIR,
    alpha: float = 0.45,
) -> Path:
    """Generates the overlay and writes it to output_dir/filename. Returns the full path."""
    overlay = overlay_heatmap(face_bgr, cam, alpha=alpha)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    ok = cv2.imwrite(str(output_path), overlay)
    if not ok:
        raise IOError(f"Failed to write heatmap to {output_path}")

    return output_path
