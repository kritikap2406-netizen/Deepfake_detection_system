"""
TrueSight — ONNX export.

Converts the trained spatial classifier (PyTorch) to ONNX format for
optimized inference via ONNX Runtime, per the plan's deployment tech
stack (ONNX / ONNX Runtime for "efficient inference").

Only the spatial model is exported here — it's the heaviest of the
learned models (EfficientNet-B0, ~5.3M params) and the one where
inference-speed optimization actually matters. The frequency/noise
models are already lightweight from-scratch CNNs (~98K params, see
Phase 5-6), and fusion is a small XGBoost/sklearn model — neither
benefits meaningfully from ONNX conversion at this scale.
"""

from __future__ import annotations

from pathlib import Path

import torch

from spatial.model import SpatialClassifier, build_model

DEFAULT_PYTORCH_CHECKPOINT = Path("models/spatial/efficientnet_b0.pt")
DEFAULT_ONNX_PATH = Path("models/spatial/efficientnet_b0.onnx")


def export_spatial_to_onnx(
    checkpoint_path: Path = DEFAULT_PYTORCH_CHECKPOINT,
    output_path: Path = DEFAULT_ONNX_PATH,
    opset_version: int = 18,  # this torch/onnxscript version's minimum supported opset;
                               # requesting 17 triggers a noisy (though non-fatal) failed
                               # downgrade attempt before falling back to 18 anyway.
) -> Path:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}. Run `python -m spatial.train` first.")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model: SpatialClassifier = build_model(backbone=checkpoint["backbone"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["face_crop"],
        output_names=["fake_logit"],
        dynamic_axes={"face_crop": {0: "batch_size"}, "fake_logit": {0: "batch_size"}},
        opset_version=opset_version,
    )

    return output_path


if __name__ == "__main__":
    path = export_spatial_to_onnx()
    print(f"Exported ONNX model to {path}")
