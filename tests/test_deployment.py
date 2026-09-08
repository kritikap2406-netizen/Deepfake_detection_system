"""
Unit tests for Phase 21 — Deployment.

Run with:
    pytest tests/test_deployment.py -v

Scope note: Docker build/run cannot be tested in this environment (no
Docker daemon available in the dev sandbox — see Dockerfile.backend /
Dockerfile.frontend / docker-compose.yml comments). What IS genuinely
tested here: ONNX export correctness and numerical equivalence with the
original PyTorch model, plus docker-compose.yml's YAML validity.
"""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
import yaml

from spatial.export_onnx import export_spatial_to_onnx, DEFAULT_ONNX_PATH, DEFAULT_PYTORCH_CHECKPOINT
from spatial.onnx_inference import SpatialONNXInference
from spatial.inference import SpatialInference
from spatial.model import build_model
from preprocessing.face_alignment import align_from_bbox
from preprocessing.face_detector import BoundingBox
from preprocessing.frame_extractor import load_single_image

ASSETS = Path("/home/claude/test_assets")


# --- ONNX export ---

@pytest.mark.skipif(not DEFAULT_PYTORCH_CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_export_produces_valid_onnx_file(tmp_path):
    output_path = tmp_path / "test_export.onnx"
    result_path = export_spatial_to_onnx(output_path=output_path)
    assert result_path.exists()

    model = onnx.load(str(result_path))
    onnx.checker.check_model(model)  # raises if structurally invalid


@pytest.mark.skipif(not DEFAULT_PYTORCH_CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_export_raises_on_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_spatial_to_onnx(
            checkpoint_path=Path("models/spatial/does_not_exist.pt"),
            output_path=tmp_path / "out.onnx",
        )


@pytest.mark.skipif(not DEFAULT_PYTORCH_CHECKPOINT.exists(), reason="Run `python -m spatial.train` first")
def test_exported_onnx_has_expected_io_names(tmp_path):
    output_path = tmp_path / "test_export.onnx"
    export_spatial_to_onnx(output_path=output_path)

    model = onnx.load(str(output_path))
    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]
    assert "face_crop" in input_names
    assert "fake_logit" in output_names


# --- ONNX vs PyTorch numerical equivalence (the critical check) ---

@pytest.mark.skipif(not DEFAULT_ONNX_PATH.exists(), reason="Run `python -m spatial.export_onnx` first")
def test_onnx_inference_requires_existing_file():
    with pytest.raises(FileNotFoundError):
        SpatialONNXInference(onnx_path=Path("models/spatial/does_not_exist.onnx"))


@pytest.mark.skipif(
    not (DEFAULT_ONNX_PATH.exists() and DEFAULT_PYTORCH_CHECKPOINT.exists()),
    reason="Run `python -m spatial.train` then `python -m spatial.export_onnx` first",
)
def test_onnx_matches_pytorch_on_real_face_crop():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    face = align_from_bbox(frame.image, bbox, face_id=1)

    pytorch_result = SpatialInference().score_face(face)
    onnx_result = SpatialONNXInference().score_face(face)

    assert abs(pytorch_result["spatial_score"] - onnx_result["spatial_score"]) < 0.001


@pytest.mark.skipif(
    not (DEFAULT_ONNX_PATH.exists() and DEFAULT_PYTORCH_CHECKPOINT.exists()),
    reason="Run `python -m spatial.train` then `python -m spatial.export_onnx` first",
)
def test_onnx_matches_pytorch_across_diverse_random_inputs():
    """The core equivalence check, stress-tested across many inputs
    rather than trusting a single lucky match."""
    checkpoint = torch.load(DEFAULT_PYTORCH_CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(backbone=checkpoint["backbone"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    import onnxruntime as ort
    session = ort.InferenceSession(str(DEFAULT_ONNX_PATH))
    input_name = session.get_inputs()[0].name

    torch.manual_seed(1)
    max_diff = 0.0
    for _ in range(10):
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            pytorch_proba = torch.sigmoid(model(x)).item()

        onnx_logit = session.run(None, {input_name: x.numpy()})[0]
        onnx_proba = float(1 / (1 + np.exp(-onnx_logit[0])))

        max_diff = max(max_diff, abs(pytorch_proba - onnx_proba))

    assert max_diff < 1e-4


@pytest.mark.skipif(
    not (DEFAULT_ONNX_PATH.exists() and DEFAULT_PYTORCH_CHECKPOINT.exists()),
    reason="Run `python -m spatial.train` then `python -m spatial.export_onnx` first",
)
def test_onnx_batch_scoring_matches_single_scoring():
    frame = load_single_image(str(ASSETS / "synthetic_face.jpg"))
    bbox = BoundingBox(x=230, y=120, width=180, height=240, confidence=1.0)
    face = align_from_bbox(frame.image, bbox, face_id=1)

    engine = SpatialONNXInference()
    single = engine.score_face(face)
    batch = engine.score_faces([face])[0]
    assert single == batch


@pytest.mark.skipif(not DEFAULT_ONNX_PATH.exists(), reason="Run `python -m spatial.export_onnx` first")
def test_onnx_handles_empty_face_list():
    engine = SpatialONNXInference()
    assert engine.score_faces([]) == []


# --- Deployment config validity ---

def test_docker_compose_yaml_is_valid():
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists()
    with open(compose_path) as f:
        data = yaml.safe_load(f)
    assert "services" in data
    assert "backend" in data["services"]
    assert "frontend" in data["services"]


def test_docker_compose_backend_exposes_expected_port():
    with open("docker-compose.yml") as f:
        data = yaml.safe_load(f)
    assert "8000:8000" in data["services"]["backend"]["ports"]


def test_dockerfiles_exist():
    assert Path("Dockerfile.backend").exists()
    assert Path("Dockerfile.frontend").exists()
    assert Path("nginx.conf").exists()
