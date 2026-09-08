"""
TrueSight — FastAPI application.

Run with:
    uvicorn backend.main:app --reload

Endpoints:
    POST /analyze/image
    POST /analyze/video
    GET  /result/{id}
    GET  /health
    GET  /static/heatmaps/{filename}  (serves Grad-CAM++ overlays)
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.routes import analyze, health, result
from configs_loader import load_config

_config = load_config()
HEATMAP_DIR = Path(_config["paths"]["heatmap_output_dir"])
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="TrueSight",
    description="Multi-stream deepfake detection API — SIH 2026, Team CipherForge",
    version=_config["project"]["version"],
)

app.include_router(health.router)
app.include_router(analyze.router)
app.include_router(result.router)

app.mount("/static/heatmaps", StaticFiles(directory=str(HEATMAP_DIR)), name="heatmaps")
