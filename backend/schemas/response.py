"""
TrueSight — API response schemas.

Shapes match the development plan's "Final API Response Example"
exactly: one top-level prediction/confidence, a list of per-face stream
scores + heatmap + suspicious regions, and an overall explanation list.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FaceResult(BaseModel):
    face_id: int
    spatial_score: float
    frequency_score: float
    noise_score: float
    blink_score: float
    rppg_score: float
    temporal_score: float
    heatmap: str
    suspicious_regions: list[str]


class AnalysisResponse(BaseModel):
    id: str
    media_type: str  # "image" | "video"
    prediction: str  # "REAL" | "FAKE"
    confidence: float
    faces: list[FaceResult]
    explanation: list[str]
    metadata_flag: float = Field(description="File-level metadata evidence score, supporting only — see metadata/analyzer.py")


class NoFaceDetectedResponse(BaseModel):
    id: str
    media_type: str
    prediction: str = "UNKNOWN"
    message: str = "No face could be detected in the provided media."


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict[str, bool]
