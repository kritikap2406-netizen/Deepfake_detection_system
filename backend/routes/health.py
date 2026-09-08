"""GET /health — reports whether required model checkpoints are present."""

from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.response import HealthResponse
from backend.services.orchestrator import engines_available

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    models = engines_available()
    status = "ok" if all(models.values()) else "degraded"
    return HealthResponse(status=status, models_loaded=models)
