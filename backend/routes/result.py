"""GET /result/{id} — re-fetch a previously computed analysis result."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.services.storage import load_result

router = APIRouter()


@router.get("/result/{result_id}")
def get_result(result_id: str) -> dict:
    result = load_result(result_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No result found for id {result_id!r}")
    return result
