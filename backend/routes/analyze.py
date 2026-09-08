"""
POST /analyze/image and POST /analyze/video.

Accepts an uploaded file, saves it to a temp location, runs the full
orchestrated pipeline, persists the result, and returns it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.services.orchestrator import analyze_image as run_image_analysis
from backend.services.orchestrator import analyze_video as run_video_analysis
from backend.services.storage import save_result

router = APIRouter()

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB, generous for a hackathon demo


async def _save_upload(file: UploadFile, allowed_extensions: set[str]) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Allowed: {sorted(allowed_extensions)}",
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(contents)
    tmp.close()
    return Path(tmp.name)


@router.post("/analyze/image")
async def analyze_image_endpoint(file: UploadFile = File(...)) -> dict:
    tmp_path = await _save_upload(file, ALLOWED_IMAGE_EXTENSIONS)
    try:
        result = run_image_analysis(str(tmp_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    save_result(result["id"], result)
    return result


@router.post("/analyze/video")
async def analyze_video_endpoint(file: UploadFile = File(...)) -> dict:
    tmp_path = await _save_upload(file, ALLOWED_VIDEO_EXTENSIONS)
    try:
        result = run_video_analysis(str(tmp_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    save_result(result["id"], result)
    return result
