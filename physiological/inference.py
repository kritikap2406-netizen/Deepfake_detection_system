"""
TrueSight — physiological stream inference.

Unlike spatial/frequency/noise (which score one face crop at a time),
the physiological stream needs a *sequence* of frames for the same
tracked face_id — blinking and rPPG are both inherently temporal
signals. This is why the entry point here takes preprocessing.pipeline's
FrameFaces list (per-video, face_id-consistent) rather than a single
AlignedFace.

Output matches configs/pipeline.yaml's physiological_output contract:
    {"face_id": int, "blink_score": float, "rppg_score": float}

Note: this module is not in the plan's literal physiological/ file list
(blink.py, ear.py, rppg.py, temporal.py) but follows the same
inference.py convention established in spatial/ and frequency_noise/,
so every stream has one sanctioned entry point fusion/backend can call.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from physiological.blink import analyze_blinks
from physiological.ear import average_ear
from physiological.landmarks import FaceMeshLandmarker
from physiological.rppg import analyze_rppg
from preprocessing.pipeline import FrameFaces


def _crop_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    h_img, w_img = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        # Degenerate ROI (e.g. landmarks pushed the box outside the crop) —
        # fall back to the full face crop rather than crashing.
        return image
    return crop


class PhysiologicalInference:
    def __init__(self, landmarker: FaceMeshLandmarker | None = None):
        self.landmarker = landmarker or FaceMeshLandmarker.create()

    def score_video(self, frame_faces: list[FrameFaces], fps: float) -> list[dict]:
        """frame_faces: per-frame tracked faces from preprocessing.pipeline.preprocess_video.
        fps: the effective sample rate used when the frames were extracted
        (preprocess_video's sample_rate_fps), needed for blink-rate and
        rPPG frequency calculations.

        Returns one result dict per face_id seen anywhere in the video.
        """
        ear_by_face: dict[int, list[float]] = defaultdict(list)
        cheek_by_face: dict[int, list[np.ndarray]] = defaultdict(list)

        for frame in frame_faces:
            for aligned_face in frame.faces:
                mesh = self.landmarker.get_landmarks(aligned_face.image)
                if mesh is None:
                    continue

                ear = average_ear(mesh.left_eye, mesh.right_eye)
                ear_by_face[aligned_face.face_id].append(ear)

                cheek_crop = _crop_roi(aligned_face.image, mesh.cheek_roi)
                cheek_by_face[aligned_face.face_id].append(cheek_crop)

        all_face_ids = set(ear_by_face) | set(cheek_by_face)
        results = []
        for face_id in sorted(all_face_ids):
            blink_analysis = analyze_blinks(ear_by_face.get(face_id, []), fps=fps)
            rppg_analysis = analyze_rppg(cheek_by_face.get(face_id, []), fps=fps)

            results.append({
                "face_id": face_id,
                "blink_score": blink_analysis.blink_score,
                "rppg_score": rppg_analysis.rppg_score,
            })

        return results
