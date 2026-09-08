"""
TrueSight — face tracking (video).

For a single image there's nothing to track. For video, we need to know
that "face_id 1 in frame 3" is the same physical person as "face_id 1 in
frame 4" — this is what lets the physiological stream compute blink
sequences and rPPG over time, and what lets temporal.py aggregate
frame-level scores into one video-level score per person.

We use simple IoU (intersection-over-union) matching between consecutive
frames rather than a learned tracker. This is intentional and matches the
plan's guidance: establish reliable frame-level tracking first, don't
reach for a heavier solution before the simple one is proven insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from preprocessing.face_detector import BoundingBox


def iou(a: BoundingBox, b: BoundingBox) -> float:
    ax0, ay0, ax1, ay1 = a.x, a.y, a.x + a.width, a.y + a.height
    bx0, by0, bx1, by1 = b.x, b.y, b.x + b.width, b.y + b.height

    inter_x0, inter_y0 = max(ax0, bx0), max(ay0, by0)
    inter_x1, inter_y1 = min(ax1, bx1), min(ay1, by1)

    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    area_a = a.width * a.height
    area_b = b.width * b.height
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class TrackedFace:
    face_id: int
    bbox: BoundingBox
    frames_seen: int = 1
    frames_missing: int = 0


class FaceTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missing_frames: int = 5):
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames
        self._tracks: list[TrackedFace] = []
        self._next_id = 1

    def update(self, detections: list[BoundingBox]) -> list[tuple[int, BoundingBox]]:
        """Match new detections to existing tracks (or create new ones).

        Returns list of (face_id, bbox) for this frame, in detection order.
        """
        unmatched_detections = list(range(len(detections)))
        results: list[tuple[int, BoundingBox] | None] = [None] * len(detections)

        # Greedy IoU matching: for each existing track, find its best
        # unmatched detection above threshold.
        for track in self._tracks:
            best_iou = 0.0
            best_det_idx = None
            for det_idx in unmatched_detections:
                score = iou(track.bbox, detections[det_idx])
                if score > best_iou:
                    best_iou = score
                    best_det_idx = det_idx

            if best_det_idx is not None and best_iou >= self.iou_threshold:
                track.bbox = detections[best_det_idx]
                track.frames_seen += 1
                track.frames_missing = 0
                results[best_det_idx] = (track.face_id, track.bbox)
                unmatched_detections.remove(best_det_idx)
            else:
                track.frames_missing += 1

        # Any leftover detections become new tracks
        for det_idx in unmatched_detections:
            new_track = TrackedFace(face_id=self._next_id, bbox=detections[det_idx])
            self._next_id += 1
            self._tracks.append(new_track)
            results[det_idx] = (new_track.face_id, new_track.bbox)

        # Drop tracks that have been missing too long
        self._tracks = [t for t in self._tracks if t.frames_missing <= self.max_missing_frames]

        return [r for r in results if r is not None]
