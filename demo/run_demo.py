"""
TrueSight — final demo workflow.

Runs the complete pipeline against a sample file and prints the result,
for a live SIH demo or a quick sanity check that everything is wired up
correctly. Uses the real orchestrator built in Phase 14 — no shortcuts.

Usage:
    python -m demo.run_demo path/to/image.jpg
    python -m demo.run_demo path/to/video.mp4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from backend.services.orchestrator import analyze_image, analyze_video, engines_available

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m demo.run_demo <path-to-image-or-video>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    print("Checking model availability...")
    availability = engines_available()
    for name, ok in availability.items():
        print(f"  [{'OK' if ok else 'MISSING'}] {name}")
    if not all(availability.values()):
        print("\nSome models are missing. Run the relevant training scripts first "
              "(see README.md 'Development order').")
        sys.exit(1)

    ext = file_path.suffix.lower()
    print(f"\nAnalyzing {file_path.name}...")

    if ext in IMAGE_EXTENSIONS:
        result = analyze_image(str(file_path))
    elif ext in VIDEO_EXTENSIONS:
        result = analyze_video(str(file_path))
    else:
        print(f"Unsupported file extension: {ext}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
