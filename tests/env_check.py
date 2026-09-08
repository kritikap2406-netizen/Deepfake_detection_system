"""
TrueSight — environment verification script.

Every team member should run this after setting up their virtualenv,
before writing or pulling any pipeline code. It confirms that the
core dependencies are importable and reports whether a GPU is
available for training/inference.

Usage:
    python -m tests.env_check
"""

import importlib
import sys

REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "cv2",            # opencv-python
    "mediapipe",
    "numpy",
    "fastapi",
    "sklearn",
    "xgboost",
]


def check_module(name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown version")
        return True, version
    except ImportError as exc:
        return False, str(exc)


def main() -> int:
    print("TrueSight environment check")
    print("=" * 40)

    all_ok = True
    for module_name in REQUIRED_MODULES:
        ok, info = check_module(module_name)
        status = "OK" if ok else "MISSING"
        print(f"[{status:7}] {module_name:15} {info}")
        all_ok = all_ok and ok

    print("-" * 40)

    # GPU check (does not fail the overall check — CPU-only dev is fine)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[GPU]     CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("[GPU]     CUDA not available — running on CPU")
    except ImportError:
        print("[GPU]     torch not installed, cannot check CUDA")

    print("=" * 40)
    if all_ok:
        print("All required modules importable. Environment ready.")
    else:
        print("One or more modules are missing. Run: pip install -r requirements.txt")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
