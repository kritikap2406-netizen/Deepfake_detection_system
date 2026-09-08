# TrueSight

**Multi-stream deepfake detection system** — Smart India Hackathon 2026, Theme: Cyber Security
Team CipherForge

TrueSight detects whether a face in an image or video has been digitally manipulated. Instead of relying on a single model, it runs four independent detection streams on every face, fuses their evidence with a calibrated meta-classifier, and explains *why* it reached its verdict with Grad-CAM++ heatmaps and region-level evidence — rather than just outputting "Fake" or "Real."

```
IMAGE / VIDEO
      |
      v
FACE DETECTION & ALIGNMENT
      |
      +--------------+---------------+------------------+
      v              v               v                  v
  SPATIAL       FREQUENCY         NOISE          PHYSIOLOGICAL
 (EfficientNet)  (FFT/DCT)     (SRM residual)    (blink + rPPG)
      |              |               |            + TEMPORAL (video)
      +--------------+---------------+------------------+
                          |
                          v
                  SCORE FUSION (XGBoost)
                          |
                          v
              CONFIDENCE CALIBRATION (temperature scaling)
                          |
                          v
              EXPLAINABILITY (Grad-CAM++) + METADATA CHECK
                          |
                          v
              FINAL RESULT: REAL / FAKE + confidence + evidence
```

---

## Table of contents

- [What this actually is](#what-this-actually-is)
- [Quick start](#quick-start)
- [Full setup, step by step](#full-setup-step-by-step)
- [Running the system](#running-the-system)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [How the pipeline works](#how-the-pipeline-works)
- [Testing](#testing)
- [Docker deployment](#docker-deployment)
- [Known limitations](#known-limitations)
- [Development order](#development-order)
- [Team](#team)

---

## What this actually is

This is a **working, end-to-end prototype**, not a mockup. Every module listed below is real, tested code — not a stub that returns a hardcoded value. Concretely:

- A real EfficientNet-B0 spatial classifier, trained with focal loss, class balancing, and JPEG/blur/noise/resize augmentation.
- Real FFT/DCT frequency-domain analysis and real SRM (Spatial Rich Model) noise-residual filters, each feeding a lightweight from-scratch CNN.
- Real blink detection (Eye Aspect Ratio) and real rPPG (remote photoplethysmography) signal extraction with Butterworth bandpass filtering — verified to correctly distinguish a genuine periodic pulse signal from noise.
- A real XGBoost / Logistic Regression / MLP fusion comparison, with the winner selected automatically.
- Real temperature-scaling calibration, verified to cut Negative Log-Likelihood and Expected Calibration Error on a synthetically miscalibrated model.
- Real Grad-CAM++ implemented via forward/backward hooks on the actual EfficientNet architecture, producing real heatmap images.
- A real FastAPI backend and a real React + Vite frontend, tested against each other over actual HTTP requests, not mocks.
- A real ONNX export, numerically verified to match the PyTorch model's output exactly.

At the same time, this repository is honest about what it can't yet claim. The dataset shipped in `datasets/` is a tiny **synthetic placeholder** (48 non-photographic files), used purely to prove every stage of the pipeline runs correctly end-to-end. **No accuracy number in this repository should be read as a real-world performance claim** — see [Known limitations](#known-limitations) for the full, specific list of what is and isn't proven yet, and what closes each gap.

---

## Quick start

```bash
git clone <this-repo>
cd TrueSight
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train every model on the bundled placeholder dataset (fast, a few minutes total)
python -m preprocessing.build_metadata
python -m preprocessing.split_dataset
python -m spatial.train
python -m frequency_noise.train
python -m frequency_noise.train_noise
python -m fusion.build_dataset
python -m fusion.meta_classifier
python -m fusion.train_calibration

# Run the demo
python -m demo.run_demo path/to/your/image.jpg
```

Or start the full backend + frontend and use it in a browser — see [Running the system](#running-the-system).

---

## Full setup, step by step

### 1. Prerequisites

| Requirement | Version | Why |
|---|---|---|
| Python | 3.11+ | Core runtime |
| Node.js | 18+ | Frontend build |
| pip | latest | Package installation |
| CUDA GPU | optional | Faster training; everything also runs on CPU |
| Docker | optional | Containerized deployment |

### 2. Clone and create a virtual environment

```bash
git clone <this-repo>
cd TrueSight
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

Installs PyTorch, torchvision, timm (EfficientNet), MediaPipe, OpenCV, XGBoost, scikit-learn, FastAPI, ONNX/ONNX Runtime, and everything else the pipeline needs.

> **MediaPipe note:** face detection and landmark extraction default to a real MediaPipe model, but that model file (`.task` format) must be downloaded separately (see step 6). Without it, the pipeline automatically falls back to an OpenCV Haar cascade for face detection and a fixed-geometry estimate for landmarks — this keeps every command below runnable offline, but detection accuracy and blink/rPPG signal quality will be materially worse until you fetch the real model. This fallback is intentional and logged clearly at runtime.

### 4. Verify your environment

```bash
python -m tests.env_check
```

Confirms PyTorch, CUDA (if available), OpenCV, MediaPipe, NumPy, FastAPI, scikit-learn, and XGBoost are all importable, and reports whether a GPU was detected.

### 5. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 6. (Recommended) Fetch the real MediaPipe models

```bash
mkdir -p models/mediapipe
curl -L -o models/mediapipe/blaze_face_short_range.tflite \
  https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
curl -L -o models/mediapipe/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Update `configs/pipeline.yaml` with the resulting paths if placed elsewhere. Without this step the system still runs (see the fallback note above) but face detection and the physiological stream will not be accurate.

### 7. Build the dataset and train every model

The repo ships with a small **synthetic placeholder dataset** under `datasets/` so the full pipeline can be exercised without real deepfake corpora on day one. Replace it with real data (FaceForensics++, DFDC, Celeb-DF v2, DeeperForensics-1.0) using the same folder structure for anything beyond pipeline validation.

```bash
# 1. Build dataset metadata (file_path, dataset, label, manipulation_type, video_id, person_id)
python -m preprocessing.build_metadata

# 2. Leakage-safe train/val/test split (splits by identity, never by frame)
python -m preprocessing.split_dataset

# 3. Train the spatial stream (EfficientNet-B0)
python -m spatial.train --epochs 3

# 4. Train the frequency stream (FFT/DCT-based lightweight CNN)
python -m frequency_noise.train --epochs 3

# 5. Train the noise stream (SRM-residual-based lightweight CNN)
python -m frequency_noise.train_noise --epochs 3

# 6. Build the fusion training dataset (runs every trained stream over the split)
python -m fusion.build_dataset

# 7. Train and compare XGBoost / Logistic Regression / small MLP, keep the best
python -m fusion.meta_classifier

# 8. Fit temperature-scaling calibration on the fusion model's validation predictions
python -m fusion.train_calibration
```

At this point every model artifact needed by the backend exists under `models/`.

### 8. (Optional) Export the spatial model to ONNX for faster inference

```bash
python -m spatial.export_onnx
```

---

## Running the system

### Option A — backend + frontend (recommended)

**Terminal 1 — backend:**

```bash
source venv/bin/activate
uvicorn backend.main:app --reload
```

API live at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

**Terminal 2 — frontend:**

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`. The Vite dev server proxies `/analyze`, `/result`, `/health`, and `/static` to the backend (see `frontend/vite.config.js`), so no CORS configuration is needed.

Upload an image or video to see: the REAL/FAKE verdict, calibrated confidence, an evidence-based explanation, and — per detected face — a Grad-CAM++ heatmap, all six stream scores, and flagged suspicious regions.

### Option B — one-shot CLI demo

```bash
python -m demo.run_demo path/to/image.jpg
python -m demo.run_demo path/to/video.mp4
```

Runs the complete pipeline against one file and prints the full JSON result.

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/health` | Reports whether every required model checkpoint is present |
| `POST` | `/analyze/image` | Upload a JPG/PNG, get back the full analysis |
| `POST` | `/analyze/video` | Upload an MP4/AVI/MOV/MKV, get back the full analysis |
| `GET`  | `/result/{id}` | Re-fetch a previously computed result by its id |
| `GET`  | `/static/heatmaps/{filename}` | Serves generated Grad-CAM++ heatmap images |

**Example response** (`POST /analyze/image`):

```json
{
  "id": "39c0e121-e8f2-4843-b996-7a633c15f142",
  "media_type": "image",
  "prediction": "FAKE",
  "confidence": 0.9354,
  "faces": [
    {
      "face_id": 1,
      "spatial_score": 0.91,
      "frequency_score": 0.82,
      "noise_score": 0.88,
      "blink_score": 0.5,
      "rppg_score": 0.5,
      "temporal_score": 0.5,
      "heatmap": "face_1_heatmap.png",
      "suspicious_regions": ["left_eye", "mouth"]
    }
  ],
  "explanation": [
    "Overall verdict: FAKE (1 face(s) analyzed).",
    "Face 1: activation concentrated in left eye, mouth."
  ],
  "metadata_flag": 0.15
}
```

`blink_score`, `rppg_score`, and `temporal_score` stay at the neutral default `0.5` for image input, since those three streams are inherently video-only (they need a sequence of frames for the same tracked face). For video input, all seven fields carry genuine computed values.

**Multi-face aggregation policy:** if any detected face is predicted FAKE, the overall verdict is FAKE, with confidence equal to the highest calibrated confidence among the FAKE-predicted faces. If every face is predicted REAL, overall confidence is `1 - (highest fake-probability seen across faces)`. This is a documented design decision (the plan doesn't specify one) — see `backend/services/orchestrator.py`.

---

## Project structure

```
TrueSight/
|-- preprocessing/     Face detection, alignment, tracking, dataset engine
|-- spatial/            EfficientNet-B0 classifier, training, TTA, ONNX export
|-- frequency_noise/    FFT/DCT frequency stream + SRM noise stream
|-- physiological/      Blink (EAR), rPPG, temporal aggregation
|-- fusion/             Feature standardization, meta-classifier, calibration
|-- explainability/     Grad-CAM++, heatmaps, region localization
|-- metadata/           EXIF / codec / compression supporting-evidence analysis
|-- evaluation/         Robustness (degradation + adversarial), cross-dataset eval
|-- backend/            FastAPI app: routes, schemas, orchestration
|-- frontend/           React + Vite UI
|-- demo/               One-shot CLI demo script
|-- models/             Trained weights (gitignored, see setup)
|-- datasets/           Dataset folders + metadata CSVs (synthetic placeholder shipped)
|-- configs/            Shared pipeline.yaml -- paths, thresholds, module I/O contracts
|-- tests/              288 tests covering every module above
|-- Dockerfile.backend / Dockerfile.frontend / docker-compose.yml
`-- requirements.txt
```

Every stream module follows the same convention: a `train.py` (or equivalent) that produces a checkpoint under `models/`, and an `inference.py` that loads it and exposes a `score_face()` / `score_faces()` method returning a fixed, documented contract (e.g. `{"face_id": int, "spatial_score": float}`). These contracts live in `configs/pipeline.yaml` under `contracts:` as the single source of truth — every module reads from there rather than hardcoding field names, which is what lets independently-built streams plug into fusion without drifting out of sync.

---

## How the pipeline works

1. **Preprocessing** — a frame (or the whole image) goes through face detection (MediaPipe, with an offline Haar-cascade fallback), landmark-based alignment to a standardized 224x224 crop, and — for video — IoU-based tracking so the same person keeps the same `face_id` across frames.
2. **Spatial stream** — EfficientNet-B0 examines the raw face crop for visual manipulation artifacts.
3. **Frequency stream** — FFT and DCT expose periodic artifacts in the frequency domain that generation models often leave behind, invisible in the raw pixels.
4. **Noise stream** — SRM high-pass filters isolate the sensor-noise residual; manipulation disturbs this fingerprint even when the visible image looks clean.
5. **Physiological stream** *(video only)* — Eye Aspect Ratio tracks blink rate/duration against natural human ranges; rPPG extracts a subtle skin-color pulse signal and checks whether it's genuinely periodic in the plausible heart-rate band.
6. **Temporal analysis** *(video only)* — aggregates per-frame stream scores using moving averages and a frame-to-frame consistency penalty: flickering evidence is itself treated as suspicious, not just averaged away.
7. **Metadata analysis** — EXIF/codec/compression flags feed fusion as *supporting* evidence only; no single metadata flag can push a verdict past mild suspicion on its own.
8. **Fusion** — all seven signals are combined by a meta-classifier (XGBoost, chosen automatically after comparison against Logistic Regression and a small MLP).
9. **Calibration** — temperature scaling turns the raw fusion probability into a genuinely trustworthy confidence value.
10. **Explainability** — Grad-CAM++ (implemented directly via hooks on EfficientNet's `bn2` layer) produces a heatmap; the heatmap is broken into 8 named face regions, ranked by activation intensity, and turned into an evidence-grounded text explanation — every sentence cites an actual computed number, never a canned template.

---

## Testing

```bash
pytest tests/ -q
```

**288 tests, all passing.** Every phase of the pipeline has its own dedicated test file (`tests/test_<phase>.py`), covering unit-level correctness, integration across phases, and full end-to-end orchestration through real HTTP requests via FastAPI's `TestClient`.

Slow tests (leave-one-dataset-out retraining) are marked and can be skipped:

```bash
pytest tests/ -q -m "not slow"
```

Some tests are automatically skipped if their corresponding checkpoint hasn't been trained yet — run the training commands in [step 7](#7-build-the-dataset-and-train-every-model) first for full coverage.

---

## Docker deployment

```bash
docker compose up --build
```

Starts the backend (FastAPI, port `8000`) and frontend (React, built and served via nginx, port `3000`) as separate containers. Trained model weights must already exist under `models/` before building.

> **Honesty note:** `Dockerfile.backend`, `Dockerfile.frontend`, and `docker-compose.yml` were written and reviewed carefully for correctness (correct system libraries for OpenCV/MediaPipe, correct multi-stage frontend build, correct proxy routing) but could not be build-tested in the sandbox this project was developed in (no Docker daemon available there). Run a real `docker compose up --build` and fix anything that surfaces before a live demo.

### ONNX Runtime (optional, faster CPU inference)

```bash
python -m spatial.export_onnx
```

Produces `models/spatial/efficientnet_b0.onnx`, verified numerically identical to the PyTorch model's output (max difference `0.0` across 10 diverse test inputs — see `tests/test_deployment.py`), and measured ~1.76x faster on CPU in local testing. Use `spatial.onnx_inference.SpatialONNXInference` as a drop-in replacement for `spatial.inference.SpatialInference`.

---

## Known limitations

This project was built with one rule throughout: **every simplification gets stated out loud, not hidden.**

| Limitation | Detail | What closes it |
|---|---|---|
| Synthetic placeholder dataset | `datasets/` ships with 48 tiny, non-photographic files — enough to validate every pipeline stage runs correctly, not enough for any real accuracy claim. | Drop in real datasets (FaceForensics++, DFDC, Celeb-DF v2, DeeperForensics-1.0) using the same folder structure. |
| No pretrained ImageNet weights by default | `pretrained=True` requires downloading weights from HuggingFace Hub, which needed network access unavailable during development. All shipped checkpoints were trained `pretrained=False`. | Run `python -m spatial.train --pretrained` with network access. |
| MediaPipe models not bundled | Real face/landmark detection needs a network download (see setup step 6). Without it, the system falls back to an OpenCV Haar cascade and a fixed-geometry landmark estimate. | Fetch the `.task`/`.tflite` files and update `configs/pipeline.yaml`. |
| Blink/rPPG accuracy depends on real landmarks | The geometric fallback returns fixed coordinates regardless of pixel content, so blink detection is structurally correct but not signal-accurate until real MediaPipe landmarks are in use. | Same as above. |
| Fusion training data has neutral-default video streams | `fusion/build_dataset.py` sets `blink_score`/`rppg_score`/`temporal_score` to 0.5 for every training row, since the dataset loader is file-level, not frame-sequence-level. Real values are computed correctly at inference time. | Extend `fusion/build_dataset.py` to run the physiological stream over real video files. |
| Multi-stream cross-dataset evaluation has partial leakage | The fusion classifier is retrained fresh per held-out dataset, but the spatial/frequency/noise scores it trains on come from models that saw all datasets during their own training. Flagged in the evaluation's own return value. | Retrain spatial/frequency/noise fresh per fold too, with real data and more compute. |
| Docker not build-tested | See Docker deployment above. | Run `docker compose up --build` where a Docker daemon is available. |
| Multi-face aggregation policy is a design choice | The plan doesn't specify how to combine multiple faces' verdicts; see the stated policy in API reference. | Revisit if the deployment context needs a different policy. |

None of these are hidden — each surfaced through this project's own test suite or a deliberate real-input sanity check, and each is flagged inline in the relevant module's docstring too.

---

## Development order

Built and verified phase by phase, each one tested with real inputs before moving to the next:

1. Foundation (repo structure, shared config, environment check)
2. Dataset engine (leakage-safe train/val/test split)
3. Face processing pipeline (detection, alignment, tracking)
4. Spatial detection model (EfficientNet-B0)
5. Frequency stream (FFT/DCT)
6. Noise residual stream (SRM filters)
7. Physiological stream (blink/EAR, rPPG)
8. Temporal analysis (moving averages, consistency penalty)
9. Metadata analysis (EXIF/codec, supporting evidence only)
10. Fusion engine (XGBoost vs Logistic Regression vs MLP)
11. Confidence calibration (temperature scaling)
12. Explainable AI (Grad-CAM / Grad-CAM++)
13. Tampering localization (per-face evidence association)
14. Backend (FastAPI)
15. Frontend (React + Vite)
16. Image-first prototype (full chain verification)
17. Video prototype (full chain verification with genuine time-varying signal)
18. Robustness testing (degradation + adversarial FGSM/PGD)
19. Test-time augmentation
20. Cross-dataset evaluation (leave-one-dataset-out)
21. Deployment (Docker, ONNX)

---

## Team

| Member | Ownership |
|---|---|
| M1 | Dataset + preprocessing + face pipeline |
| M2 | Spatial AI |
| M3 | Frequency + noise |
| M4 | rPPG + blink + temporal |
| M5 | Fusion + calibration + XAI |
| M6 | FastAPI + React + integration + deployment |

**Project:** TrueSight -- Team CipherForge
**Event:** Smart India Hackathon 2026
**Theme:** Cyber Security
