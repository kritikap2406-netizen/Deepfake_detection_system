# TrueSight : Multi-stream deepfake detection system


Smart India Hackathon 2024 problem statement · Theme: Cyber Security · 

TrueSight looks at a face in an image or video and answers one question: *has this face been digitally manipulated?* Instead of trusting a single AI model, it runs **four independent detection streams** on every face it finds, combines their evidence with a calibrated decision model, and — critically — **explains its reasoning** with heatmaps and plain-language evidence, instead of just printing "Fake" or "Real" with no justification.

This document is written so that someone who has never seen this codebase before — including a total beginner — can get it running and understand what it's doing.

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

- [What this project is](#what-this-project-is)
- [Why "multi-stream"?](#why-multi-stream)
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
- [Glossary](#glossary)

---

## What this project is

TrueSight is a **working, end-to-end prototype** — every module below is real, tested code, not a stub or a placeholder that returns a hardcoded answer. Concretely, this repository contains:

- A real **EfficientNet-B0** spatial classifier, trained with focal loss, class balancing, and JPEG/blur/noise/resize augmentation, that looks for visual manipulation artifacts in a face crop.
- Real **FFT/DCT frequency-domain analysis** and real **SRM (Spatial Rich Model) noise-residual filters**, each feeding a lightweight from-scratch CNN — these catch artifacts that generative models leave behind but that are invisible to the naked eye.
- Real **blink detection** (Eye Aspect Ratio) and real **rPPG** (remote photoplethysmography — detecting a person's pulse from subtle skin-color changes on video), with Butterworth bandpass filtering, verified to correctly tell a genuine periodic pulse apart from noise.
- A real **fusion comparison** between XGBoost, Logistic Regression, and a small MLP, with the best-performing one selected automatically.
- Real **temperature-scaling calibration**, verified to reduce Negative Log-Likelihood and Expected Calibration Error on a synthetically miscalibrated model — this is what turns a raw model score into a confidence number you can actually trust.
- Real **Grad-CAM++**, implemented directly via forward/backward hooks on the EfficientNet architecture, producing genuine heatmap images (not simulated ones).
- A real **FastAPI backend** and a real **React + Vite frontend**, tested against each other over actual HTTP requests, not mocks.
- A real **ONNX export**, numerically verified to match the PyTorch model's output exactly.

At the same time, this project is upfront about what it can't yet claim. The dataset shipped in `datasets/` is a tiny **synthetic placeholder** (48 non-photographic files) used purely to prove that every stage of the pipeline runs correctly end-to-end. **No accuracy number in this repository should be read as a real-world performance claim.** See [Known limitations](#known-limitations) for the full, specific list of gaps and exactly what closes each one.

---

## Why "multi-stream"?

A single deepfake detector trained only on pixels can be fooled, or can overfit to artifacts specific to one generation method. TrueSight instead looks for manipulation evidence in four *independent* places at once, so a fake has to fool all of them simultaneously to slip through:

| Stream | What it looks at | Why it catches fakes |
|---|---|---|
| **Spatial** | Raw pixels of the face | Visual blending, warping, and texture inconsistencies |
| **Frequency** | FFT/DCT of the image | Periodic artifacts generative models leave in the frequency domain, invisible in raw pixels |
| **Noise** | Sensor-noise residual (SRM filters) | Manipulation disturbs a camera's noise "fingerprint" even when the image looks clean |
| **Physiological** *(video only)* | Blink rate, pulse signal (rPPG) | Deepfakes struggle to fake a biologically consistent heartbeat or natural blink pattern |

The results are then merged by a fusion model rather than simply averaged, so the system can learn which streams to trust more in which situations.

---

## Quick start

If you just want to see it run:

```bash
git clone <this-repo>
cd TrueSight
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train every model on the bundled placeholder dataset (fast — a few minutes total)
python -m preprocessing.build_metadata
python -m preprocessing.split_dataset
python -m spatial.train
python -m frequency_noise.train
python -m frequency_noise.train_noise
python -m fusion.build_dataset
python -m fusion.meta_classifier
python -m fusion.train_calibration

# Run the demo on one file
python -m demo.run_demo path/to/your/image.jpg
```

For the full web experience (upload a file in a browser and see heatmaps), see [Running the system](#running-the-system).

---

## Full setup, step by step

This section explains each step in more detail — useful if something in Quick Start didn't work, or if you want to understand what each command actually does.

### 1. Prerequisites

| Requirement | Version | Why you need it |
|---|---|---|
| Python | 3.11+ | Runs the entire ML pipeline and backend |
| Node.js | 18+ | Builds and runs the frontend |
| pip | latest | Installs Python packages |
| CUDA GPU | optional | Speeds up training; everything also works on a normal CPU, just slower |
| Docker | optional | Only needed if you want containerized deployment |

### 2. Clone the repository and create a virtual environment

A virtual environment keeps this project's Python packages separate from everything else on your machine, so nothing conflicts.

```bash
git clone <this-repo>
cd TrueSight
python3.11 -m venv venv
source venv/bin/activate      # on Windows, use: venv\Scripts\activate
```

You'll know it worked if your terminal prompt now starts with `(venv)`.

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

This one command installs everything the project needs: PyTorch and torchvision (deep learning), `timm` (EfficientNet backbone), MediaPipe (face detection), OpenCV and NumPy/SciPy (image and signal processing), XGBoost and scikit-learn (the fusion model), FastAPI and Uvicorn (the backend server), ONNX/ONNX Runtime (faster deployment), and Pillow (image I/O).

> **MediaPipe note:** by default, face detection and landmark extraction use a real MediaPipe model — but the model file (`.task` format) has to be downloaded separately (see step 6 below). Until you do that, the pipeline automatically falls back to a simpler OpenCV Haar-cascade detector and a fixed-geometry landmark estimate. This keeps everything runnable offline from the start, but detection accuracy — and blink/rPPG signal quality specifically — will be noticeably worse until you fetch the real model. The fallback is intentional, and it's logged clearly whenever it kicks in.

### 4. Verify your environment

```bash
python -m tests.env_check
```

This checks that PyTorch, CUDA (if you have a GPU), OpenCV, MediaPipe, NumPy, FastAPI, scikit-learn, and XGBoost are all correctly installed and importable, and tells you whether it found a GPU. Run this before anything else — it catches most setup problems in a few seconds.

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

If you place the files somewhere else, update the paths in `configs/pipeline.yaml` to match. Skipping this step is fine for exploring the codebase, but face detection and the physiological (blink/pulse) stream won't be accurate without it.

### 7. Build the dataset and train every model

The repo ships with a small **synthetic placeholder dataset** under `datasets/`, so you can run the entire pipeline on day one without needing a real deepfake dataset. For anything beyond exploring how the pipeline works, swap in a real dataset (FaceForensics++, DFDC, Celeb-DF v2, or DeeperForensics-1.0) using the same folder layout.

```bash
# 1. Build dataset metadata (file path, dataset, label, manipulation type, video id, person id)
python -m preprocessing.build_metadata

# 2. Leakage-safe train/val/test split (splits by identity, never by individual frame,
#    so the same person never appears in both training and test data)
python -m preprocessing.split_dataset

# 3. Train the spatial stream (EfficientNet-B0)
python -m spatial.train --epochs 3

# 4. Train the frequency stream (FFT/DCT-based lightweight CNN)
python -m frequency_noise.train --epochs 3

# 5. Train the noise stream (SRM-residual-based lightweight CNN)
python -m frequency_noise.train_noise --epochs 3

# 6. Build the fusion training set (runs every trained stream over the split data)
python -m fusion.build_dataset

# 7. Train and compare XGBoost / Logistic Regression / a small MLP, and keep the best one
python -m fusion.meta_classifier

# 8. Fit temperature-scaling calibration on the fusion model's validation predictions
python -m fusion.train_calibration
```

Once this finishes, every model file the backend needs exists under `models/`.

### 8. (Optional) Export the spatial model to ONNX for faster inference

```bash
python -m spatial.export_onnx
```

---

## Running the system

### Option A — full backend + frontend (recommended)

Open two terminal windows.

**Terminal 1 — start the backend:**

```bash
source venv/bin/activate
uvicorn backend.main:app --reload
```

The API is now live at `http://127.0.0.1:8000`, with interactive, browsable documentation at `http://127.0.0.1:8000/docs`.

**Terminal 2 — start the frontend:**

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173` in your browser. The dev server automatically forwards API requests to the backend (see `frontend/vite.config.js`), so you don't need to configure anything else.

Upload an image or video and you'll see: the REAL/FAKE verdict, a calibrated confidence score, a plain-language evidence-based explanation, and — for every detected face — a Grad-CAM++ heatmap, all stream scores, and the specific facial regions flagged as suspicious.

### Option B — quick one-shot CLI demo

If you just want to test one file without launching a server:

```bash
python -m demo.run_demo path/to/image.jpg
python -m demo.run_demo path/to/video.mp4
```

This runs the complete pipeline on that one file and prints the full result as JSON straight to your terminal.

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET`  | `/health` | Reports whether every required model checkpoint is present and ready |
| `POST` | `/analyze/image` | Upload a JPG/PNG, get back the full analysis |
| `POST` | `/analyze/video` | Upload an MP4/AVI/MOV/MKV, get back the full analysis |
| `GET`  | `/result/{id}` | Re-fetch a previously computed result by its id |
| `GET`  | `/static/heatmaps/{filename}` | Serves the generated Grad-CAM++ heatmap images |

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

A few things worth explaining about that response:

- `blink_score`, `rppg_score`, and `temporal_score` stay at the neutral default `0.5` for **image** input — these three streams are inherently video-only, since they need a sequence of frames from the same tracked face to mean anything. For **video** input, all seven score fields carry genuine, computed values.
- `metadata_flag` comes from checking EXIF/codec/compression data. It's *supporting* evidence only — on its own, no metadata flag can push a verdict past "mildly suspicious."
- **Multi-face aggregation policy:** if *any* detected face is predicted FAKE, the overall verdict is FAKE, with confidence equal to the highest calibrated confidence among the FAKE-predicted faces. If every face is predicted REAL, the overall confidence is `1 - (highest fake-probability seen across any face)`. This is a deliberate design decision (the original spec didn't dictate one) — see `backend/services/orchestrator.py` for the implementation.

---

## Project structure

```
TrueSight/
|-- preprocessing/     Face detection, alignment, tracking, dataset engine
|-- spatial/           EfficientNet-B0 classifier: training, test-time augmentation, ONNX export
|-- frequency_noise/   FFT/DCT frequency stream + SRM noise stream
|-- physiological/     Blink (EAR), rPPG, temporal aggregation
|-- fusion/             Feature standardization, meta-classifier, calibration
|-- explainability/    Grad-CAM++, heatmaps, region localization
|-- metadata/          EXIF / codec / compression supporting-evidence analysis
|-- evaluation/        Robustness testing (degradation + adversarial), cross-dataset eval
|-- backend/           FastAPI app: routes, schemas, orchestration
|-- frontend/          React + Vite UI
|-- demo/              One-shot CLI demo script
|-- models/            Trained weights (gitignored — created by the training steps above)
|-- datasets/          Dataset folders + metadata CSVs (synthetic placeholder shipped)
|-- configs/           Shared pipeline.yaml — paths, thresholds, module I/O contracts
|-- tests/             288 tests covering every module above
|-- Dockerfile.backend / Dockerfile.frontend / docker-compose.yml
`-- requirements.txt
```

Every stream module follows the same convention, which is worth understanding if you plan to read or extend the code: a `train.py` (or equivalent) produces a checkpoint saved under `models/`, and an `inference.py` loads that checkpoint and exposes a `score_face()` / `score_faces()` method that returns a fixed, documented output shape (e.g. `{"face_id": int, "spatial_score": float}`). These output shapes are all defined in one place — `configs/pipeline.yaml`, under `contracts:` — rather than hardcoded inside each module. That's what lets streams that were built independently plug into fusion without silently drifting out of sync with each other.

---

## How the pipeline works

A step-by-step walk-through of what happens to a file from the moment it's uploaded to the moment a verdict comes back:

1. **Preprocessing** — the frame (or the whole image) goes through face detection (MediaPipe, with an offline Haar-cascade fallback), landmark-based alignment into a standardized 224×224 crop, and, for video, IoU-based tracking so the same person keeps the same `face_id` across frames.
2. **Spatial stream** — EfficientNet-B0 examines the raw face crop for visual manipulation artifacts.
3. **Frequency stream** — FFT and DCT expose periodic artifacts in the frequency domain that generation models often leave behind, invisible in the raw pixels.
4. **Noise stream** — SRM high-pass filters isolate the sensor-noise residual; manipulation disturbs this fingerprint even when the visible image looks clean.
5. **Physiological stream** *(video only)* — Eye Aspect Ratio tracks blink rate and duration against natural human ranges, while rPPG extracts a subtle skin-color pulse signal and checks whether it's genuinely periodic within a plausible heart-rate range.
6. **Temporal analysis** *(video only)* — aggregates per-frame stream scores using moving averages and a frame-to-frame consistency penalty: flickering evidence is itself treated as suspicious, not simply averaged away.
7. **Metadata analysis** — EXIF/codec/compression flags feed into fusion as supporting evidence only.
8. **Fusion** — all seven signals are combined by a meta-classifier (XGBoost, automatically chosen after comparison against Logistic Regression and a small MLP).
9. **Calibration** — temperature scaling turns the raw fusion probability into a genuinely trustworthy confidence value, so "90% confidence" actually means what it sounds like.
10. **Explainability** — Grad-CAM++ (implemented directly via hooks on EfficientNet's `bn2` layer) produces a heatmap; that heatmap is broken into 8 named facial regions, ranked by activation intensity, and turned into an evidence-grounded text explanation — every sentence cites an actual computed number rather than a canned template.

---

## Testing

```bash
pytest tests/ -q
```

**288 tests, all passing.** Every phase of the pipeline has its own dedicated test file (`tests/test_<phase>.py`), covering unit-level correctness of individual functions, integration across phases, and full end-to-end checks made through real HTTP requests via FastAPI's `TestClient`.

To skip the slower tests (which retrain models in a leave-one-dataset-out setup):

```bash
pytest tests/ -q -m "not slow"
```

Note: some tests are automatically skipped if their corresponding model checkpoint hasn't been trained yet — run the training commands in [step 7](#7-build-the-dataset-and-train-every-model) first if you want full test coverage.

---

## Docker deployment

```bash
docker compose up --build
```

This starts the backend (FastAPI, port `8000`) and the frontend (React, built and served via nginx, port `3000`) as two separate containers. Trained model weights must already exist under `models/` before you build the images.

> **Honesty note:** `Dockerfile.backend`, `Dockerfile.frontend`, and `docker-compose.yml` were written and reviewed carefully for correctness (correct system libraries for OpenCV/MediaPipe, a correct multi-stage frontend build, correct reverse-proxy routing) but could not be build-tested in the sandbox this project was developed in, since no Docker daemon was available there. Run a real `docker compose up --build` yourself and fix anything that surfaces before relying on it for a live demo.

### ONNX Runtime (optional, faster CPU inference)

```bash
python -m spatial.export_onnx
```

This produces `models/spatial/efficientnet_b0.onnx`, verified numerically identical to the PyTorch model's output (maximum difference of `0.0` across 10 diverse test inputs — see `tests/test_deployment.py`), and measured roughly 1.76× faster on CPU in local testing. Use `spatial.onnx_inference.SpatialONNXInference` as a drop-in replacement for `spatial.inference.SpatialInference` wherever CPU speed matters.

---

## Known limitations

This project was built around one rule: **every simplification gets stated out loud, not hidden.**

| Limitation | Detail | What closes it |
|---|---|---|
| Synthetic placeholder dataset | `datasets/` ships with 48 tiny, non-photographic files — enough to validate that every pipeline stage runs correctly, not enough for any real accuracy claim. | Drop in real datasets (FaceForensics++, DFDC, Celeb-DF v2, DeeperForensics-1.0) using the same folder structure. |
| No pretrained ImageNet weights by default | `pretrained=True` requires downloading weights from the HuggingFace Hub, which needed network access unavailable during development. All shipped checkpoints were trained with `pretrained=False`. | Run `python -m spatial.train --pretrained` with network access. |
| MediaPipe models not bundled | Real face/landmark detection needs a network download (see setup step 6). Without it, the system falls back to an OpenCV Haar cascade and a fixed-geometry landmark estimate. | Fetch the `.task`/`.tflite` files and update `configs/pipeline.yaml`. |
| Blink/rPPG accuracy depends on real landmarks | The geometric fallback returns fixed coordinates regardless of pixel content, so blink detection is structurally correct but not signal-accurate until real MediaPipe landmarks are in use. | Same as above. |
| Fusion training data has neutral-default video streams | `fusion/build_dataset.py` sets `blink_score`/`rppg_score`/`temporal_score` to 0.5 for every training row, since the dataset loader is file-level, not frame-sequence-level. Real values are still computed correctly at inference time. | Extend `fusion/build_dataset.py` to run the physiological stream over real video files. |
| Multi-stream cross-dataset evaluation has partial leakage | The fusion classifier is retrained fresh per held-out dataset, but the spatial/frequency/noise scores it trains on come from models that saw all datasets during their own training. This is flagged in the evaluation's own return value. | Retrain spatial/frequency/noise fresh per fold too, with real data and more compute. |
| Docker not build-tested | See the Docker deployment section above. | Run `docker compose up --build` in an environment with a Docker daemon available. |
| Multi-face aggregation policy is a design choice | The original plan didn't specify how to combine multiple faces' verdicts; see the stated policy in the API reference section. | Revisit if a real deployment needs a different policy. |

None of these are hidden — each surfaced through this project's own test suite or a deliberate real-input sanity check, and each is also flagged inline in the relevant module's docstring.

---

## Development order

Built and verified phase by phase, with each phase tested against real inputs before moving on to the next:

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

**Project:** TrueSight — Team CipherForge
**Event:** Smart India Hackathon 2026
**Theme:** Cyber Security

---

## Glossary

A quick reference for terms used throughout this document, for anyone new to deepfake detection or ML:

- **Deepfake** — media (image or video) where a face has been synthetically generated or swapped using AI.
- **EfficientNet-B0** — a compact, efficient convolutional neural network architecture used here as the spatial stream's backbone.
- **FFT / DCT** — Fast Fourier Transform / Discrete Cosine Transform; mathematical tools that convert an image into its frequency components, revealing patterns invisible in raw pixels.
- **SRM (Spatial Rich Model)** — a set of high-pass filters originally designed for steganalysis, repurposed here to expose sensor-noise inconsistencies.
- **rPPG (remote photoplethysmography)** — a technique for detecting a person's heartbeat from tiny, invisible-to-the-eye color changes in video of their skin.
- **EAR (Eye Aspect Ratio)** — a simple geometric ratio computed from eye landmarks, used to detect blinks.
- **Fusion / meta-classifier** — the model that takes all individual stream scores and combines them into one final decision.
- **Calibration** — the process of adjusting a model's raw output so that its confidence score reflects true likelihood (a well-calibrated 80% confidence should be right about 80% of the time).
- **Grad-CAM++** — a technique for visualizing which parts of an image a neural network focused on when making its decision, produced as a heatmap.
- **ONNX** — an open format for representing trained models so they can run efficiently across different hardware/runtimes.
