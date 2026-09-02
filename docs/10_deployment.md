# 10 — Deployment

This document covers all ways to run BrainTumorXAI: local development, REST API server, and Docker container.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.12 recommended (matches Docker base) |
| pip | Latest | For installing dependencies |
| Git | Any | For cloning the repository |
| Docker | Any | Optional — only for containerized deployment |

No GPU is required. The system runs fully on CPU.

---

## Option 1: Local Setup (Recommended for Development)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/bhola-dev58/brain-xai-ensemble.git
cd brain-xai-ensemble
```

---

### Step 2 — Create a Virtual Environment

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

---

### Step 3 — Install Dependencies

All versions are pinned in `requirements.txt` for reproducibility.

```bash
pip install -r requirements.txt
```

**What gets installed:**

| Package | Version | Purpose |
|---|---|---|
| `tensorflow` | 2.21.0 | Deep learning engine |
| `numpy` | 2.4.4 | Numerical computing |
| `opencv-python` | 4.13.0.92 | Image processing |
| `matplotlib` | 3.10.9 | Optional visualization |
| `gradio` | 6.14.0 | Web UI framework |
| `fastapi` | 0.136.1 | REST API framework |
| `uvicorn` | 0.46.0 | ASGI server for FastAPI |
| `python-multipart` | 0.0.27 | File upload support for FastAPI |

---

### Step 4 — Verify Model Files Exist

```bash
ls models/
# Should show:
#   densenet_best.h5   (~37 MB)
#   inception_best.h5  (~102 MB)
```

If model files are missing, contact the project maintainer to obtain the pre-trained weights.

---

### Step 5 — Run the Gradio Dashboard

```bash
python app.py
```

Expected output:
```
INFO  BrainTumorXAI: Initializing ensemble models...
INFO  BrainTumorXAI: Models loaded successfully.
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxxxxx.gradio.live
```

Open `http://127.0.0.1:7860` in your browser.

---

### Step 6 — Run the REST API Server (Optional)

The API server runs separately from the dashboard on a different port.

```bash
python scripts/run_api.py
```

Expected output:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Open `http://localhost:8000/docs` for the Swagger UI.

---

### Step 7 — Run Tests (Optional)

```bash
pytest tests/ -v
```

Expected: `33 passed`

---

## Option 2: Docker Container

### Build the Image

```bash
docker build -t brain-tumor-xai .
```

This builds a `python:3.12-slim` based container with:
- System libraries: `libgl1`, `libglib2.0-0` (OpenCV runtime dependencies)
- Python packages from `requirements.txt`
- App source code and model files
- Port 7860 exposed

### Run the Container

```bash
docker run -p 7860:7860 brain-tumor-xai
```

Then open `http://localhost:7860` in your browser.

### Environment Variables (Built Into Dockerfile)

| Variable | Value | Purpose |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `-1` | Forces CPU-only inference — disables GPU |
| `TF_CPP_MIN_LOG_LEVEL` | `3` | Suppresses TensorFlow C++ verbose logs |
| `PYTHONUNBUFFERED` | `1` | Ensures real-time stdout logging in containers |

### What Is Excluded From Docker Image

The `.dockerignore` file excludes:

```
venv/
brain-tumor-2d-dataset/
__pycache__/
*.pyc
.git/
.pytest_cache/
tests/
```

This keeps the image lean — no training dataset, no virtual environment, no test files.

---

## Option 3: Running Both Dashboard + API Simultaneously

Both services can run at the same time in separate terminals:

**Terminal 1 — Gradio Dashboard:**
```bash
source venv/bin/activate
python app.py
# → http://localhost:7860
```

**Terminal 2 — REST API:**
```bash
source venv/bin/activate
python scripts/run_api.py
# → http://localhost:8000
```

Both share the same model weights from `models/`. The models are loaded independently in each process.

---

## Environment Variables (Optional Overrides)

You can override settings via environment variables before launching:

```bash
# Suppress TensorFlow logs
export TF_CPP_MIN_LOG_LEVEL=3

# Force CPU (already set in inference.py, but can set here too)
export CUDA_VISIBLE_DEVICES=-1

# Then run
python app.py
```

---

## Production Considerations

| Concern | Recommendation |
|---|---|
| **Authentication** | Add OAuth2 / API key middleware to FastAPI before exposing publicly |
| **HTTPS** | Put the API behind a reverse proxy (nginx) with SSL termination |
| **GPU Acceleration** | Remove `CUDA_VISIBLE_DEVICES=-1` from config if GPU is available |
| **Rate Limiting** | Add `slowapi` or similar to FastAPI for production endpoints |
| **DICOM Support** | Integrate `pydicom` to accept `.dcm` files and convert to numpy before inference |
| **Model Versioning** | Use MLflow or DVC to track model versions alongside data versions |
| **Scaling** | Deploy multiple Uvicorn workers behind a load balancer: `uvicorn src.api:app --workers 4` |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `OSError: models/densenet_best.h5 not found` | Ensure both `.h5` files exist in the `models/` directory |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root directory, not from inside `src/` |
| `libGL.so.1: cannot open shared object` | Install system libs: `apt-get install -y libgl1 libglib2.0-0` |
| Port 7860 already in use | Kill existing process: `lsof -ti:7860 \| xargs kill` |
| Dashboard loads but shows blank panels | Clear browser cache or use incognito mode |
| Inference very slow | First run takes 30–60s due to model initialization. Subsequent runs are faster. |
