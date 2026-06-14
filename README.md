# BrainTumorXAI: Explainable Brain Tumor Detection and Segmentation

An advanced, clinical-grade medical imaging application that uses **Ensemble Deep Learning** to detect, classify, and segment brain tumors from MRI scans. The project is centered on **Explainable AI (XAI)**, providing visual evidence (Grad-CAM heatmaps) alongside every prediction so clinicians can understand and trust the model's reasoning.

---

## Features

- **Ensemble Classification:** Combines DenseNet121 and InceptionV3 using Soft Voting for high-reliability, low-false-positive results.
- **Explainable AI (Grad-CAM):** Generates heatmaps that highlight exactly which regions of the MRI the model focused on.
- **Automated Tumor Segmentation:** Produces a tumor region mask using adaptive morphological image processing driven by the Grad-CAM activation.
- **Tumor Property Analytics:** Estimates anatomical location (brain lobe), tumor area percentage, and a severity rating.
- **Interactive Dashboard:** A dark-themed Gradio interface with clickable sample scans for instant testing.
- **REST API:** A FastAPI backend exposing the full inference pipeline as HTTP endpoints.
- **Docker Support:** A production-ready Dockerfile for containerized deployment.
- **Automated Test Suite:** 33 pytest tests covering unit logic and full integration pipeline.

---

## Technology Stack

| Layer          | Technology                            |
|----------------|---------------------------------------|
| Deep Learning  | TensorFlow 2.21, Keras                |
| Models         | DenseNet121, InceptionV3              |
| Explainability | Grad-CAM                              |
| Vision         | OpenCV 4.x, NumPy                     |
| Web UI         | Gradio 6.x                            |
| REST API       | FastAPI, Uvicorn                      |
| Testing        | pytest                                |
| Container      | Docker (CPU-only, Python 3.12)        |
| Language       | Python 3.10+                          |

---

## Project Structure

```
Brain_Tumor_Project/
├── app.py                        # Main entrypoint — launches the Gradio dashboard
├── Dockerfile                    # Production container definition (CPU-only)
├── .dockerignore                 # Excludes dev/build artifacts from Docker context
├── requirements.txt              # Pinned Python dependencies
│
├── src/                          # Core application package
│   ├── __init__.py
│   ├── config.py                 # Centralized constants, paths, thresholds, logging
│   ├── inference.py              # Ensemble model loading, prediction, Grad-CAM
│   ├── processor.py              # Image segmentation, location & severity estimation
│   ├── dashboard.py              # Gradio UI layout, HTML formatting, CSS styling
│   └── api.py                    # FastAPI REST backend
│
├── scripts/                      # Offline tooling scripts
│   ├── run_api.py                # Launches the FastAPI server
│   └── train_segmentation.py     # U-Net segmentation training script
│
├── tests/                        # Automated test suite
│   ├── test_processor.py         # 18 unit tests for image processing logic
│   └── test_inference.py         # 15 integration tests for the full pipeline
│
├── models/                       # Pre-trained model weights (not in version control)
│   ├── densenet_best.h5
│   └── inception_best.h5
│
├── brain-tumor-2d-dataset/       # Training dataset (images and masks)
└── test_images/                  # Sample MRI scans for quick testing
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git

---

### 1. Clone the Repository

```bash
git clone https://github.com/bhola-dev58/final_year_project_brain_tumor_segmentation_explaination.git
cd final_year_project_brain_tumor_segmentation_explaination
```

---

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

### 3. Install Dependencies

All versions are pinned for reproducibility.

```bash
pip install -r requirements.txt
```

---

### 4. Run the Gradio Dashboard

```bash
python app.py
```

Once running, open the local URL in your browser:

```
http://127.0.0.1:7860
```

A shareable public link is also printed to the terminal automatically (valid for 1 week).

The dashboard includes a **Quick Examples** section — click any sample thumbnail to load it instantly without manually uploading a file.

---

### 5. Run the REST API Server (Optional)

The FastAPI server runs independently from the Gradio UI on port 8000.

```bash
python scripts/run_api.py
```

Available endpoints:

| Method | Endpoint        | Description                                      |
|--------|-----------------|--------------------------------------------------|
| GET    | /api/health     | Liveness check — confirms models are loaded      |
| POST   | /api/predict    | Upload an MRI image, receive full diagnosis JSON |

Interactive API documentation (Swagger UI):

```
http://localhost:8000/docs
```

Example health check using curl:

```bash
curl http://localhost:8000/api/health
```

Example prediction using curl:

```bash
curl -X POST http://localhost:8000/api/predict \
     -F "file=@test_images/Tr-me_0025.jpg"
```

The response includes `class_name`, `confidence`, `is_tumor`, `location`, `tumor_percentage`, `severity`, `inference_time`, and base64-encoded `gradcam_overlay_b64` and `segmentation_b64` images.

---

### 6. Run the Automated Tests

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests (fast, no model loading)
pytest tests/test_processor.py -v

# Run only integration tests (loads models, ~30 seconds)
pytest tests/test_inference.py -v
```

Expected result: **33 passed** (18 unit + 15 integration).

---

### 7. Run with Docker (Optional)

Build the image:

```bash
docker build -t brain-tumor-xai .
```

Run the container:

```bash
docker run -p 7860:7860 brain-tumor-xai
```

Then open `http://localhost:7860` in your browser.

The container runs CPU-only inference with no GPU required. The training dataset and virtual environment are excluded from the image to keep it lean.

---

## Dataset

The classification models were trained on a Brain Tumor MRI dataset with 4 classes:

1. Glioma Tumor
2. Meningioma Tumor
3. Pituitary Tumor
4. No Tumor

The segmentation script (`scripts/train_segmentation.py`) trains a lightweight U-Net on paired image and mask files stored in `brain-tumor-2d-dataset/`.

---

## How It Works

1. The uploaded MRI scan is resized to 224x224 and normalized.
2. Both DenseNet121 and InceptionV3 independently predict class probabilities.
3. The two probability vectors are averaged (Soft Voting) to produce the final classification.
4. Grad-CAM backpropagation is run on DenseNet121 to generate a spatial heatmap showing which pixels influenced the prediction.
5. The heatmap is used to define a region of interest, inside which Otsu thresholding extracts the tumor boundary.
6. Location (brain lobe), area percentage, and severity are computed and reported alongside the visual overlays.

---

## Medical Disclaimer

This software is for **educational and research purposes only**. It is not intended for clinical use and must not replace professional medical diagnosis. Always consult a qualified and licensed healthcare professional for any medical concerns.

---

**Author:** Bhola Yadav
**Project:** Final Year / Major Project — Explainable Brain Tumor Detection and Segmentation
