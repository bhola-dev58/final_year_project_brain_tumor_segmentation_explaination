# BrainTumorXAI: Explainable Brain Tumor Detection and Segmentation

An advanced, clinical-grade medical imaging application that uses **Ensemble Deep Learning** to detect, classify, and segment brain tumors from MRI scans. The project is centered on **Explainable AI (XAI)**, providing visual evidence (Grad-CAM heatmaps) alongside every prediction so clinicians can understand and trust the model's reasoning.

---

## Features

- **Ensemble Classification:** Combines InceptionV3 (70%) and DenseNet121 (30%) using Weighted Soft Voting for high-reliability, low-false-positive results (reaching **95.59% Ensemble Validation Accuracy**).
- **Automated Brain Region Cropping:** Uses OpenCV contour detection to isolate the cerebrum and eliminate background noise/skull artifacts before inference.
- **Dual Native Resolutions:** Feeds $224 \times 224$ to DenseNet121 and native $299 \times 299$ to InceptionV3 for maximum feature extraction fidelity.
- **Explainable AI (Grad-CAM):** Generates heatmaps that highlight exactly which regions of the MRI the model focused on.
- **Automated Tumor Segmentation:** Produces a tumor region mask using adaptive morphological image processing driven by the Grad-CAM activation.
- **Tumor Property Analytics:** Estimates anatomical location (brain lobe), tumor area percentage, and clinical severity rating (including Borderline flagging).
- **Interactive Dashboard:** A dark-themed Gradio interface with clickable sample scans for instant testing.
- **REST API:** A FastAPI backend exposing the full inference pipeline as HTTP endpoints.
- **Docker Support:** A production-ready Dockerfile for containerized deployment.
- **Automated Test Suite:** 35 pytest tests covering unit logic, image preprocessing, and full integration pipeline.
- **High-Accuracy Training Notebook:** Includes `BrainTumor_98pct_Ensemble_Training.ipynb` for 1-click cloud training on Kaggle/Google Colab.

---

## Technology Stack

| Layer          | Technology                            |
|----------------|---------------------------------------|
| Deep Learning  | TensorFlow 2.21, Keras                |
| Models         | InceptionV3, DenseNet121, EfficientNet|
| Explainability | Grad-CAM                              |
| Vision         | OpenCV 4.x, NumPy                     |
| Web UI         | Gradio 6.x                            |
| REST API       | FastAPI, Uvicorn                      |
| Testing        | pytest (35 test cases)                |
| Container      | Docker (CPU-only, Python 3.12)        |
| Language       | Python 3.10+                          |

---

## Project Structure

```
Brain_Tumor_Project/
├── app.py                                   # Main entrypoint — launches the Gradio dashboard
├── Dockerfile                               # Production container definition (CPU-only)
├── .dockerignore                            # Excludes dev/build artifacts from Docker context
├── requirements.txt                         # Pinned Python dependencies
├── BrainTumor_98pct_Ensemble_Training.ipynb # 1-Click Kaggle/Colab Training Pipeline
├── brain_tumor_xai_paper.tex                # IEEE Research Paper LaTeX Source
│
├── src/                                     # Core application package
│   ├── __init__.py
│   ├── config.py                            # Centralized constants, paths, thresholds, logging
│   ├── inference.py                         # Dual-resolution ensemble prediction & Grad-CAM
│   ├── processor.py                         # Brain cropping, morphology segmentation & severity
│   ├── dashboard.py                         # Gradio UI layout, HTML formatting, CSS styling
│   └── api.py                               # FastAPI REST backend
│
├── scripts/                                 # Offline tooling scripts
│   ├── evaluate_models.py                   # Automated Model Performance & Confusion Matrix
│   ├── compute_detailed_metrics.py          # 4-decimal Precision/Recall/F1 calculator
│   ├── run_api.py                           # Launches the FastAPI server
│   ├── train_segmentation.py                # U-Net segmentation training script
│   └── evaluate_segmentation.py             # Dice/IoU scoring script
│
├── tests/                                   # Automated test suite (35 Tests)
│   ├── test_processor.py                    # 20 unit tests for cropping & segmentation logic
│   └── test_inference.py                    # 15 integration tests for full inference pipeline
│
├── models/                                  # Trained model weights (Phase 3 Full Fine-Tuning)
│   ├── inception_full_best.keras            # InceptionV3 weights (95.51% Accuracy)
│   ├── densenet_full_best.keras             # DenseNet121 weights (92.83% Accuracy)
│   └── effnet_full_best.keras               # EfficientNetV2S weights
│
├── datasets/                                # Dataset directory (image/ and mask/)
└── test_images/                             # Sample MRI scans for quick testing
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

### 4. Quick Execution Commands

You can run any command directly from your terminal:

**Launch Gradio App:**
```powershell
python app.py
```
Open in browser: `http://127.0.0.1:7860`

**Launch REST API:**
```powershell
python scripts/run_api.py
```
Open interactive Swagger documentation: `http://localhost:8000/docs`

**Evaluate Model Accuracy:**
```powershell
python scripts/evaluate_models.py --dataset_dir "datasets/image" --val_split 0.30
```

**Run Automated Tests:**
```powershell
pytest tests/ -v
```

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
# Run all 35 tests
pytest tests/ -v

# Run only unit tests (fast, no model loading)
pytest tests/test_processor.py -v

# Run only integration tests (loads models, ~15 seconds)
pytest tests/test_inference.py -v
```

Expected result: 35 passed (20 unit + 15 integration).

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

1. No Tumor (`0`)
2. Glioma Tumor (`1`)
3. Meningioma Tumor (`2`)
4. Pituitary Tumor (`3`)

The segmentation pipeline automatically extracts tumor boundaries using Grad-CAM guided mathematical morphology.

---

## How It Works

1. **Brain Cropping:** The raw MRI scan passes through `extract_brain_region` which uses OpenCV thresholding and contour bounding to eliminate dark backgrounds and skull artifacts.
2. **Dual-Resolution Feeding:** The cropped scan is resized to $224 \times 224$ for DenseNet121 and native $299 \times 299$ for InceptionV3.
3. **Weighted Soft-Voting:** Predictions are combined using optimal weights (70% InceptionV3 + 30% DenseNet121), achieving **95.59% accuracy**.
4. **Grad-CAM Localization:** Backpropagation is computed on DenseNet121's top convolutional layer to generate a spatial heatmap.
5. **Morphological Segmentation:** Otsu thresholding within the Grad-CAM region of interest extracts the tumor mask.
6. **Clinical Property Analytics:** Anatomical lobe location, tumor surface area percentage, and severity rating (with Borderline detection for confidence < 55%) are computed in real time.


---

## Medical Disclaimer

This software is for **educational and research purposes only**. It is not intended for clinical use and must not replace professional medical diagnosis. Always consult a qualified and licensed healthcare professional for any medical concerns.

---

**Author:** Bhola Yadav
**Project:** Final Year / Major Project — Explainable Brain Tumor Detection and Segmentation
