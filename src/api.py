"""
FastAPI REST backend for BrainTumorXAI.

Endpoints:
    GET  /api/health   — Liveness check; confirms models are loaded.
    POST /api/predict  — Accepts an MRI image file and returns full
                         diagnosis JSON with base64-encoded visual overlays.

Usage (run independently from the Gradio app):
    python scripts/run_api.py
    # Then open:  http://localhost:8000/docs
"""
import base64
import io
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from src.config import logger
from src.inference import predict_tumor_logic

# ------------------------------------------------------------------ #
# Application factory
# ------------------------------------------------------------------ #
app = FastAPI(
    title="BrainTumorXAI API",
    description=(
        "REST API for the Explainable Brain Tumor Detection system. "
        "Provides ensemble DenseNet121 + InceptionV3 classification with "
        "Grad-CAM explainability and morphological segmentation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow cross-origin requests (useful when a separate frontend calls this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def _ndarray_to_base64(img: np.ndarray) -> str:
    """Encodes a numpy image array to a base64 PNG string for JSON transport."""
    # Convert BGR → RGB if needed (Gradio already works in RGB)
    pil_img = Image.fromarray(img.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


async def _read_upload_as_numpy(file: UploadFile) -> np.ndarray:
    """Reads an uploaded image file and converts it to a uint8 RGB numpy array."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=422,
            detail="Could not decode the uploaded file as a valid image. "
                   "Please upload a JPEG or PNG MRI scan.",
        )
    # Convert BGR → RGB to match what the models were trained on
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ #
# Routes
# ------------------------------------------------------------------ #
@app.get(
    "/api/health",
    summary="Health check",
    description="Returns server status and confirms the ensemble models are loaded.",
    tags=["System"],
)
def health_check() -> dict:
    """Liveness endpoint."""
    return {
        "status": "ok",
        "models_loaded": True,
        "ensemble": "DenseNet121 + InceptionV3",
        "version": "1.0.0",
    }


@app.post(
    "/api/predict",
    summary="Predict brain tumor from MRI",
    description=(
        "Upload a brain MRI image (JPEG/PNG). "
        "Returns classification result, confidence score, tumor properties, "
        "and base64-encoded Grad-CAM and segmentation overlays."
    ),
    tags=["Inference"],
)
async def predict(
    file: UploadFile = File(..., description="Brain MRI image file (JPEG or PNG)"),
) -> JSONResponse:
    """
    Performs full inference on the uploaded MRI scan.

    Returns:
        JSON response containing:
            - class_name (str): Detected class (e.g., 'Glioma Tumor').
            - confidence (float): Prediction confidence in percent.
            - is_tumor (bool): Whether a tumor was detected.
            - location (str): Estimated anatomical lobe location.
            - tumor_percentage (float): Tumor area as % of total scan.
            - severity (str): Severity category (High / Moderate / Low / Uncertain).
            - severity_color (str): Corresponding hex color for the severity.
            - inference_time (float): Time taken for inference in seconds.
            - gradcam_overlay_b64 (str): Base64-encoded Grad-CAM overlay PNG.
            - segmentation_b64 (str): Base64-encoded segmentation overlay PNG.
    """
    logger.info(f"Received prediction request: filename={file.filename}")

    img = await _read_upload_as_numpy(file)
    result = predict_tumor_logic(img)

    if not result.get("is_valid"):
        raise HTTPException(status_code=500, detail=result.get("error", "Inference failed."))

    # Encode visual outputs to base64 strings for JSON transport
    gradcam_b64 = _ndarray_to_base64(result["gradcam_overlay"])
    seg_b64 = _ndarray_to_base64(result["segmentation_img"])

    return JSONResponse(
        status_code=200,
        content={
            "class_name": result["class_name"],
            "confidence": round(result["confidence"], 4),
            "is_tumor": result["is_tumor"],
            "location": result["location"],
            "tumor_percentage": round(result["tumor_percentage"], 4),
            "severity": result["severity"],
            "severity_color": result["severity_color"],
            "inference_time": round(result["inference_time"], 4),
            "prediction_breakdown": {
                cls: round(float(prob) * 100, 4)
                for cls, prob in zip(result["classes"], result["avg_pred"])
            },
            "gradcam_overlay_b64": gradcam_b64,
            "segmentation_b64": seg_b64,
        },
    )
