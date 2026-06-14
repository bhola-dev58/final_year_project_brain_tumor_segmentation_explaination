import os
import time
from typing import Dict, Any, Optional
import numpy as np

# Suppress TF and CUDA warnings BEFORE importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import cv2

# Suppress deep library warnings
import logging
logging.getLogger('absl').setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
tf.get_logger().setLevel('ERROR')

from src.config import (
    DENSENET_PATH,
    INCEPTION_PATH,
    CLASSES,
    IMG_SIZE_CLASSIFY,
    GRADCAM_CLEAN_THRESHOLD,
    GRADCAM_OVERLAY_OPACITY,
    logger
)
from src.processor import create_segmentation, estimate_location, estimate_severity

# Module-level model initialization
logger.info("Initializing ensemble models...")
try:
    model_dense = tf.keras.models.load_model(DENSENET_PATH, compile=False)
    model_inc = tf.keras.models.load_model(INCEPTION_PATH, compile=False)
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load weight files from models directory: {e}", exc_info=True)
    raise e

# Dynamic Grad-CAM sub-model construction using the last 4D conv layer of DenseNet121
_last_conv_layer_name = None
for layer in reversed(model_dense.layers):
    try:
        if len(layer.output.shape) == 4:
            _last_conv_layer_name = layer.name
            break
    except Exception:
        continue

if _last_conv_layer_name is None:
    raise RuntimeError("Could not locate a suitable 4D convolutional layer in DenseNet121.")

_model_output = model_dense.output[0] if isinstance(model_dense.output, list) else model_dense.output

_grad_model = tf.keras.Model(
    inputs=model_dense.inputs,
    outputs=[model_dense.get_layer(_last_conv_layer_name).output, _model_output]
)


def make_gradcam_heatmap(
    img_array: np.ndarray,
    pred_index: Optional[int] = None
) -> np.ndarray:
    """
    Generates a Grad-CAM heatmap indicating important regions of focus in DenseNet121.

    Args:
        img_array: Preprocessed image tensor array of shape (1, H, W, C).
        pred_index: The class index to run Grad-CAM against. If None, uses class with highest prediction.

    Returns:
        2D normalized numpy array representing the heatmap activations.
    """
    img_tensor = tf.constant(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, preds = _grad_model(img_tensor)
        tape.watch(conv_outputs)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise ValueError("Gradients are missing during Grad-CAM backpropagation.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def predict_tumor_logic(img: Optional[np.ndarray]) -> Dict[str, Any]:
    """
    Inference handler representing classification, heatmap generation,
    segmentation, and metadata analysis.

    Args:
        img: Input image array read by OpenCV or Gradio.

    Returns:
        Dictionary containing validation state, original image, overlay outputs,
        classification labels, confidence level, severity estimation, location, etc.
    """
    if img is None:
        return {"is_valid": False, "error": "Please upload an MRI image first."}

    start_time = time.time()

    try:
        if img.dtype != np.uint8:
            img = np.uint8(np.clip(img, 0, 255))

        # Prep image for models
        img_resized = cv2.resize(img, IMG_SIZE_CLASSIFY)
        img_array = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

        # Perform Ensemble Classification (Soft Voting)
        pred_dense = model_dense.predict(img_array, verbose=0)
        pred_inc = model_inc.predict(img_array, verbose=0)
        avg_pred = (pred_dense + pred_inc) / 2.0

        class_idx = int(np.argmax(avg_pred))
        class_name = CLASSES[class_idx]
        confidence = float(np.max(avg_pred)) * 100
        is_tumor = class_idx != 0

        inference_time = time.time() - start_time

        gradcam_overlay = img.copy()
        segmentation_img = img.copy()
        tumor_percentage = 0.0
        location = "N/A"

        try:
            # Generate explainability overlays
            heatmap_raw = make_gradcam_heatmap(img_array)
            heatmap_resized = cv2.resize(heatmap_raw, (img.shape[1], img.shape[0]))

            # Discard weak background activations
            heatmap_cleaned = np.where(heatmap_resized > GRADCAM_CLEAN_THRESHOLD, heatmap_resized, 0)

            heatmap_uint8 = np.uint8(255 * heatmap_cleaned)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # Eliminate non-activated areas (avoid blue tint overlaying normal regions)
            heatmap_color[heatmap_cleaned == 0] = [0, 0, 0]

            # Blend with dynamic transparency (alpha mapping)
            alpha = (heatmap_cleaned * GRADCAM_OVERLAY_OPACITY)[:, :, np.newaxis]
            gradcam_overlay = np.uint8(img * (1.0 - alpha) + heatmap_color * alpha)

            if is_tumor:
                segmentation_img, _, tumor_percentage = create_segmentation(heatmap_raw, img)
                location = estimate_location(heatmap_raw, img.shape)

        except Exception as ex:
            logger.error(f"Failed to generate visualization overlays: {ex}", exc_info=True)

        severity, severity_color = estimate_severity(confidence, tumor_percentage)

        return {
            "is_valid": True,
            "img": img,
            "segmentation_img": segmentation_img,
            "gradcam_overlay": gradcam_overlay,
            "class_name": class_name,
            "confidence": confidence,
            "is_tumor": is_tumor,
            "inference_time": inference_time,
            "tumor_percentage": tumor_percentage,
            "location": location,
            "severity": severity,
            "severity_color": severity_color,
            "avg_pred": avg_pred[0],
            "classes": CLASSES,
            "class_idx": class_idx
        }

    except Exception as e:
        logger.error(f"Inference error in prediction pipeline: {e}", exc_info=True)
        return {"is_valid": False, "error": f"An internal server error occurred: {e}"}
