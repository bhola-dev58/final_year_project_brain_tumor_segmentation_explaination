import os
import time
import numpy as np

# Suppress TF and CUDA warnings BEFORE importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import cv2

from image_processing import create_segmentation, estimate_location, estimate_severity

import logging
logging.getLogger('absl').setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
tf.get_logger().setLevel('ERROR')

print("Loading Models...")

model_dense = tf.keras.models.load_model('models/densenet_best.h5', compile=False)
model_inc   = tf.keras.models.load_model('models/inception_best.h5', compile=False)

classes = ['No Tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']

# Cache layer for Grad-CAM
_last_conv_layer_name = None
for layer in reversed(model_dense.layers):
    try:
        if len(layer.output.shape) == 4:
            _last_conv_layer_name = layer.name
            break
    except Exception:
        continue

_model_output = model_dense.output[0] if isinstance(model_dense.output, list) else model_dense.output

_grad_model = tf.keras.Model(
    inputs=model_dense.inputs,
    outputs=[model_dense.get_layer(_last_conv_layer_name).output, _model_output]
)

print("Models loaded successfully.")

def make_gradcam_heatmap(img_array, pred_index=None):
    # Generates a heatmap indicating important regions
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
        raise ValueError("Gradients missing")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def predict_tumor_logic(img):
    # Main logic handler, returns all outputs in a dictionary
    if img is None:
        return {"is_valid": False, "error": "Please upload an MRI image first."}

    start_time = time.time()

    if img.dtype != np.uint8:
        img = np.uint8(np.clip(img, 0, 255))

    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

    # Ensemble prediction
    pred_dense = model_dense.predict(img_array, verbose=0)
    pred_inc   = model_inc.predict(img_array, verbose=0)
    avg_pred   = (pred_dense + pred_inc) / 2.0

    class_idx  = int(np.argmax(avg_pred))
    class_name = classes[class_idx]
    confidence = float(np.max(avg_pred)) * 100
    is_tumor   = class_idx != 0

    inference_time = time.time() - start_time

    gradcam_overlay = img.copy()
    segmentation_img = img.copy()
    tumor_percentage = 0.0
    location = "N/A"

    try:
        heatmap_raw = make_gradcam_heatmap(img_array)
        heatmap_resized = cv2.resize(heatmap_raw, (img.shape[1], img.shape[0]))

        # Kill weak activations so only the important focus area shows color
        heatmap_cleaned = np.where(heatmap_resized > 0.3, heatmap_resized, 0)

        heatmap_uint8 = np.uint8(255 * heatmap_cleaned)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Make zero-activation areas fully transparent (no blue tint)
        heatmap_color[heatmap_cleaned == 0] = [0, 0, 0]

        # Per-pixel alpha: hot areas are vivid, cool areas are transparent
        alpha = (heatmap_cleaned * 0.65)[:, :, np.newaxis]
        gradcam_overlay = np.uint8(img * (1 - alpha) + heatmap_color * alpha)

        if is_tumor:
            segmentation_img, _, tumor_percentage = create_segmentation(heatmap_raw, img)
            location = estimate_location(heatmap_raw, img.shape)

    except Exception as e:
        print(f"Visualization error: {e}")

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
        "classes": classes,
        "class_idx": class_idx
    }
