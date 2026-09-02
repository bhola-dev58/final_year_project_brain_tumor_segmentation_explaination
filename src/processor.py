from typing import Tuple, Union
import cv2
import numpy as np
from src.config import (
    COLOR_TUMOR_SEGMENT,
    SEGMENTATION_OPACITY,
    COLOR_SEVERITY_HIGH,
    COLOR_SEVERITY_MODERATE,
    COLOR_SEVERITY_LOW,
    COLOR_SEVERITY_BORDERLINE,
    COLOR_SEVERITY_UNCERTAIN,
    logger
)

def extract_brain_region(img: np.ndarray) -> np.ndarray:
    """
    Extracts the primary brain bounding region by cropping out dark background padding.
    If cropping is invalid or too small, returns original image.

    Args:
        img: Input image array (uint8).

    Returns:
        Cropped numpy array containing the focused brain region, or original image.
    """
    if img is None:
        return img
    try:
        h, w = img.shape[:2]
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        # Threshold background vs brain tissue
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest_contour)

        # Ensure cropped region is at least 25% of original dimensions
        if bw > w * 0.25 and bh > h * 0.25:
            pad_x = int(w * 0.02)
            pad_y = int(h * 0.02)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)
            return img[y1:y2, x1:x2]
        return img
    except Exception as e:
        logger.error(f"Error during brain region extraction: {e}")
        return img


def create_segmentation(
    heatmap_raw: np.ndarray,
    original_img: np.ndarray,
    brain_mask: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Creates a segmentation mask using the provided classification heatmap.
    Now brain-mask gated: no tumor pixels can appear outside brain boundary.

    Args:
        heatmap_raw: 2D numpy array representing the cleaned activation heatmap.
        original_img: 3D or 2D numpy array containing the original MRI scan.
        brain_mask: Optional binary mask (H, W, uint8) from _compute_brain_mask().
                    If None, falls back to threshold-based approach.

    Returns:
        Tuple containing:
            - Segmented image with semi-transparent overlay.
            - Binary mask of the segmented tumor (same spatial dimensions).
            - Percentage of brain area covered by the tumor.
    """
    try:
        h, w = original_img.shape[:2]

        # 1. Resize and normalize heatmap to full image resolution
        hm = cv2.resize(heatmap_raw, (w, h)).astype(np.float32)
        hm_max = hm.max()
        if hm_max > 1e-8:
            hm = hm / hm_max

        # 2. Get grayscale image
        if len(original_img.shape) == 3:
            gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = original_img.copy()

        # 3. Compute brain mask if not provided
        if brain_mask is None:
            _, brain_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_c = max(contours, key=cv2.contourArea)
                brain_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(brain_mask, [largest_c], -1, 255, cv2.FILLED)
            else:
                brain_mask = np.ones((h, w), dtype=np.uint8) * 255

        # Brain area in pixels (used for % calculation)
        brain_area_px = np.sum(brain_mask > 0)
        if brain_area_px == 0:
            brain_area_px = h * w

        # 4. ROI mask from top heatmap activations, constrained to brain interior
        # Use 70th percentile of brain-interior heatmap as threshold
        brain_hm_vals = hm[brain_mask > 0]
        if len(brain_hm_vals) == 0 or brain_hm_vals.max() == 0:
            return original_img.copy(), np.zeros((h, w), dtype=np.uint8), 0.0

        roi_threshold = np.percentile(brain_hm_vals, 70)  # top 30% of brain activations
        roi_threshold = max(roi_threshold, 0.45)          # never go below 0.45

        roi_mask = np.uint8(hm > roi_threshold) * 255
        roi_mask = np.uint8(roi_mask & brain_mask)        # strictly inside brain

        # Clean up ROI: morphological close to fill gaps
        k_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, k_roi)

        # Keep only the largest connected blob inside ROI
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return original_img.copy(), np.zeros((h, w), dtype=np.uint8), 0.0

        largest_roi = max(contours, key=cv2.contourArea)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(roi_mask, [largest_roi], -1, 255, cv2.FILLED)

        # 5. Precise tumor pixel detection using Otsu within ROI
        roi_pixels = gray[roi_mask > 0]
        if len(roi_pixels) < 50:
            return original_img.copy(), np.zeros((h, w), dtype=np.uint8), 0.0

        # Otsu threshold on ROI-interior gray pixels
        try:
            roi_pixels_u8 = roi_pixels.astype(np.uint8)
            otsu_thresh, _ = cv2.threshold(roi_pixels_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except Exception:
            otsu_thresh = float(np.mean(roi_pixels) + 0.5 * np.std(roi_pixels))

        # Tumor mask: bright pixels inside ROI AND inside brain
        tumor_mask = np.uint8((gray.astype(np.float32) > otsu_thresh) &
                               (roi_mask > 0) &
                               (brain_mask > 0)) * 255

        # 6. Morphological cleanup: remove noise, smooth edges
        k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, k_clean, iterations=1)
        tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, k_clean, iterations=3)

        # Keep largest connected tumor blob
        contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_tumor = max(contours, key=cv2.contourArea)
            tumor_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(tumor_mask, [largest_tumor], -1, 255, cv2.FILLED)

        # Smooth final mask edges
        tumor_mask = cv2.GaussianBlur(tumor_mask, (9, 9), sigmaX=2)
        _, tumor_mask = cv2.threshold(tumor_mask, 127, 255, cv2.THRESH_BINARY)

        # 7. Area percentage — calculated as % of BRAIN area (not whole image)
        tumor_px = np.sum(tumor_mask > 0)
        area_pct = (tumor_px / brain_area_px) * 100

        # Fallback: if tumor mask is too small or unrealistically large, use ROI
        if area_pct == 0 or area_pct > 30.0:
            tumor_mask = roi_mask
            tumor_px = np.sum(tumor_mask > 0)
            area_pct = (tumor_px / brain_area_px) * 100

        # 8. Apply semi-transparent red fill overlay
        result = original_img.copy().astype(np.float32)
        red_layer = np.zeros_like(result)
        red_layer[:, :] = COLOR_TUMOR_SEGMENT

        tumor_bool = tumor_mask > 0
        result[tumor_bool] = (
            result[tumor_bool] * (1.0 - SEGMENTATION_OPACITY) +
            red_layer[tumor_bool] * SEGMENTATION_OPACITY
        )
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result, tumor_mask, area_pct

    except Exception as e:
        logger.error(f"Error during tumor segmentation: {e}", exc_info=True)
        h, w = original_img.shape[:2]
        return original_img.copy(), np.zeros((h, w), dtype=np.uint8), 0.0


def estimate_location(
    heatmap_raw: np.ndarray,
    img_shape: Union[Tuple[int, int], Tuple[int, int, int]]
) -> str:
    """
    Estimates the location of the tumor based on the spatial coordinate
    of the peak Grad-CAM activation value.

    Args:
        heatmap_raw: Raw heatmap values.
        img_shape: Shape tuple of the original image (height, width, channels).

    Returns:
        String describing anatomical position (e.g., 'Left Frontal Lobe (Superior)').
    """
    try:
        h, w = img_shape[:2]
        hm = cv2.resize(heatmap_raw, (w, h))
        y_center, x_center = np.unravel_index(np.argmax(hm), hm.shape)

        v_pos = "Superior" if y_center < h / 2 else "Inferior"
        h_pos = "Left" if x_center < w / 2 else "Right"

        if y_center < h * 0.4:
            lobe = "Frontal Lobe"
        elif y_center < h * 0.7:
            lobe = "Parietal Lobe"
        else:
            lobe = "Occipital Lobe"

        return f"{h_pos} {lobe} ({v_pos})"
    except Exception as e:
        logger.error(f"Error estimating tumor location: {e}")
        return "Unknown Location"


def estimate_severity(
    confidence: float,
    tumor_percentage: float
) -> Tuple[str, str]:
    """
    Categorizes the tumor severity based on ensemble classification confidence
    and segmented tumor area ratio.

    Args:
        confidence: Prediction confidence percentage (0 to 100).
        tumor_percentage: Tumor area coverage percentage (0 to 100).

    Returns:
        Tuple of (Severity Category String, Hex Color Code).
    """
    if confidence < 55.0:
        return "Borderline", COLOR_SEVERITY_BORDERLINE
    elif confidence > 95 and tumor_percentage > 5:
        return "High", COLOR_SEVERITY_HIGH
    elif confidence > 80 and tumor_percentage > 3:
        return "Moderate", COLOR_SEVERITY_MODERATE
    elif confidence > 60:
        return "Low", COLOR_SEVERITY_LOW
    else:
        return "Uncertain", COLOR_SEVERITY_UNCERTAIN

