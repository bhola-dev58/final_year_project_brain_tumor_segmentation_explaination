from typing import Tuple, Union
import cv2
import numpy as np
from src.config import (
    GRADCAM_ROI_THRESHOLD,
    GRADCAM_MAX_TUMOR_PCT_FALLBACK,
    COLOR_TUMOR_SEGMENT,
    SEGMENTATION_OPACITY,
    COLOR_SEVERITY_HIGH,
    COLOR_SEVERITY_MODERATE,
    COLOR_SEVERITY_LOW,
    COLOR_SEVERITY_UNCERTAIN,
    logger
)

def create_segmentation(
    heatmap_raw: np.ndarray,
    original_img: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Creates a segmentation mask using the provided classification heatmap.
    Uses intensity thresholding strictly within the heatmap's peak region.

    Args:
        heatmap_raw: 2D numpy array representing the raw activation heatmap.
        original_img: 3D or 2D numpy array containing the original MRI scan.

    Returns:
        Tuple containing:
            - Segmented image with semi-transparent overlay.
            - Binary mask of the segmented tumor (same spatial dimensions).
            - Percentage of total scan area covered by the tumor.
    """
    try:
        h, w = original_img.shape[:2]
        
        # 1. Resize and normalize heatmap
        hm = cv2.resize(heatmap_raw, (w, h)).astype(np.float32)
        hm_max = hm.max()
        if hm_max > 0:
            hm = hm / hm_max

        # 2. Get grayscale image
        if len(original_img.shape) == 3:
            gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = original_img.copy()
            
        # 3. Create a region of interest (ROI) from the most confident heatmap area
        # Only look at the top activations to avoid outer brain parts/scaffolding
        roi_mask = np.uint8(hm > GRADCAM_ROI_THRESHOLD) * 255
        
        # Clean up the ROI
        k_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, k_roi)
        
        # Keep only the largest blob in ROI to avoid scattered attention points
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return original_img.copy(), np.zeros((h, w), dtype=np.uint8), 0.0
            
        largest_roi = max(contours, key=cv2.contourArea)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(roi_mask, [largest_roi], -1, 255, cv2.FILLED)

        # 4. Extract pixels inside ROI and apply precise intensity thresholding
        roi_pixels = gray[roi_mask > 0]
        if len(roi_pixels) < 50:
            return original_img.copy(), np.zeros((h, w), dtype=np.uint8), 0.0
            
        # Tumors are generally the brightest part in this focused ROI
        try:
            otsu_thresh, _ = cv2.threshold(roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        except Exception:
            otsu_thresh = np.mean(roi_pixels) + np.std(roi_pixels)
            
        # Create the actual tumor mask
        tumor_mask = np.uint8((gray > otsu_thresh) & (roi_mask > 0)) * 255
        
        # 5. Clean up the tumor mask (remove noise, smooth edges)
        k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, k_clean, iterations=1)
        tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, k_clean, iterations=2)
        
        # Keep largest tumor blob inside the ROI
        contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_tumor = max(contours, key=cv2.contourArea)
            tumor_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(tumor_mask, [largest_tumor], -1, 255, cv2.FILLED)
            
        # Smooth the final mask edges nicely using Gaussian Blur
        tumor_mask = cv2.GaussianBlur(tumor_mask, (7, 7), 0)
        _, tumor_mask = cv2.threshold(tumor_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Check if we found anything meaningful
        area_pct = (np.sum(tumor_mask > 0) / (h * w)) * 100
        if area_pct == 0 or area_pct > GRADCAM_MAX_TUMOR_PCT_FALLBACK: # If too huge or zero, fallback to simple ROI
            tumor_mask = roi_mask
            area_pct = (np.sum(tumor_mask > 0) / (h * w)) * 100
            
        # 6. Apply soft transparent red fill (no borders)
        result = original_img.copy().astype(np.float32)
        red_layer = np.zeros_like(result)
        red_layer[:, :] = COLOR_TUMOR_SEGMENT
        
        # Semi-transparent opacity overlay (soft, no border)
        tumor_px = tumor_mask > 0
        opacity_base = 1.0 - SEGMENTATION_OPACITY
        result[tumor_px] = result[tumor_px] * opacity_base + red_layer[tumor_px] * SEGMENTATION_OPACITY
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result, tumor_mask, area_pct

    except Exception as e:
        logger.error(f"Error during tumor segmentation: {e}", exc_info=True)
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
    if confidence > 95 and tumor_percentage > 5:
        return "High", COLOR_SEVERITY_HIGH
    elif confidence > 80 and tumor_percentage > 3:
        return "Moderate", COLOR_SEVERITY_MODERATE
    elif confidence > 60:
        return "Low", COLOR_SEVERITY_LOW
    else:
        return "Uncertain", COLOR_SEVERITY_UNCERTAIN
