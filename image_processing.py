import cv2
import numpy as np


def create_segmentation(heatmap_raw, original_img):
    """
    Creates a segmentation mask using the provided classification heatmap.
    Uses intensity thresholding strictly within the heatmap's peak region.
    """
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
    # Only look at the top 30% of activations to avoid outer parts
    roi_mask = np.uint8(hm > 0.70) * 255
    
    # Clean up the ROI
    k_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, k_roi)
    
    # Keep only the largest blob in ROI to avoid scattered attention
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
        
    # Smooth the final mask edges nicely
    tumor_mask = cv2.GaussianBlur(tumor_mask, (7, 7), 0)
    _, tumor_mask = cv2.threshold(tumor_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Check if we found anything meaningful
    area_pct = (np.sum(tumor_mask > 0) / (h * w)) * 100
    if area_pct == 0 or area_pct > 25: # If too huge or zero, fallback to simple ROI
        tumor_mask = roi_mask
        area_pct = (np.sum(tumor_mask > 0) / (h * w)) * 100
        
    # 6. Apply soft transparent red fill (no borders)
    result = original_img.copy().astype(np.float32)
    red_layer = np.zeros_like(result)
    red_layer[:, :] = [220, 40, 40]  # Deep red color
    
    # 35% red opacity as requested (soft, no border)
    tumor_px = tumor_mask > 0
    result[tumor_px] = result[tumor_px] * 0.65 + red_layer[tumor_px] * 0.35
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result, tumor_mask, area_pct


def estimate_location(heatmap_raw, img_shape):
    h, w = img_shape[:2]
    hm = cv2.resize(heatmap_raw, (w, h))
    y_center, x_center = np.unravel_index(np.argmax(hm), hm.shape)

    v_pos = "Superior" if y_center < h / 2 else "Inferior"
    h_pos = "Left"     if x_center < w / 2 else "Right"

    if y_center < h * 0.4:
        lobe = "Frontal Lobe"
    elif y_center < h * 0.7:
        lobe = "Parietal Lobe"
    else:
        lobe = "Occipital Lobe"

    return f"{h_pos} {lobe} ({v_pos})"


def estimate_severity(confidence, tumor_percentage):
    if confidence > 95 and tumor_percentage > 5:
        return "High", "#ef4444"
    elif confidence > 80 and tumor_percentage > 3:
        return "Moderate", "#f59e0b"
    elif confidence > 60:
        return "Low", "#22c55e"
    else:
        return "Uncertain", "#6b7280"
