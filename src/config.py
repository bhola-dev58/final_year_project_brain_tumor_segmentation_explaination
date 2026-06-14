import os
import logging

# Central Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BrainTumorXAI")

# Project Root and Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model Paths
DENSENET_PATH = os.path.join(MODELS_DIR, "densenet_best.h5")
INCEPTION_PATH = os.path.join(MODELS_DIR, "inception_best.h5")

# Classification Settings
CLASSES = ['No Tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
IMG_SIZE_CLASSIFY = (224, 224)
IMG_SIZE_TRAIN = (256, 256)

# Threshold & Activation Settings
GRADCAM_CLEAN_THRESHOLD = 0.30
GRADCAM_ROI_THRESHOLD = 0.70
GRADCAM_MAX_TUMOR_PCT_FALLBACK = 25.0
GRADCAM_OVERLAY_OPACITY = 0.65
SEGMENTATION_OPACITY = 0.35

# Color Palettes
COLOR_TUMOR_SEGMENT = [220, 40, 40]  # Deep Red (RGB)
COLOR_SEVERITY_HIGH = "#ef4444"
COLOR_SEVERITY_MODERATE = "#f59e0b"
COLOR_SEVERITY_LOW = "#22c55e"
COLOR_SEVERITY_UNCERTAIN = "#6b7280"
