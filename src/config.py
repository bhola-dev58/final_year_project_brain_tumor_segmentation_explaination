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

def _get_model_path(base_name, extensions=[".keras", ".h5"]):
    for ext in extensions:
        full_candidate = os.path.join(MODELS_DIR, f"{base_name}_full_best{ext}")
        if os.path.exists(full_candidate):
            return full_candidate
    for ext in extensions:
        candidate = os.path.join(MODELS_DIR, f"{base_name}_best{ext}")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(MODELS_DIR, f"{base_name}_best.h5")

# Model Paths (Auto-resolves Phase 3 _full_best.keras weights)
CONVNEXT_PATH = _get_model_path("convnext")
INCEPTION_PATH = _get_model_path("inception")
DENSENET_PATH = _get_model_path("densenet")
EFFNET_PATH = _get_model_path("effnet", extensions=[".keras"])

# Classification Settings
CLASSES = ['No Tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
IMG_SIZE_CLASSIFY = (224, 224)
IMG_SIZE_INCEPTION = (299, 299)
IMG_SIZE_TRAIN = (224, 224)

# Threshold & Activation Settings
GRADCAM_CLEAN_THRESHOLD = 0.30
GRADCAM_ROI_THRESHOLD = 0.70
GRADCAM_MAX_TUMOR_PCT_FALLBACK = 25.0
GRADCAM_OVERLAY_OPACITY = 0.65
SEGMENTATION_OPACITY = 0.35

# Tri-Ensemble Weighting Settings (ConvNeXtSmall: 45%, InceptionV3: 35%, DenseNet121: 20%)
CONVNEXT_VOTE_WEIGHT = 0.45
INCEPTION_VOTE_WEIGHT = 0.35
DENSENET_VOTE_WEIGHT = 0.20
EFFNET_VOTE_WEIGHT = 0.00

# Color Palettes
COLOR_TUMOR_SEGMENT = [220, 40, 40]  # Deep Red (RGB)
COLOR_SEVERITY_HIGH = "#ef4444"
COLOR_SEVERITY_MODERATE = "#f59e0b"
COLOR_SEVERITY_LOW = "#22c55e"
COLOR_SEVERITY_BORDERLINE = "#f59e0b"
COLOR_SEVERITY_UNCERTAIN = "#6b7280"

