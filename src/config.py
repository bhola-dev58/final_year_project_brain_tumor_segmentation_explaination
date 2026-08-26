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
DENSENET_PATH = _get_model_path("densenet")
INCEPTION_PATH = _get_model_path("inception")
EFFNET_PATH = _get_model_path("effnet", extensions=[".keras"])


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

# Ensemble Weighting Settings (Optimal Dual-Ensemble: 70% Inception + 30% DenseNet)
DENSENET_VOTE_WEIGHT = 0.30
INCEPTION_VOTE_WEIGHT = 0.70
EFFNET_VOTE_WEIGHT = 0.00



# Color Palettes
COLOR_TUMOR_SEGMENT = [220, 40, 40]  # Deep Red (RGB)
COLOR_SEVERITY_HIGH = "#ef4444"
COLOR_SEVERITY_MODERATE = "#f59e0b"
COLOR_SEVERITY_LOW = "#22c55e"
COLOR_SEVERITY_BORDERLINE = "#f59e0b"
COLOR_SEVERITY_UNCERTAIN = "#6b7280"

