# ============================================================
# BLOCK 2: Dataset Paths and Global Configuration
# ============================================================
DATASET_PATH = '/kaggle/input/datasets/bholadev58/brain-tumor-mri-4-class-dataset/datasets/image'

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"[ERROR] Dataset directory not found at: {DATASET_PATH}")
else:
    print(f"[INFO] Located Dataset at: {DATASET_PATH}")

# Training Hyperparameters
BATCH_SIZE   = 32         # Stable gradient batch size
VAL_SPLIT    = 0.20       # 80/20 train and validation split
SEED         = 42
NUM_CLASSES  = 4

# Phase Epoch Budgets
EPOCHS_P1    = 25         # Phase 1: Warmup classifier head
EPOCHS_P2    = 35         # Phase 2: Top layers fine-tuning
EPOCHS_P3    = 80         # Phase 3: Full backbone fine-tuning (EarlyStopping prevents over-running)

if os.path.exists(DATASET_PATH):
    classes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    print("Detected Classes in Directory:", classes)
