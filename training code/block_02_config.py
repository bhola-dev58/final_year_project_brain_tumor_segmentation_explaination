# ============================================================
# BLOCK 2: Dataset Paths & Global Config
# ============================================================
DATASET_PATH = '/kaggle/input/datasets/bholadev58/data-source/brain-tumor-2d-dataset/image'
BATCH_SIZE   = 32         # larger batch = more stable gradients
VAL_SPLIT    = 0.20       # 80/20 split gives more training data
SEED         = 42
EPOCHS_P1    = 20         # Phase 1: frozen base
EPOCHS_P2    = 30         # Phase 2: fine-tune top layers
NUM_CLASSES  = 4

print("Dataset path:", DATASET_PATH)
print("Classes:", sorted(os.listdir(DATASET_PATH)))
