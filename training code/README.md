# ============================================================
# README — Training Code Folder Guide
# ============================================================

## Folder Structure

| File | Description |
|------|-------------|
| block_01_install_imports.py | Install libraries + all imports |
| block_02_config.py | Dataset path + global config |
| block_03_data_generators.py | 224x224 and 299x299 generators |
| block_04_class_weights.py | Balanced class weights |
| block_05_model_builder.py | Build DenseNet + InceptionV3 + EfficientNetV2S |
| block_06_phase1_training.py | Phase 1: frozen base training (20 epochs) |
| block_07_phase2_finetune.py | Phase 2: unfreeze top 60 layers (30 epochs) |
| block_08_tta_evaluation.py | TTA ensemble evaluation (FIXED) |
| block_09_save_models.py | Save models + confusion matrix (FIXED) |
| block_10_aggressive_finetune.py | Phase 3: full unfreeze for 98%+ |
| block_recovery.py | Use when Kaggle session restarts |

## Run Order for 98%+ Accuracy

### Fresh Run (Full Training)
1 → 2 → 3 → 4 → 5 → 6 → 7 → 10 → 8* → 9*

*In Block 8, change model names to: densenet_full_best.keras, inception_full_best.keras, effnet_full_best.keras

### After Session Restart
Run only: block_recovery.py (standalone, no dependencies)

## Important Notes

- EfficientNetV2S cannot be saved as .h5 — always saves as .keras
- InceptionV3 uses 299x299 input (native size)
- DenseNet121 and EfficientNetV2S use 224x224
- Kaggle GPU: ~8-9 hours total for full run (Blocks 1-10)
- Download from /kaggle/working/: densenet_best.h5, inception_best.h5, effnet_best.keras
