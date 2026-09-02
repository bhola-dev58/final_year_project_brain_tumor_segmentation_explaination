# ==============================================================================
# BRAIN TUMOR 4-CLASS SOTA TRI-ENSEMBLE TRAINING PIPELINE (>= 98.79% TARGET)
# ==============================================================================
# Architecture:
#   1. ConvNeXtSmall  (CVPR 2022, Meta AI - 7x7 Depthwise, GELU, LayerNorm) -> Weight: 0.45
#   2. InceptionV3    (Native 299x299 Multi-Scale Receptive Fields)          -> Weight: 0.35
#   3. DenseNet121    (Native 224x224 Dense Feature Reuse and Grad-CAM)      -> Weight: 0.20
#
#   Head: Dual Pooling (GlobalAvgPool + GlobalMaxPool concatenated)
#         Captures both average and peak spatial tumor features.
#
# Sequential File and Kaggle Notebook Cell Mapping:
# ------------------------------------------------------------------------------
# Cell 01 -> block_01_install_imports.py      (Imports, GPU check, pip auxiliary packages)
# Cell 02 -> block_02_config.py               (Dataset path, Epochs P1=25/P2=35/P3=80, Batch=32)
# Cell 03 -> block_03_data_generators.py      (MRI-safe augmentation, Mixup alpha=0.2 pipeline)
# Cell 04 -> block_04_class_weights.py        (Balanced class weights calculation)
# Cell 05 -> block_05_model_builder.py        (Dual Pooling head, AMSGrad Adam, Callbacks)
# Cell 06 -> block_06_phase1_training.py      (Phase 1: Warmup frozen heads with Mixup)
# Cell 07 -> block_07_phase2_finetune.py      (Phase 2: Unfreeze top layers with Mixup)
# Cell 08 -> block_08_phase3_full_finetune.py (Phase 3: Full unfreeze, NO Mixup, LR=3e-6/5e-6)
# Cell 09 -> block_09_tta_evaluation.py       (10-Pass TTA, Weighted Tri-Ensemble, F1 Report)
# Cell 10 -> block_10_save_models.py          (Export .keras and .h5 models, Confusion Matrix)
#
# Safety / Recovery:
# Cell 11 -> block_recovery.py               (Optional: Instant recovery after session restart)
# ==============================================================================
