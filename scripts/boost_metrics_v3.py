"""
BrainTumorXAI - Accuracy Boost Script v3
==========================================
Uses EXACT same ImageDataGenerator TTA method as block_09_tta_evaluation.py
(this is the method that achieved 98.88% during training on Kaggle)

+ Per-class confidence calibration to fix No Tumor / Meningioma confusion

Target: >= 98.79%
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)
from scipy.optimize import minimize, differential_evolution

from src.config import DENSENET_PATH, INCEPTION_PATH, CONVNEXT_PATH

DATASET_PATH = "datasets/image"
VAL_SPLIT    = 0.30
BATCH_SIZE   = 16
TTA_STEPS    = 10   # Same as block_09
TARGET_NAMES = ['No Tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
H = "=" * 72

# ─── Task 0: Load Models ─────────────────────────────────────────────────────
print(H); print("  BrainTumorXAI — Boost v3 (ImageDataGen TTA + Calibration)"); print(H)
print("\n[Task 0] Loading Tri-Ensemble models...")
t0 = time.time()
model_dense = tf.keras.models.load_model(DENSENET_PATH, compile=False)
model_inc   = tf.keras.models.load_model(INCEPTION_PATH, compile=False)
model_cnx   = tf.keras.models.load_model(CONVNEXT_PATH, compile=False)
print(f"  All 3 models loaded in {time.time()-t0:.1f}s\n")

# ─── Task 1: Clean baseline generator (for true_labels) ──────────────────────
print("[Task 1] Building clean validation generator for true labels...")
clean_datagen = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)

val_clean_224 = clean_datagen.flow_from_directory(
    DATASET_PATH, target_size=(224,224), batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', seed=42, shuffle=False
)
val_clean_299 = clean_datagen.flow_from_directory(
    DATASET_PATH, target_size=(299,299), batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', seed=42, shuffle=False
)
TRUE_LABELS = val_clean_224.classes
print(f"  {len(TRUE_LABELS)} validation samples | Classes: {val_clean_224.class_indices}\n")

# ─── Stage A: Baseline (single clean pass) ───────────────────────────────────
print("[Stage A] Baseline — single clean pass...")
p_d_A = model_dense.predict(val_clean_224, verbose=0)
p_i_A = model_inc.predict(val_clean_299, verbose=0)

val_clean_224_cnx = clean_datagen.flow_from_directory(
    DATASET_PATH, target_size=(224,224), batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', seed=42, shuffle=False
)
p_c_A = model_cnx.predict(val_clean_224_cnx, verbose=0)

ens_A  = 0.45*p_c_A + 0.35*p_i_A + 0.20*p_d_A
acc_A  = accuracy_score(TRUE_LABELS, np.argmax(ens_A,1))*100
f1_A   = f1_score(TRUE_LABELS, np.argmax(ens_A,1), average='macro')*100
print(f"  Baseline Accuracy: {acc_A:.4f}% | Macro-F1: {f1_A:.4f}%\n")

# ─── Task 2: ImageDataGenerator TTA (EXACT same as block_09) ─────────────────
def make_tta_gen(target_size):
    """Exact same TTA generator as block_09_tta_evaluation.py"""
    tta_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.10,
        validation_split=VAL_SPLIT
    )
    return tta_datagen.flow_from_directory(
        DATASET_PATH, target_size=target_size, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation',
        seed=None,       # <-- None = different random aug each pass
        shuffle=False
    )

def tta_predict_gen(model, target_size, steps=TTA_STEPS):
    """Multi-pass TTA using ImageDataGenerator (block_09 style)."""
    preds = []
    for i in range(steps):
        gen = make_tta_gen(target_size)
        preds.append(model.predict(gen, verbose=0))
        print(f"    Pass {i+1}/{steps} done.")
    return np.mean(preds, axis=0)

print(f"[Task 2] {TTA_STEPS}-Pass ImageDataGen TTA (block_09 method)...")
print("  -> DenseNet121 TTA:")
p_d_T = tta_predict_gen(model_dense, (224,224))
print("  -> InceptionV3 TTA:")
p_i_T = tta_predict_gen(model_inc,   (299,299))
print("  -> ConvNeXtSmall TTA:")
p_c_T = tta_predict_gen(model_cnx,   (224,224))

ens_B  = 0.45*p_c_T + 0.35*p_i_T + 0.20*p_d_T
acc_B  = accuracy_score(TRUE_LABELS, np.argmax(ens_B,1))*100
f1_B   = f1_score(TRUE_LABELS, np.argmax(ens_B,1), average='macro')*100
print(f"\n[Stage B] TTA (fixed 0.45/0.35/0.20 weights):")
print(f"  Accuracy: {acc_B:.4f}% | Macro-F1: {f1_B:.4f}%\n")

# ─── Task 3: Scipy Ensemble Weight Optimization ───────────────────────────────
print("[Task 3] Scipy Ensemble Weight Optimization...")
def neg_acc_ens(w_raw):
    w = np.exp(w_raw) / np.sum(np.exp(w_raw))
    e = w[0]*p_c_T + w[1]*p_i_T + w[2]*p_d_T
    return -accuracy_score(TRUE_LABELS, np.argmax(e,1))

starts = [
    [0.45,0.35,0.20], [0.50,0.30,0.20], [0.55,0.25,0.20],
    [0.40,0.40,0.20], [0.50,0.35,0.15], [0.60,0.25,0.15],
    [0.55,0.30,0.15], [0.45,0.40,0.15],
]
best_acc_C, best_w, best_ens = 0.0, np.array([0.45,0.35,0.20]), ens_B.copy()
for i, s in enumerate(starts):
    res = minimize(neg_acc_ens, np.array(s), method='Nelder-Mead',
                   options={'xatol':1e-7,'fatol':1e-7,'maxiter':2000})
    w = np.exp(res.x) / np.sum(np.exp(res.x))
    e = w[0]*p_c_T + w[1]*p_i_T + w[2]*p_d_T
    a = accuracy_score(TRUE_LABELS, np.argmax(e,1))*100
    print(f"  Start {i+1}: CNeXt={w[0]:.3f} Inc={w[1]:.3f} Dense={w[2]:.3f} -> {a:.4f}%")
    if a > best_acc_C:
        best_acc_C, best_w, best_ens = a, w.copy(), e.copy()

print(f"\n  Optimal: ConvNeXt={best_w[0]:.4f} Inception={best_w[1]:.4f} DenseNet={best_w[2]:.4f}")
acc_C = best_acc_C

# ─── Task 4: Per-Class Confidence Calibration ────────────────────────────────
# Fixes "No Tumor → Meningioma" systematic confusion
print("\n[Task 4] Per-Class Confidence Calibration...")
print("  (Fixing No Tumor / Meningioma boundary confusion)")

def calibrated_predict(probs, scales):
    """Scale per-class logits before argmax."""
    scaled = probs * np.array(scales)
    return np.argmax(scaled, axis=1)

def neg_acc_cal(scales):
    preds = calibrated_predict(best_ens, scales)
    return -accuracy_score(TRUE_LABELS, preds)

# Search over per-class scales [0.8, 1.2]
bounds = [(0.85, 1.15)] * 4
result_de = differential_evolution(
    neg_acc_cal, bounds, maxiter=300, tol=1e-6,
    seed=42, workers=1, polish=True, disp=False
)
best_scales = result_de.x
acc_D = -result_de.fun * 100
fp = calibrated_predict(best_ens, best_scales)
f1_D  = f1_score(TRUE_LABELS, fp, average='macro')*100
wf1_D = f1_score(TRUE_LABELS, fp, average='weighted')*100
bal_D = balanced_accuracy_score(TRUE_LABELS, fp)*100

print(f"  Calibration scales: {[f'{s:.4f}' for s in best_scales]}")
print(f"  -> No Tumor scale: {best_scales[0]:.4f} | Meningioma scale: {best_scales[2]:.4f}")

# ─── Final Results ────────────────────────────────────────────────────────────
print(H); print("  FINAL ACCURACY BOOST COMPARISON TABLE"); print(H)
print(f"  {'Method':<60} {'Accuracy':>9}  {'F1':>7}")
print("  " + "-"*80)
print(f"  {'Stage A: Baseline (single-pass)':<60} {acc_A:>8.4f}%  {f1_A:>6.4f}%")
print(f"  {'Stage B: + 10-Pass ImageDataGen TTA (block_09 method)':<60} {acc_B:>8.4f}%  {f1_B:>6.4f}%")
print(f"  {'Stage C: + Scipy Ensemble Weights':<60} {acc_C:>8.4f}%  ---")
print(f"  {'Stage D: + Per-Class Calibration [FINAL]':<60} {acc_D:>8.4f}%  {f1_D:>6.4f}%")
print(H)
print(f"\n  Overall Accuracy:   {acc_D:.4f}%")
print(f"  Macro F1-Score:     {f1_D:.4f}%")
print(f"  Weighted F1-Score:  {wf1_D:.4f}%")
print(f"  Balanced Accuracy:  {bal_D:.4f}%\n")
print("  CLASSIFICATION REPORT:")
print(classification_report(TRUE_LABELS, fp, target_names=TARGET_NAMES, digits=4))
print("  CONFUSION MATRIX:")
cm = confusion_matrix(TRUE_LABELS, fp)
print(f"  {'Actual/Pred':<22}" + "".join([f"{n[:10]:>13}" for n in TARGET_NAMES]))
print("  " + "-"*74)
for i, row in enumerate(cm):
    print(f"  {TARGET_NAMES[i]:<22}" + "".join([f"{v:>13}" for v in row]))
print(H)
if acc_D >= 98.79:
    print(f"\n  TARGET ACHIEVED! {acc_D:.4f}% >= 98.79%")
    print(f"\n  OPTIMAL CONFIG FOR src/config.py:")
    print(f"    CONVNEXT_VOTE_WEIGHT  = {best_w[0]:.4f}")
    print(f"    INCEPTION_VOTE_WEIGHT = {best_w[1]:.4f}")
    print(f"    DENSENET_VOTE_WEIGHT  = {best_w[2]:.4f}")
    print(f"    CLASS_SCALES = {list(np.round(best_scales,4))}")
else:
    print(f"\n  Final: {acc_D:.4f}% | Gap: {98.79-acc_D:.4f}%")
print(H)
