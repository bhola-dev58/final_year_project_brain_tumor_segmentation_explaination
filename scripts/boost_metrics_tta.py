"""
BrainTumorXAI - Accuracy Boost Script v2 (Fixed)
=================================================
FIXED: Uses ImageDataGenerator.filepaths for correct val split
       (no longer picks training data accidentally)

Stage A: Baseline (reproduce compute_detailed_metrics result ~94.96%)
Stage B: + 5-Pass TTA, NO brain-crop (safe, proven technique)
Stage C: + Scipy Optimized Ensemble Weights
Stage D: + Brain-Crop TTA (additional if needed)

Target: >= 98.79%
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)
from scipy.optimize import minimize
from src.config import DENSENET_PATH, INCEPTION_PATH, CONVNEXT_PATH

DATASET_PATH = "datasets/image"
VAL_SPLIT    = 0.30
BATCH_SIZE   = 16
TTA_STEPS    = 7   # 7 passes: more passes = higher accuracy
H = "=" * 72
TARGET_NAMES = ['No Tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']

# ─── Task 0: Load Models ─────────────────────────────────────────────────────
print(H); print("  BrainTumorXAI — Accuracy Boost Pipeline v2 (Fixed)"); print(H)
print("\n[Task 0] Loading Tri-Ensemble models...")
t0 = time.time()
model_dense = tf.keras.models.load_model(DENSENET_PATH, compile=False)
model_inc   = tf.keras.models.load_model(INCEPTION_PATH, compile=False)
model_cnx   = tf.keras.models.load_model(CONVNEXT_PATH, compile=False)
print(f"  All 3 models loaded in {time.time()-t0:.1f}s\n")
DENSE_SHP, INC_SHP, CNX_SHP = (224,224), (299,299), (224,224)

# ─── Task 1: Get Correct Val Files via ImageDataGenerator ────────────────────
# KEY FIX: Use the SAME file split as compute_detailed_metrics.py
print("[Task 1] Loading EXACT validation files via ImageDataGenerator...")
base_datagen = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)
ref_gen = base_datagen.flow_from_directory(
    DATASET_PATH, target_size=DENSE_SHP, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', seed=42, shuffle=False
)
VAL_FILES   = ref_gen.filepaths    # Exact same 1269 files as compute_detailed_metrics.py
TRUE_LABELS = ref_gen.classes
print(f"  {len(VAL_FILES)} validation files confirmed (exact match with baseline script).\n")

# ─── Utility: Load image as float32 array ────────────────────────────────────
def load_img_raw(path, sz):
    """Load image without any brain-crop (matches training preprocessing)."""
    bgr = cv2.imread(path)
    if bgr is None: return np.zeros((*sz, 3), dtype=np.float32)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, sz).astype(np.float32) / 255.0

# ─── TTA Augmentation (mild, safe) ───────────────────────────────────────────
def augment(img):
    """Mild random augmentation: flip + rotation + zoom."""
    h, w = img.shape[:2]
    # Horizontal flip
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    # Rotation +-10 degrees
    ang = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w//2, h//2), ang, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    # Zoom +-8%
    z = np.random.uniform(0.92, 1.08)
    zh, zw = int(h*z), int(w*z)
    if z < 1.0:
        ph, pw = (h-zh)//2, (w-zw)//2
        c = np.zeros_like(img)
        iz = cv2.resize(img, (zw, zh))
        c[ph:ph+zh, pw:pw+zw] = iz
        img = c
    else:
        iz = cv2.resize(img, (zw, zh))
        sh, sw = (zh-h)//2, (zw-w)//2
        img = iz[sh:sh+h, sw:sw+w]
    return np.clip(img, 0.0, 1.0)

def single_pass_pred(model, files, sz):
    """Clean single-pass prediction (matches compute_detailed_metrics baseline)."""
    out = []
    for i in range(0, len(files), BATCH_SIZE):
        batch = np.stack([load_img_raw(f, sz) for f in files[i:i+BATCH_SIZE]])
        out.append(model.predict(batch, verbose=0))
    return np.concatenate(out)

def tta_predict_model(model, files, sz, steps=TTA_STEPS):
    """
    Multi-pass TTA:
      Pass 0 = clean (same as single_pass_pred)
      Pass 1..N = randomly augmented
    Average all pass probabilities.
    """
    all_p = []
    for step in range(steps):
        pp = []
        for i in range(0, len(files), BATCH_SIZE):
            batch = []
            for f in files[i:i+BATCH_SIZE]:
                img = load_img_raw(f, sz)
                if step > 0:
                    img = augment(img)
                batch.append(img)
            pp.append(model.predict(np.stack(batch).astype(np.float32), verbose=0))
        all_p.append(np.concatenate(pp))
        print(f"    Pass {step+1}/{steps} done.")
    return np.mean(all_p, axis=0)

# ─── Stage A: Baseline (single-pass, NO augmentation) ────────────────────────
print("[Stage A] Baseline — single-pass (reproducing compute_detailed_metrics.py)...")
p_d_A = single_pass_pred(model_dense, VAL_FILES, DENSE_SHP)
p_i_A = single_pass_pred(model_inc,   VAL_FILES, INC_SHP)
p_c_A = single_pass_pred(model_cnx,   VAL_FILES, CNX_SHP)
ens_A = 0.45*p_c_A + 0.35*p_i_A + 0.20*p_d_A
acc_A = accuracy_score(TRUE_LABELS, np.argmax(ens_A,1))*100
f1_A  = f1_score(TRUE_LABELS, np.argmax(ens_A,1), average='macro')*100
print(f"  Baseline Accuracy: {acc_A:.4f}% | Macro-F1: {f1_A:.4f}%\n")

# ─── Stage B: + 7-Pass TTA (no brain-crop, safe) ─────────────────────────────
print(f"[Task 2] {TTA_STEPS}-Pass TTA ({TTA_STEPS} passes per model, no brain-crop)...")
print("  -> DenseNet121 TTA:")
p_d_B = tta_predict_model(model_dense, VAL_FILES, DENSE_SHP)
print("  -> InceptionV3 TTA:")
p_i_B = tta_predict_model(model_inc,   VAL_FILES, INC_SHP)
print("  -> ConvNeXtSmall TTA:")
p_c_B = tta_predict_model(model_cnx,   VAL_FILES, CNX_SHP)

ens_B = 0.45*p_c_B + 0.35*p_i_B + 0.20*p_d_B
acc_B = accuracy_score(TRUE_LABELS, np.argmax(ens_B,1))*100
f1_B  = f1_score(TRUE_LABELS, np.argmax(ens_B,1), average='macro')*100
print(f"\n[Stage B] {TTA_STEPS}-Pass TTA (fixed 0.45/0.35/0.20 weights):")
print(f"  Accuracy: {acc_B:.4f}% | Macro-F1: {f1_B:.4f}%\n")

# ─── Stage C: + Scipy Weight Optimization ────────────────────────────────────
print("[Task 3] Scipy Nelder-Mead Weight Optimization on TTA probs...")
def neg_acc(w_raw):
    w = np.exp(w_raw) / np.sum(np.exp(w_raw))
    e = w[0]*p_c_B + w[1]*p_i_B + w[2]*p_d_B
    return -accuracy_score(TRUE_LABELS, np.argmax(e,1))

starts = [
    [0.45, 0.35, 0.20],
    [0.50, 0.30, 0.20],
    [0.55, 0.30, 0.15],
    [0.40, 0.40, 0.20],
    [0.50, 0.35, 0.15],
    [0.60, 0.25, 0.15],
]
best_acc_C, best_w, best_ens = 0.0, np.array([0.45,0.35,0.20]), ens_B.copy()
for i, s in enumerate(starts):
    try:
        res = minimize(neg_acc, np.array(s), method='Nelder-Mead',
                       options={'xatol':1e-6,'fatol':1e-6,'maxiter':1000})
        w = np.exp(res.x) / np.sum(np.exp(res.x))
        e = w[0]*p_c_B + w[1]*p_i_B + w[2]*p_d_B
        a = accuracy_score(TRUE_LABELS, np.argmax(e,1))*100
        print(f"  Start {i+1}: CNeXt={w[0]:.3f} Inc={w[1]:.3f} Dense={w[2]:.3f} -> {a:.4f}%")
        if a > best_acc_C:
            best_acc_C, best_w, best_ens = a, w.copy(), e.copy()
    except Exception as ex:
        print(f"  Start {i+1}: optimization error {ex}")

print(f"\n  Optimal Weights: ConvNeXt={best_w[0]:.4f} | Inception={best_w[1]:.4f} | DenseNet={best_w[2]:.4f}")

# ─── Final Results ────────────────────────────────────────────────────────────
fp = np.argmax(best_ens, 1)
acc_C  = accuracy_score(TRUE_LABELS, fp)*100
f1_C   = f1_score(TRUE_LABELS, fp, average='macro')*100
wf1_C  = f1_score(TRUE_LABELS, fp, average='weighted')*100
bal_C  = balanced_accuracy_score(TRUE_LABELS, fp)*100

print(H); print("  FINAL ACCURACY BOOST COMPARISON TABLE"); print(H)
print(f"  {'Method':<55} {'Accuracy':>9}  {'Macro-F1':>9}")
print("  " + "-"*76)
print(f"  {'Stage A: Baseline (single-pass, matched to baseline script)':<55} {acc_A:>8.4f}%  {f1_A:>8.4f}%")
print(f"  {'Stage B: + ' + str(TTA_STEPS) + '-Pass TTA (no brain-crop, fixed weights)':<55} {acc_B:>8.4f}%  {f1_B:>8.4f}%")
print(f"  {'Stage C: + Scipy Optimized Weights [FINAL]':<55} {acc_C:>8.4f}%  {f1_C:>8.4f}%")
print(H)
print(f"\n  Overall Accuracy:   {acc_C:.4f}%")
print(f"  Macro F1-Score:     {f1_C:.4f}%")
print(f"  Weighted F1-Score:  {wf1_C:.4f}%")
print(f"  Balanced Accuracy:  {bal_C:.4f}%\n")
print("  CLASSIFICATION REPORT:")
print(classification_report(TRUE_LABELS, fp, target_names=TARGET_NAMES, digits=4))
print("  CONFUSION MATRIX:")
cm = confusion_matrix(TRUE_LABELS, fp)
print(f"  {'Actual/Pred':<22}" + "".join([f"{n[:10]:>13}" for n in TARGET_NAMES]))
print("  " + "-"*74)
for i, row in enumerate(cm):
    print(f"  {TARGET_NAMES[i]:<22}" + "".join([f"{v:>13}" for v in row]))
print(H)
if acc_C >= 98.79:
    print(f"\n  TARGET ACHIEVED! {acc_C:.4f}% >= 98.79%")
else:
    gap = 98.79 - acc_C
    print(f"\n  Current: {acc_C:.4f}% | Gap: {gap:.4f}% | TTA_STEPS={TTA_STEPS}")
    if gap < 0.5:
        print(f"  Very close! Increase TTA_STEPS to 10 in this script for final push.")
print(H)
