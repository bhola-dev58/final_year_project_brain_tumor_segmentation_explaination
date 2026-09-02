# ============================================================
# RECOVERY BLOCK - Run after session restart to restore models
# Reloads best saved checkpoints, runs 10-Pass TTA and exports metrics
#
# Fully synchronized with Blocks 08, 09 and 10 (ConvNeXt Tri-Ensemble)
# ============================================================
import os, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -----------------------------------------------
# Step 1: Dataset Path & Global Configuration
# -----------------------------------------------
DATASET_PATH = '/kaggle/input/datasets/bholadev58/brain-tumor-mri-4-class-dataset/datasets/image'

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"[ERROR] Dataset directory not found at: {DATASET_PATH}")
else:
    print(f"[INFO] Located Dataset at: {DATASET_PATH}")

BATCH_SIZE = 32
VAL_SPLIT  = 0.20
SEED       = 42

# -----------------------------------------------
# Step 2: Auto-detect best available checkpoints
# Priority: Phase 3 (full_best) > Phase 2 (p2_best) > Phase 1 (p1_best)
# -----------------------------------------------
output_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
print("\nFiles in output directory:")
all_files = os.listdir(output_dir)
for f in sorted(all_files):
    if f.endswith('.keras') or f.endswith('.h5'):
        size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
        print(f"  {f:<32} ({size:.1f} MB)")

def best_checkpoint(prefix):
    """Auto-pick highest phase checkpoint available"""
    for suffix in ['full_best', 'p2_best', 'p1_best']:
        path = os.path.join(output_dir, f'{prefix}_{suffix}.keras')
        if os.path.exists(path):
            print(f"[INFO] Found Checkpoint: {path}")
            return path
    raise FileNotFoundError(f"No checkpoint found for '{prefix}'. Run at least Block 6 first.")

print("\nAuto-detecting best available checkpoints...")
dense_path = best_checkpoint('densenet')
inc_path   = best_checkpoint('inception')
cnx_path   = best_checkpoint('convnext')

# -----------------------------------------------
# Step 3: Load models
# -----------------------------------------------
print("\nLoading models into memory...")
best_densenet  = tf.keras.models.load_model(dense_path)
best_inception = tf.keras.models.load_model(inc_path)
best_convnext  = tf.keras.models.load_model(cnx_path)
print("[INFO] All 3 Tri-Ensemble models loaded successfully.")

# -----------------------------------------------
# Step 4: Recreate clean validation generator
# -----------------------------------------------
val_gen_clean = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)
val_clean = val_gen_clean.flow_from_directory(
    DATASET_PATH, target_size=(224, 224),
    batch_size=BATCH_SIZE, class_mode='categorical',
    subset='validation', seed=SEED, shuffle=False
)
true_labels = val_clean.classes
class_names = list(val_clean.class_indices.keys())

# -----------------------------------------------
# Step 5: TTA functions (10 passes for max accuracy)
# -----------------------------------------------
def make_tta_generator(target_size):
    tta_datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=15,
        horizontal_flip=True, zoom_range=0.10,
        validation_split=VAL_SPLIT
    )
    return tta_datagen.flow_from_directory(
        DATASET_PATH, target_size=target_size,
        batch_size=BATCH_SIZE, class_mode='categorical',
        subset='validation', seed=None, shuffle=False
    )

def tta_predict(model, target_size, tta_steps=10):
    preds = []
    for i in range(tta_steps):
        gen = make_tta_generator(target_size)
        preds.append(model.predict(gen, verbose=0))
        print(f"  Pass {i+1}/{tta_steps} complete...")
    return np.mean(preds, axis=0)

# -----------------------------------------------
# Step 6: Run 10-Pass TTA predictions
# -----------------------------------------------
print("\nRunning 10-Pass TTA Predictions on DenseNet121...")
pred_dense = tta_predict(best_densenet, (224, 224), tta_steps=10)

print("\nRunning 10-Pass TTA Predictions on InceptionV3...")
pred_inc = tta_predict(best_inception, (299, 299), tta_steps=10)

print("\nRunning 10-Pass TTA Predictions on ConvNeXtSmall...")
pred_cnx = tta_predict(best_convnext, (224, 224), tta_steps=10)

# -----------------------------------------------
# Step 7: Weighted Tri-Ensemble + Evaluation
# ConvNeXtSmall: 0.45 | InceptionV3: 0.35 | DenseNet121: 0.20
# -----------------------------------------------
ensemble_probs = (0.20 * pred_dense) + (0.35 * pred_inc) + (0.45 * pred_cnx)
ensemble_preds = np.argmax(ensemble_probs, axis=1)

final_acc = accuracy_score(true_labels, ensemble_preds) * 100
final_f1  = f1_score(true_labels, ensemble_preds, average='macro') * 100

print("\n" + "=" * 60)
print(f"FINAL ENSEMBLE VALIDATION ACCURACY: {final_acc:.2f}%")
print(f"FINAL ENSEMBLE MACRO F1-SCORE:     {final_f1:.2f}%")
print("=" * 60)
print("\nDetailed Classification Report:")
print(classification_report(true_labels, ensemble_preds, target_names=class_names, digits=4))

# -----------------------------------------------
# Step 8: Save final models (.keras & .h5)
# -----------------------------------------------
dense_save = os.path.join(output_dir, 'densenet_full_best.keras')
inc_save   = os.path.join(output_dir, 'inception_full_best.keras')
cnx_save   = os.path.join(output_dir, 'convnext_full_best.keras')

best_densenet.save(dense_save)
best_inception.save(inc_save)
best_convnext.save(cnx_save)
print(f"\n[INFO] Saved all 3 .keras models to: {output_dir}")

try:
    best_densenet.save(os.path.join(output_dir, 'densenet_best.h5'))
    best_inception.save(os.path.join(output_dir, 'inception_best.h5'))
    print("[INFO] Saved legacy .h5 formats for DenseNet and Inception.")
except Exception as e:
    print("[INFO] Note on .h5 export:", e)

# -----------------------------------------------
# Step 9: Plot Confusion Matrix
# -----------------------------------------------
cm = confusion_matrix(true_labels, ensemble_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"Tri-Ensemble Confusion Matrix (Acc: {final_acc:.2f}%)")
plt.xlabel("Predicted Label")
plt.ylabel("Actual True Label")
plt.tight_layout()
cm_path = os.path.join(output_dir, 'ensemble_confusion_matrix.png')
plt.savefig(cm_path, dpi=200)
plt.show()
print(f"[INFO] Saved Confusion Matrix plot to: {cm_path}")

# -----------------------------------------------
# Step 10: Individual model accuracy summary
# -----------------------------------------------
print("\nFinal Model-wise Accuracy:")
for name, p in [('DenseNet121', pred_dense), ('InceptionV3', pred_inc), ('ConvNeXtSmall', pred_cnx)]:
    print(f"  {name:<22}: {accuracy_score(true_labels, np.argmax(p, 1)) * 100:.4f}%")
print(f"  {'Tri-Ensemble (TTA x10)':<22}: {final_acc:.4f}%")
