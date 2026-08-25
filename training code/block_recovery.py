# ============================================================
# RECOVERY BLOCK — Session restart ke baad chalao
# Training dobara nahi karni — sirf reload + evaluate + save
# Use this when Kaggle session restarts and variables are lost
# ============================================================
import os, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Config (same as before)
DATASET_PATH = '/kaggle/input/datasets/bholadev58/data-source/brain-tumor-2d-dataset/image'
BATCH_SIZE   = 32
VAL_SPLIT    = 0.20
SEED         = 42

# Step 1: Check which files are available
print("Files in /kaggle/working/:")
for f in os.listdir('/kaggle/working/'):
    if f.endswith('.keras') or f.endswith('.h5'):
        size = os.path.getsize(f'/kaggle/working/{f}') / (1024 * 1024)
        print(f"  {f}  ({size:.1f} MB)")

# Step 2: Load best models
# NOTE: Change filenames below based on which phase was last completed
# After Phase 2: densenet_ft_best.keras, inception_ft_best.keras, effnet_ft_best.keras
# After Phase 3: densenet_full_best.keras, inception_full_best.keras, effnet_full_best.keras
best_densenet  = tf.keras.models.load_model('/kaggle/working/densenet_ft_best.keras')
best_inception = tf.keras.models.load_model('/kaggle/working/inception_ft_best.keras')
best_effnet    = tf.keras.models.load_model('/kaggle/working/effnet_ft_best.keras')
print("\nAll 3 models loaded successfully!")

# Step 3: Recreate clean val generators
val_gen_224 = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)
val_clean = val_gen_224.flow_from_directory(
    DATASET_PATH, target_size=(224, 224),
    batch_size=BATCH_SIZE, class_mode='categorical',
    subset='validation', seed=SEED, shuffle=False
)
true_labels = val_clean.classes
class_names = list(val_clean.class_indices.keys())

# Step 4: TTA functions
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

def tta_predict(model, target_size, tta_steps=5):
    preds = []
    for i in range(tta_steps):
        gen = make_tta_generator(target_size)
        preds.append(model.predict(gen, verbose=0))
        print(f"  TTA pass {i+1}/{tta_steps} done")
    return np.mean(preds, axis=0)

# Step 5: Run TTA predictions
print("\nRunning TTA for DenseNet121...")
pred_dense = tta_predict(best_densenet, (224, 224))
print("\nRunning TTA for InceptionV3...")
pred_inc   = tta_predict(best_inception, (299, 299))
print("\nRunning TTA for EfficientNetV2S...")
pred_eff   = tta_predict(best_effnet, (224, 224))

# Step 6: Ensemble + Results
ensemble_probs = (0.35 * pred_dense + 0.30 * pred_inc + 0.35 * pred_eff)
ensemble_preds = np.argmax(ensemble_probs, axis=1)
acc = accuracy_score(true_labels, ensemble_preds)
print(f"\nFINAL ENSEMBLE (TTA) ACCURACY: {acc * 100:.4f}%")
print(classification_report(true_labels, ensemble_preds, target_names=class_names, digits=4))

# Step 7: Save models
best_densenet.save('/kaggle/working/densenet_best.h5')
best_inception.save('/kaggle/working/inception_best.h5')
best_effnet.save('/kaggle/working/effnet_best.keras')
print("\nModels saved!")

# Step 8: Confusion Matrix
cm = confusion_matrix(true_labels, ensemble_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix — 3-Model TTA Ensemble")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png', dpi=150)
plt.show()

# Step 9: Individual accuracies
print("\nModel-wise Accuracy:")
for name, p in [('DenseNet121', pred_dense), ('InceptionV3', pred_inc), ('EfficientNetV2S', pred_eff)]:
    print(f"  {name:<22}: {accuracy_score(true_labels, np.argmax(p, 1)) * 100:.4f}%")
print(f"  {'Ensemble (TTA)':<22}: {acc * 100:.4f}%")
