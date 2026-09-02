# ============================================================
# BLOCK 10: Save Models and Confusion Matrix
# All 3 Tri-Ensemble models saved for inference pipeline
# DenseNet & Inception saved in both .keras and .h5 formats
# ConvNeXtSmall saved as .keras format
# ============================================================
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

output_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'

# 1. Save all 3 fine-tuned models
dense_full_path = os.path.join(output_dir, 'densenet_full_best.keras')
inc_full_path   = os.path.join(output_dir, 'inception_full_best.keras')
cnx_full_path   = os.path.join(output_dir, 'convnext_full_best.keras')

best_densenet.save(dense_full_path)
print(f"[INFO] Saved DenseNet121 (.keras) to: {dense_full_path}")

best_inception.save(inc_full_path)
print(f"[INFO] Saved InceptionV3 (.keras) to:  {inc_full_path}")

best_convnext.save(cnx_full_path)
print(f"[INFO] Saved ConvNeXtSmall (.keras) to: {cnx_full_path}")

# Save .h5 for backwards compatibility with older loaders
try:
    best_densenet.save(os.path.join(output_dir, 'densenet_best.h5'))
    best_inception.save(os.path.join(output_dir, 'inception_best.h5'))
    print("[INFO] Saved legacy .h5 formats for DenseNet and Inception.")
except Exception as e:
    print("[INFO] Note on .h5 export:", e)

# 2. Plot Confusion Matrix
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

print("\nFiles in output directory:")
for f in os.listdir(output_dir):
    if f.endswith('.h5') or f.endswith('.keras') or f.endswith('.png'):
        size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
        print(f"  {f:<32} ({size:.1f} MB)")

print("\n[SUCCESS] All steps finished successfully. Models are ready for deployment.")
