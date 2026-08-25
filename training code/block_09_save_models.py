# ============================================================
# BLOCK 9 (FIXED): Save models + Confusion Matrix
# Note: EfficientNetV2S cannot be saved as .h5 — saved as .keras
# ============================================================
import os

# Save DenseNet and Inception as .h5 (compatible with existing project)
best_densenet.save('/kaggle/working/densenet_best.h5')
print("densenet_best.h5 saved")

best_inception.save('/kaggle/working/inception_best.h5')
print("inception_best.h5 saved")

# EfficientNetV2S CANNOT save as .h5 — save as .keras format
best_effnet.save('/kaggle/working/effnet_best.keras')
print("effnet_best.keras saved")

print("\nAll models saved successfully!")
print("\nFiles in /kaggle/working/:")
for f in os.listdir('/kaggle/working/'):
    if f.endswith('.h5') or f.endswith('.keras') or f.endswith('.png'):
        size = os.path.getsize(f'/kaggle/working/{f}') / (1024 * 1024)
        print(f"  {f}  ({size:.1f} MB)")

# Confusion Matrix
cm = confusion_matrix(true_labels, ensemble_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix — 3-Model TTA Ensemble")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png', dpi=150)
plt.show()

# Individual vs Ensemble accuracy
print("\nModel-wise Accuracy:")
for name, p in [('DenseNet121', pred_dense), ('InceptionV3', pred_inc), ('EfficientNetV2S', pred_eff)]:
    print(f"  {name:<22}: {accuracy_score(true_labels, np.argmax(p, 1)) * 100:.4f}%")
print(f"  {'Ensemble (TTA)':<22}: {acc * 100:.4f}%")
