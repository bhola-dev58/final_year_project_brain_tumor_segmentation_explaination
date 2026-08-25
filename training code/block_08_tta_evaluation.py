# ============================================================
# BLOCK 8 (FIXED): Proper TTA + Ensemble Evaluation
# ============================================================

def make_tta_generator(target_size):
    """Fresh augmented generator each call = different augmentations per TTA pass"""
    tta_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.10,
        validation_split=VAL_SPLIT
    )
    return tta_datagen.flow_from_directory(
        DATASET_PATH, target_size=target_size,
        batch_size=BATCH_SIZE, class_mode='categorical',
        subset='validation', seed=None, shuffle=False  # seed=None = different aug each pass
    )

def tta_predict(model, target_size, tta_steps=5):
    preds = []
    for i in range(tta_steps):
        gen = make_tta_generator(target_size)
        preds.append(model.predict(gen, verbose=0))
        print(f"  TTA pass {i+1}/{tta_steps} done")
    return np.mean(preds, axis=0)

# Load best Phase-2 checkpoints
print("Loading best fine-tuned checkpoints...")
best_densenet  = tf.keras.models.load_model('densenet_ft_best.keras')
best_inception = tf.keras.models.load_model('inception_ft_best.keras')
best_effnet    = tf.keras.models.load_model('effnet_ft_best.keras')

# Get true labels from clean non-augmented generator
val_gen_clean = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)
val_clean = val_gen_clean.flow_from_directory(
    DATASET_PATH, target_size=(224, 224),
    batch_size=BATCH_SIZE, class_mode='categorical',
    subset='validation', seed=SEED, shuffle=False
)
true_labels = val_clean.classes
class_names = list(val_clean.class_indices.keys())

# Run TTA predictions
print("\nRunning TTA for DenseNet121...")
pred_dense = tta_predict(best_densenet, (224, 224))

print("\nRunning TTA for InceptionV3...")
pred_inc   = tta_predict(best_inception, (299, 299))

print("\nRunning TTA for EfficientNetV2S...")
pred_eff   = tta_predict(best_effnet, (224, 224))

# Weighted ensemble
ensemble_probs = (0.35 * pred_dense + 0.30 * pred_inc + 0.35 * pred_eff)
ensemble_preds = np.argmax(ensemble_probs, axis=1)
acc = accuracy_score(true_labels, ensemble_preds)

print(f"\nFINAL ENSEMBLE (TTA) VALIDATION ACCURACY: {acc * 100:.4f}%")
print("\nCLASSIFICATION REPORT:")
print(classification_report(true_labels, ensemble_preds, target_names=class_names, digits=4))
