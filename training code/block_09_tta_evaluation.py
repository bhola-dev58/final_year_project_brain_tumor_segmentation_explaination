# ============================================================
# BLOCK 9: TTA and Tri-Ensemble Evaluation
# Models: DenseNet121 + InceptionV3 + ConvNeXtSmall
# 10-Pass Test-Time Augmentation (TTA)
# Weighted Tri-Ensemble: ConvNeXtSmall 0.45, InceptionV3 0.35, DenseNet121 0.20
# ============================================================

def make_tta_generator(target_size):
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
        subset='validation', seed=None, shuffle=False
    )

def tta_predict(model, target_size, tta_steps=10):
    preds = []
    for i in range(tta_steps):
        gen = make_tta_generator(target_size)
        preds.append(model.predict(gen, verbose=0))
        print(f"  Pass {i+1}/{tta_steps} complete...")
    return np.mean(preds, axis=0)

# Load best checkpoints (fallback to p2 or p1 if stopped early)
def load_best(prefix):
    for candidate in [f'{prefix}_full_best.keras', f'{prefix}_p2_best.keras', f'{prefix}_p1_best.keras']:
        if os.path.exists(candidate):
            print(f"Loading checkpoint: {candidate}")
            return tf.keras.models.load_model(candidate)
    raise FileNotFoundError(f"No checkpoint found for {prefix}")

print("Loading best fine-tuned checkpoints...")
best_densenet  = load_best('densenet')
best_inception = load_best('inception')
best_convnext  = load_best('convnext')

# Ground truth from clean non-augmented generator
val_gen_clean = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)
val_clean = val_gen_clean.flow_from_directory(
    DATASET_PATH, target_size=(224, 224),
    batch_size=BATCH_SIZE, class_mode='categorical',
    subset='validation', seed=SEED, shuffle=False
)
true_labels = val_clean.classes
class_names = list(val_clean.class_indices.keys())

# Run 10-Pass TTA predictions
print("\nRunning 10-Pass TTA Predictions on DenseNet121...")
pred_dense = tta_predict(best_densenet, (224, 224), tta_steps=10)

print("\nRunning 10-Pass TTA Predictions on InceptionV3...")
pred_inc = tta_predict(best_inception, (299, 299), tta_steps=10)

print("\nRunning 10-Pass TTA Predictions on ConvNeXtSmall...")
pred_cnx = tta_predict(best_convnext, (224, 224), tta_steps=10)

# Weighted Tri-Ensemble (ConvNeXtSmall dominant)
# ConvNeXtSmall: 0.45 | InceptionV3: 0.35 | DenseNet121: 0.20
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

# Individual Model Accuracy
print("\nIndividual Model Accuracy:")
for name, p in [('DenseNet121', pred_dense), ('InceptionV3', pred_inc), ('ConvNeXtSmall', pred_cnx)]:
    print(f"  {name:<22}: {accuracy_score(true_labels, np.argmax(p, 1)) * 100:.4f}%")
print(f"  {'Tri-Ensemble (TTA x10)':<22}: {final_acc:.4f}%")
