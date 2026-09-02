# ============================================================
# BLOCK 3: Data Generators and Mixup Augmentation
# DenseNet121: 224x224 | InceptionV3: 299x299 | ConvNeXtSmall: 224x224
# Mixup Reference: bioengineering-13-00157 (2026)
# Augmentation Note:
#   - MRI anatomy must be preserved. Shear distorts tissue structure.
#   - Rotation is capped at 15 degrees to keep brain orientation realistic.
#   - channel_shift_range adds contrast variation without spatial distortion.
# ============================================================

def make_generators(target_size, augment=True):
    if augment:
        train_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,           # Reduced: MRI scans have limited realistic rotation
            horizontal_flip=True,         # Valid: left/right brain symmetry
            vertical_flip=False,          # Off: brain orientation must be upright
            zoom_range=0.10,              # Reduced: prevents over-cropping of tumor regions
            shear_range=0.0,              # Removed: shear distorts MRI tissue structure
            brightness_range=[0.90, 1.10],# Tightened: subtle contrast variation only
            width_shift_range=0.08,       # Reduced: small shift for positional robustness
            height_shift_range=0.08,      # Reduced: small shift for positional robustness
            channel_shift_range=10.0,     # Added: simulates scanner contrast differences
            fill_mode='constant',
            cval=0,
            validation_split=VAL_SPLIT
        )
    else:
        train_gen = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)

    val_gen = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)

    train_data = train_gen.flow_from_directory(
        DATASET_PATH,
        target_size=target_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=SEED,
        shuffle=True
    )
    val_data = val_gen.flow_from_directory(
        DATASET_PATH,
        target_size=target_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=SEED,
        shuffle=False
    )
    return train_data, val_data

def mixup_generator(generator, alpha=0.2):
    """
    Mixup augmentation: x_mix = lam*x1 + (1-lam)*x2
    Used in Phase 1 and Phase 2 only.
    Phase 3 uses direct (non-mixed) batches for fine-grained convergence.
    Reference: bioengineering-13-00157 (2026)
    """
    while True:
        x1, y1 = next(generator)
        x2, y2 = next(generator)
        batch = min(x1.shape[0], x2.shape[0])
        x1, y1 = x1[:batch], y1[:batch]
        x2, y2 = x2[:batch], y2[:batch]
        lam = np.random.beta(alpha, alpha)
        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2
        yield x_mixed, y_mixed

print("[INFO] Loading 224x224 generators (DenseNet121 and ConvNeXtSmall)...")
train_224, val_224 = make_generators((224, 224))

print("\n[INFO] Loading 299x299 generators (InceptionV3 Native Resolution)...")
train_299, val_299 = make_generators((299, 299))

print("\nClass mapping:", train_224.class_indices)
print(f"Total Training Samples:   {train_224.samples}")
print(f"Total Validation Samples: {val_224.samples}")
print("[INFO] Data generators and Mixup pipeline initialized.")
 