# ============================================================
# BLOCK 3: Data Generators
# DenseNet/EfficientNet: 224x224 | InceptionV3: 299x299
# ============================================================

def make_generators(target_size, augment=True):
    if augment:
        train_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.15,
            shear_range=0.10,
            brightness_range=[0.85, 1.15],
            width_shift_range=0.10,
            height_shift_range=0.10,
            validation_split=VAL_SPLIT
        )
    else:
        train_gen = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)

    val_gen = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)

    train_data = train_gen.flow_from_directory(
        DATASET_PATH, target_size=target_size,
        batch_size=BATCH_SIZE, class_mode='categorical',
        subset='training', seed=SEED, shuffle=True
    )
    val_data = val_gen.flow_from_directory(
        DATASET_PATH, target_size=target_size,
        batch_size=BATCH_SIZE, class_mode='categorical',
        subset='validation', seed=SEED, shuffle=False
    )
    return train_data, val_data

print("Loading 224x224 generators (DenseNet + EfficientNet)...")
train_224, val_224 = make_generators((224, 224))

print("\nLoading 299x299 generators (InceptionV3)...")
train_299, val_299 = make_generators((299, 299))

print("\nClass mapping:", train_224.class_indices)
print("Train samples:", train_224.samples)
print("Val samples:  ", val_224.samples)
