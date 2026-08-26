"""
Train a lightweight U-Net segmentation model on the brain tumor mask dataset.
Run this once to generate models/segmentation_model.h5

Usage: python scripts/train_segmentation.py
"""
import os
import sys

# Add the parent directory to system path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import IMG_SIZE_TRAIN as IMG_SIZE, logger

# Suppress verbose logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

BATCH_SIZE = 16
EPOCHS = 15
DATA_DIR = "brain-tumor-2d-dataset"


def load_dataset():
    images = []
    masks = []
    # Load from tumor classes only (1=glioma, 2=meningioma, 3=pituitary)
    for cls in ['1', '2', '3']:
        img_dir = os.path.join(DATA_DIR, 'image', cls)
        mask_dir = os.path.join(DATA_DIR, 'mask', cls)

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            logger.warning(f"Skipping class {cls}: directory not found")
            continue

        for fname in sorted(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, fname)
            # Mask filename: same name but with _m suffix before extension
            base, ext = os.path.splitext(fname)
            mask_path = os.path.join(mask_dir, f"{base}_m{ext}")

            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

            # Normalize
            img = img.astype(np.float32) / 255.0
            mask = (mask > 127).astype(np.float32)

            images.append(img)
            masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)[..., np.newaxis]  # Add channel dim

    logger.info(f"Loaded {len(images)} image-mask pairs")
    return images, masks


def build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder (downsampling path)
    c1 = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    c1 = layers.Conv2D(32, 3, padding='same', activation='relu')(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    c2 = layers.Conv2D(64, 3, padding='same', activation='relu')(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = layers.Conv2D(128, 3, padding='same', activation='relu')(p2)
    c3 = layers.Conv2D(128, 3, padding='same', activation='relu')(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    # Bottleneck
    c4 = layers.Conv2D(256, 3, padding='same', activation='relu')(p3)
    c4 = layers.Conv2D(256, 3, padding='same', activation='relu')(c4)

    # Decoder (upsampling path with skip connections)
    u5 = layers.UpSampling2D(2)(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(128, 3, padding='same', activation='relu')(u5)
    c5 = layers.Conv2D(128, 3, padding='same', activation='relu')(c5)

    u6 = layers.UpSampling2D(2)(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(64, 3, padding='same', activation='relu')(u6)
    c6 = layers.Conv2D(64, 3, padding='same', activation='relu')(c6)

    u7 = layers.UpSampling2D(2)(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(32, 3, padding='same', activation='relu')(u7)
    c7 = layers.Conv2D(32, 3, padding='same', activation='relu')(c7)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)

    model = keras.Model(inputs, outputs)
    return model


def dice_loss(y_true, y_pred):
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1 - (2 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def combined_loss(y_true, y_pred):
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return bce + dl


def dice_coeff(y_true, y_pred):
    smooth = 1.0
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    return (2 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) + smooth
    )


if __name__ == "__main__":
    logger.info("Starting dataset loading...")
    X, Y = load_dataset()

    if len(X) == 0:
        logger.error("No image-mask pairs loaded. Make sure the dataset paths exist.")
        sys.exit(1)

    # Split: 85% train, 15% validation
    split = int(0.85 * len(X))
    indices = np.random.permutation(len(X))
    X, Y = X[indices], Y[indices]
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    logger.info(f"Split sizes: Train={len(X_train)} | Validation={len(X_val)}")

    # Data augmentation
    aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ])

    logger.info("Building U-Net model...")
    model = build_unet()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=combined_loss,
        metrics=[dice_coeff]
    )
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_dice_coeff', patience=10, mode='max',
            restore_best_weights=True
        ),
    ]

    logger.info("Beginning model training...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    save_path = "models/segmentation_model.h5"
    model.save(save_path)
    logger.info(f"Segmentation model saved to: {save_path}")

    # Quick validation preview
    val_preds = model.predict(X_val[:5], verbose=0)
    for i in range(min(5, len(X_val))):
        pred_area = np.sum(val_preds[i] > 0.5) / (IMG_SIZE * IMG_SIZE) * 100
        gt_area = np.sum(Y_val[i] > 0.5) / (IMG_SIZE * IMG_SIZE) * 100
        logger.info(f"Sample {i}: GT area={gt_area:.2f}% | Predicted area={pred_area:.2f}%")

    logger.info("Segmentation training process completed successfully!")
