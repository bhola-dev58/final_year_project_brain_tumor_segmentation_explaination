import os
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, InceptionV3
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np

# Ensure models directory exists
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test_images")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

densenet_path = os.path.join(MODELS_DIR, "densenet_best.h5")
inception_path = os.path.join(MODELS_DIR, "inception_best.h5")

def build_custom_model(base_model):
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(4, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

if not os.path.exists(densenet_path):
    print("Creating DenseNet121 model structure...")
    base_dense = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
    model_dense = build_custom_model(base_dense)
    model_dense.save(densenet_path)
    print(f"Saved: {densenet_path}")
else:
    print(f"Found existing: {densenet_path}")

if not os.path.exists(inception_path):
    print("Creating InceptionV3 model structure...")
    base_inc = InceptionV3(weights=None, include_top=False, input_shape=(224, 224, 3))
    model_inc = build_custom_model(base_inc)
    model_inc.save(inception_path)
    print(f"Saved: {inception_path}")
else:
    print(f"Found existing: {inception_path}")

# Create sample test MRI scans if missing
sample_names = ["Tr-me_0025.jpg", "Tr-me_0070.jpg", "Tr-me_0080.jpg", "Tr-pi_0050.jpg"]
for name in sample_names:
    img_path = os.path.join(TEST_IMAGES_DIR, name)
    if not os.path.exists(img_path):
        # Generate a realistic brain MRI oval shape with tumor-like region
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        # Brain skull outline / ellipse
        cv2.ellipse(img, (128, 128), (95, 115), 0, 0, 360, (160, 160, 160), -1)
        # Inner brain structure texture
        cv2.ellipse(img, (128, 128), (85, 105), 0, 0, 360, (90, 90, 90), -1)
        # Ventricles
        cv2.ellipse(img, (115, 125), (10, 25), -15, 0, 360, (30, 30, 30), -1)
        cv2.ellipse(img, (141, 125), (10, 25), 15, 0, 360, (30, 30, 30), -1)
        # Simulated hyperintense lesion (tumor)
        cv2.circle(img, (100, 105), 18, (230, 230, 230), -1)
        # Add subtle gaussian noise
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(img_path, img)
        print(f"Created sample MRI image: {img_path}")

print("Setup completed successfully.")
