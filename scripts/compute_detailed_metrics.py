import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

from src.config import DENSENET_PATH, INCEPTION_PATH

dataset_path = "datasets/image"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.30
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

true_labels = val_data.classes
class_indices = val_data.class_indices
target_names = ['No Tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']

print(f"Loaded {len(true_labels)} validation samples across classes: {class_indices}")

model_dense = tf.keras.models.load_model(DENSENET_PATH, compile=False)
model_inc = tf.keras.models.load_model(INCEPTION_PATH, compile=False)

print("Predicting DenseNet121...")
pred_dense = model_dense.predict(val_data, verbose=0)
print("Predicting InceptionV3...")
pred_inc = model_inc.predict(val_data, verbose=0)

ensemble_probs = (pred_dense + pred_inc) / 2.0
ensemble_preds = np.argmax(ensemble_probs, axis=1)

# Metrics
overall_acc = accuracy_score(true_labels, ensemble_preds)
macro_f1 = f1_score(true_labels, ensemble_preds, average='macro')
weighted_f1 = f1_score(true_labels, ensemble_preds, average='weighted')
balanced_acc = balanced_accuracy_score(true_labels, ensemble_preds)

print("\n" + "=" * 60)
print("             EXACT VALIDATION METRICS (4 DECIMALS)")
print("=" * 60)
print(f"Overall Accuracy:   {overall_acc:.4f} ({overall_acc*100:.2f}%)")
print(f"Macro-F1:           {macro_f1:.4f} ({macro_f1*100:.2f}%)")
print(f"Weighted-F1:        {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
print(f"Balanced Accuracy:  {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
print("=" * 60)

print("\n" + "=" * 60)
print("         CLASSIFICATION REPORT (digits=4)")
print("=" * 60)
print(classification_report(true_labels, ensemble_preds, target_names=target_names, digits=4))

print("=" * 60)
print("                 CONFUSION MATRIX")
print("=" * 60)
print(confusion_matrix(true_labels, ensemble_preds))
