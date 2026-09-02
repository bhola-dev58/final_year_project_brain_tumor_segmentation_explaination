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

from src.config import DENSENET_PATH, INCEPTION_PATH, CONVNEXT_PATH, DENSENET_VOTE_WEIGHT, INCEPTION_VOTE_WEIGHT, CONVNEXT_VOTE_WEIGHT

dataset_path = "datasets/image"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.30
)

print("Loading DenseNet121 and InceptionV3...")
model_dense = tf.keras.models.load_model(DENSENET_PATH, compile=False)
model_inc = tf.keras.models.load_model(INCEPTION_PATH, compile=False)

model_cnx = None
if os.path.exists(CONVNEXT_PATH):
    try:
        model_cnx = tf.keras.models.load_model(CONVNEXT_PATH, compile=False)
        print("✔ Loaded ConvNeXtSmall for Tri-Ensemble.")
    except Exception as e:
        print(f"⚠️ Could not load ConvNeXtSmall: {e}")

dense_shape = (model_dense.input_shape[1], model_dense.input_shape[2]) if model_dense.input_shape and model_dense.input_shape[1] else (224, 224)
inc_shape = (model_inc.input_shape[1], model_inc.input_shape[2]) if model_inc.input_shape and model_inc.input_shape[1] else (299, 299)
cnx_shape = (model_cnx.input_shape[1], model_cnx.input_shape[2]) if model_cnx and model_cnx.input_shape and model_cnx.input_shape[1] else (224, 224)

val_data_dense = datagen.flow_from_directory(
    dataset_path,
    target_size=dense_shape,
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    seed=42,
    shuffle=False
)

val_data_inc = datagen.flow_from_directory(
    dataset_path,
    target_size=inc_shape,
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    seed=42,
    shuffle=False
)

val_data_cnx = datagen.flow_from_directory(
    dataset_path,
    target_size=cnx_shape,
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    seed=42,
    shuffle=False
) if model_cnx else None

true_labels = val_data_dense.classes
class_indices = val_data_dense.class_indices
target_names = ['No Tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']

print(f"Loaded {len(true_labels)} validation samples across classes: {class_indices}")

print("Predicting DenseNet121...")
pred_dense = model_dense.predict(val_data_dense, verbose=1)
print("Predicting InceptionV3...")
pred_inc = model_inc.predict(val_data_inc, verbose=1)

if model_cnx and val_data_cnx:
    print("Predicting ConvNeXtSmall...")
    pred_cnx = model_cnx.predict(val_data_cnx, verbose=1)
    tot_w = CONVNEXT_VOTE_WEIGHT + INCEPTION_VOTE_WEIGHT + DENSENET_VOTE_WEIGHT
    ensemble_probs = (pred_cnx * CONVNEXT_VOTE_WEIGHT + pred_inc * INCEPTION_VOTE_WEIGHT + pred_dense * DENSENET_VOTE_WEIGHT) / tot_w
else:
    ensemble_probs = (pred_dense * DENSENET_VOTE_WEIGHT) + (pred_inc * INCEPTION_VOTE_WEIGHT)

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
