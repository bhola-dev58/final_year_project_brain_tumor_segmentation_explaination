"""
Evaluation Script for BrainTumorXAI Ensemble Model

Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
for DenseNet121, InceptionV3, and the Soft-Voting Ensemble Model.

Usage:
    python scripts/evaluate_models.py --dataset_dir "datasets/image" --val_split 0.30
"""

import os
import sys
import argparse
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

# Robust import for IDE linters and runtime compatibility
try:
    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
except AttributeError:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.config import DENSENET_PATH, INCEPTION_PATH, CLASSES

# Human readable class name mapping
CLASS_NAME_MAP = {
    '0': 'No Tumor',
    '1': 'Glioma Tumor',
    '2': 'Meningioma Tumor',
    '3': 'Pituitary Tumor'
}

def compute_metrics(y_true, y_pred):
    """Computes accuracy, macro precision, recall, and f1-score."""
    acc = accuracy_score(y_true, y_pred) * 100
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    return acc, prec, rec, f1

def evaluate_models(dataset_path: str, val_split: float = 0.30, tta: bool = False):
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset directory not found: {dataset_path}")
        print("Please provide a valid dataset directory containing subfolders for each class.")
        return

    print(f"Loading dataset from: {dataset_path}")
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split
    )

    print("\nLoading models from disk...")
    model_dense = tf.keras.models.load_model(DENSENET_PATH, compile=False)
    model_inc = tf.keras.models.load_model(INCEPTION_PATH, compile=False)
    
    from src.config import EFFNET_PATH, DENSENET_VOTE_WEIGHT, INCEPTION_VOTE_WEIGHT, EFFNET_VOTE_WEIGHT
    model_eff = None
    if os.path.exists(EFFNET_PATH):
        try:
            model_eff = tf.keras.models.load_model(EFFNET_PATH, compile=False)
            print("✔ Loaded EfficientNetV2S for Tri-Ensemble.")
        except Exception as e:
            print(f"⚠️ Could not load EfficientNetV2S: {e}")

    # Dynamic target shape resolution
    dense_shape = (model_dense.input_shape[1], model_dense.input_shape[2]) if model_dense.input_shape and model_dense.input_shape[1] else (224, 224)
    inc_shape = (model_inc.input_shape[1], model_inc.input_shape[2]) if model_inc.input_shape and model_inc.input_shape[1] else (299, 299)
    eff_shape = (model_eff.input_shape[1], model_eff.input_shape[2]) if model_eff and model_eff.input_shape and model_eff.input_shape[1] else (224, 224)

    val_data_dense = datagen.flow_from_directory(
        dataset_path, target_size=dense_shape, batch_size=16,
        class_mode='categorical', subset='validation' if val_split > 0 else None,
        seed=42, shuffle=False
    )
    val_data_inc = datagen.flow_from_directory(
        dataset_path, target_size=inc_shape, batch_size=16,
        class_mode='categorical', subset='validation' if val_split > 0 else None,
        seed=42, shuffle=False
    )
    val_data_eff = datagen.flow_from_directory(
        dataset_path, target_size=eff_shape, batch_size=16,
        class_mode='categorical', subset='validation' if val_split > 0 else None,
        seed=42, shuffle=False
    ) if model_eff else None

    if val_data_dense.samples == 0:
        print("[ERROR] No image samples found in dataset directory.")
        return

    true_labels = val_data_dense.classes
    class_indices = val_data_dense.class_indices
    target_names = [CLASS_NAME_MAP.get(k, k) for k in class_indices.keys()]
    
    print(f"Found {val_data_dense.samples} validation images across {len(class_indices)} classes: {class_indices}")

    print("\n[1/2] Running DenseNet121 predictions...")
    pred_dense = model_dense.predict(val_data_dense, verbose=1)
    print("\n[2/2] Running InceptionV3 predictions...")
    pred_inc = model_inc.predict(val_data_inc, verbose=1)

    dense_preds = np.argmax(pred_dense, axis=1)
    inc_preds = np.argmax(pred_inc, axis=1)

    # Optimal Dual-Ensemble Soft-Voting (75% Inception + 25% DenseNet)
    print("\nApplying Optimal Soft-Voting Ensemble (75% InceptionV3 + 25% DenseNet121)...")
    ensemble_probs = (0.25 * pred_dense) + (0.75 * pred_inc)
    ensemble_preds = np.argmax(ensemble_probs, axis=1)

    # Compute overall metrics
    dense_acc, dense_prec, dense_rec, dense_f1 = compute_metrics(true_labels, dense_preds)
    inc_acc, inc_prec, inc_rec, inc_f1 = compute_metrics(true_labels, inc_preds)
    ens_acc, ens_prec, ens_rec, ens_f1 = compute_metrics(true_labels, ensemble_preds)

    # Print Comparison Table
    print("\n" + "=" * 88)
    print("                      MODEL PERFORMANCE METRICS COMPARISON")
    print("=" * 88)
    print(f"{'Architecture':<32} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 88)
    print(f"{'DenseNet121 (Phase 3)':<32} | {dense_acc:>9.2f}% | {dense_prec:>9.2f}% | {dense_rec:>9.2f}% | {dense_f1:>9.2f}%")
    print(f"{'InceptionV3 (Phase 3)':<32} | {inc_acc:>9.2f}% | {inc_prec:>9.2f}% | {inc_rec:>9.2f}% | {inc_f1:>9.2f}%")
    print(f"{'Proposed Dual-Ensemble':<32} | {ens_acc:>9.2f}% | {ens_prec:>9.2f}% | {ens_rec:>9.2f}% | {ens_f1:>9.2f}%")
    print("=" * 88)


    # Per-Class Classification Report for Ensemble
    print("\n" + "=" * 88)
    print(f"              DETAILED CLASSIFICATION REPORT {mode_str.upper()}")
    print("=" * 88)
    print(classification_report(true_labels, ensemble_preds, target_names=target_names, digits=2))

    # Confusion Matrix
    print("=" * 88)
    print("                            CONFUSION MATRIX")
    print("=" * 88)
    cm = confusion_matrix(true_labels, ensemble_preds)
    col_headers = [name[:10] for name in target_names]
    header_str = f"{'Actual \\ Pred':<18} | " + " | ".join([f"{h:>10}" for h in col_headers])
    print(header_str)
    print("-" * len(header_str))
    for i, row in enumerate(cm):
        row_str = f"{target_names[i]:<18} | " + " | ".join([f"{val:>10}" for val in row])
        print(row_str)
    print("=" * 88 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Brain Tumor Models Metrics")
    parser.add_argument(
        "--dataset_dir", 
        type=str, 
        default="datasets/image",
        help="Path to folder with subdirectories 0, 1, 2, 3 or class names"
    )
    parser.add_argument(
        "--val_split", 
        type=float, 
        default=0.30,
        help="Validation split ratio (default 0.30)"
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable 5-pass Test-Time Augmentation (TTA) for maximum accuracy"
    )
    args = parser.parse_args()
    evaluate_models(args.dataset_dir, args.val_split, args.tta)

