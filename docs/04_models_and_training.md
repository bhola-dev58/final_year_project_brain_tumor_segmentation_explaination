# 04 — Models & Training

## Overview

Two pre-trained convolutional neural networks are used as classification backbones. Both are fine-tuned on the brain tumor dataset and combined into a **Soft-Voting Ensemble** for final inference.

---

## Model 1: DenseNet121

### Architecture Summary
- **Family:** DenseNet (Densely Connected Convolutional Network)
- **Depth:** 121 layers
- **Key Feature:** Every layer receives feature maps from all preceding layers (dense connections) — promotes feature reuse and reduces gradient vanishing
- **File:** `models/densenet_best.h5` (37.4 MB)

### Transfer Learning Setup

```
DenseNet121 (ImageNet pre-trained)
     │
     └── GlobalAveragePooling2D
           │
           └── Dense(256, activation='relu')
                 │
                 └── Dropout(0.5)
                       │
                       └── Dense(4, activation='softmax')   ← 4 tumor classes
```

### Training Configuration

| Parameter | Value |
|---|---|
| Input Size | (224, 224, 3) |
| Batch Size | 10 (as per base paper) |
| Optimizer | Adam (lr = 0.0001) |
| Loss | Categorical Cross-Entropy |
| Epochs | Until EarlyStopping |
| Validation Split | 30% |
| Callbacks | ModelCheckpoint (best val_accuracy), EarlyStopping (patience=10), ReduceLROnPlateau |

### Why DenseNet121 for Grad-CAM?
DenseNet121 is used as the primary Grad-CAM backbone because its dense connectivity pattern produces rich, high-resolution activation maps in the final convolutional layer. This results in cleaner, more localized heatmaps compared to InceptionV3.

---

## Model 2: InceptionV3

### Architecture Summary
- **Family:** Inception (GoogLeNet-style)
- **Depth:** 48 convolutional layers (inception modules)
- **Key Feature:** Parallel multi-scale convolutions (1×1, 3×3, 5×5 within same module) capture features at different scales simultaneously
- **File:** `models/inception_best.h5` (102.4 MB)

### Transfer Learning Setup

```
InceptionV3 (ImageNet pre-trained)
     │
     └── GlobalAveragePooling2D
           │
           └── Dense(256, activation='relu')
                 │
                 └── Dropout(0.5)
                       │
                       └── Dense(4, activation='softmax')   ← 4 tumor classes
```

### Training Configuration

Same as DenseNet121 above — identical hyperparameters as per the base paper methodology.

---

## Soft-Voting Ensemble

The ensemble combines both models' outputs at the **probability level** (not the decision level):

```python
pred_dense = model_dense.predict(img_array, verbose=0)   # shape: (1, 4)
pred_inc   = model_inc.predict(img_array, verbose=0)      # shape: (1, 4)
avg_pred   = (pred_dense + pred_inc) / 2.0               # element-wise average
class_idx  = int(np.argmax(avg_pred))                    # final decision
confidence = float(np.max(avg_pred)) * 100               # in percent
```

### Why Soft Voting?

| Strategy | Description | Advantage |
|---|---|---|
| **Hard Voting** | Each model votes for a class label; majority wins | Simple but discards confidence information |
| **Soft Voting** ✓ | Average the raw probability vectors; pick argmax | Retains full probability signal; more nuanced |

Soft voting is preferred because it weighs predictions by how confident each model is. A model that is 99% confident in "Glioma" has more influence than one that is 51% confident.

---

## Model Performance

### Ensemble Results (Validation Set, 70/30 split)

| Metric | Value |
|---|---|
| **Accuracy** | **78.88%** |
| **Macro Precision** | **80.00%** |
| **Macro Recall** | **81.00%** |
| **Macro F1-Score** | **80.00%** |

### Baseline Comparison

| Architecture | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| InceptionV3 (standalone) | 76.89% | 77.47% | 79.00% | 77.30% |
| DenseNet121 (standalone) | 77.52% | 79.40% | 79.39% | 78.19% |
| **Proposed Ensemble** | **78.88%** | **80.00%** | **81.00%** | **80.00%** |

The ensemble consistently outperforms both individual models.

### Per-Class F1-Score (Ensemble)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No Tumor | 94.17% | 70.06% | 80.39% |
| Glioma | 90.32% | 86.60% | 88.42% |
| Meningioma | 63.66% | 68.33% | 65.91% |
| Pituitary | 74.87% | 96.97% | 84.52% |

> Meningioma has the lowest F1 — it is anatomically the most variable tumor type, often confused with glioma at its margins.

---

## Confusion Matrix (Ensemble, Validation Set)

```
                  Predicted →
                No Tumor  Glioma  Meningioma  Pituitary
Actual ↓
  No Tumor  │    339        8         83          48   │
  Glioma    │      2      168         23           1   │
  Meningioma│     17        9        205          68   │
  Pituitary │      0        0          9         289   │
```

Key observations:
- **Glioma** and **Pituitary** are classified with high accuracy
- **Meningioma** is frequently confused with No Tumor and Pituitary — expected due to its variable shape and location
- **No Tumor** has some false positives classified as Meningioma — conservative in detecting no-tumor cases

---

## Grad-CAM Sub-Model Construction

After loading DenseNet121, the Grad-CAM sub-model is built dynamically:

```python
# Find last 4D convolutional layer automatically
for layer in reversed(model_dense.layers):
    if len(layer.output.shape) == 4:
        _last_conv_layer_name = layer.name
        break

# Build sub-model that outputs (conv_activations, predictions)
_grad_model = tf.keras.Model(
    inputs=model_dense.inputs,
    outputs=[
        model_dense.get_layer(_last_conv_layer_name).output,
        model_dense.output
    ]
)
```

This sub-model is used by `make_gradcam_heatmap()` during every prediction.

---

## U-Net Segmentation Model (Optional)

A separate lightweight U-Net can be trained using `scripts/train_segmentation.py`. This is **optional** and **not used** in the primary inference pipeline — the main system uses Grad-CAM-guided morphological segmentation instead.

### U-Net Architecture

```
Encoder:   Conv(32) → MaxPool → Conv(64) → MaxPool → Conv(128) → MaxPool
Bottleneck: Conv(256)
Decoder:   UpSample + Skip(128) → UpSample + Skip(64) → UpSample + Skip(32)
Output:    Conv(1, sigmoid) — binary mask
```

### U-Net Training Config

| Parameter | Value |
|---|---|
| Input Size | (256, 256, 3) |
| Batch Size | 16 |
| Epochs | 15 (with EarlyStopping) |
| Loss | Binary Cross-Entropy + Dice Loss |
| Metric | Dice Coefficient |
| Optimizer | Adam (lr=0.001) |
| Train/Val split | 85% / 15% |
