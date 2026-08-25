# ============================================================
# BLOCK 1: Install dependencies & Imports
# ============================================================
!pip install -q tensorflow keras scikit-learn matplotlib seaborn

import os, gc, numpy as np, tensorflow as tf
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import DenseNet121, InceptionV3, EfficientNetV2S
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TF version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))
