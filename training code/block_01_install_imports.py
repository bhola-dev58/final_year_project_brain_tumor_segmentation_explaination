# ============================================================
# BLOCK 1: Install Dependencies and Imports
# ConvNeXtSmall: CVPR 2022 (Meta AI) - High performance architecture
# Research Reference: bioengineering-13-00157 (2026)
# ============================================================
!pip install -q scikit-learn matplotlib seaborn

import os, sys, gc, glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import DenseNet121, InceptionV3, ConvNeXtSmall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus if gpus else "None (Running on CPU)")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU Memory Growth Enabled.")
    except Exception as e:
        print("[INFO] GPU configuration note:", e)
