# ============================================================
# BLOCK 5: Model Builder - ConvNeXtSmall, DenseNet121 and InceptionV3
# ConvNeXtSmall: CVPR 2022 (Meta AI, 50M parameters)
# Research paper reference: bioengineering-13-00157 (2026)
# Architecture Note:
#   - Dual Pooling Head (GAP + GMP) captures both average and peak features.
#   - Peak features from GlobalMaxPooling are critical for detecting small tumors.
#   - ConvNeXtSmall Phase 1 LR is 5e-4 (lower than others) for gradient stability.
# ============================================================

def build_model(base_model, num_classes=4, dropout=0.25, lr=1e-3):
    """Build a classification head with Dual Pooling (GAP + GMP)."""
    base_model.trainable = False  # Freeze backbone for Phase 1

    # Dual Pooling: captures both average and peak spatial features
    gap = layers.GlobalAveragePooling2D()(base_model.output)
    gmp = layers.GlobalMaxPooling2D()(base_model.output)
    x = layers.Concatenate()([gap, gmp])

    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=lr, amsgrad=True),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model

def get_callbacks(name, lr_patience=4):
    return [
        ModelCheckpoint(f'{name}_best.keras', monitor='val_accuracy',
                        save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.4,
                          patience=lr_patience, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=10,
                      restore_best_weights=True, verbose=1)
    ]

# Build Model 1: DenseNet121
print("Building DenseNet121...")
base_dense = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_densenet = build_model(base_dense, lr=1e-3)
print(f"  DenseNet121 ready. Total layers: {len(base_dense.layers)}")

# Build Model 2: InceptionV3
print("Building InceptionV3...")
base_inc = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
model_inception = build_model(base_inc, lr=1e-3)
print(f"  InceptionV3 ready. Total layers: {len(base_inc.layers)}")

# Build Model 3: ConvNeXtSmall
# Phase 1 LR=5e-4 (lower than others): ConvNeXt LayerNorm is sensitive to large gradient updates
print("Building ConvNeXtSmall (CVPR 2022 - Meta AI, 50M params)...")
base_convnext = ConvNeXtSmall(weights='imagenet', include_top=False,
                               input_shape=(224, 224, 3), include_preprocessing=False)
model_convnext = build_model(base_convnext, lr=5e-4)  # Conservative LR for ConvNeXt head warmup
print(f"  ConvNeXtSmall ready. Total layers: {len(base_convnext.layers)}")

print("\n[INFO] All 3 Backbones initialized successfully.")
print("  Model 1: DenseNet121    (Dense feature reuse, Grad-CAM backbone)")
print("  Model 2: InceptionV3    (Multi-scale parallel convolutions, 299x299)")
print("  Model 3: ConvNeXtSmall  (CVPR 2022, 50M params, 7x7 depthwise, GELU)")
print("  Head: Dual Pooling (GlobalAvgPool + GlobalMaxPool) for peak feature capture")
