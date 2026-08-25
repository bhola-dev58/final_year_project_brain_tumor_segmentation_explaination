# ============================================================
# BLOCK 5: Model Builder with Label Smoothing Loss
# ============================================================

def build_model(base_model, num_classes=4, dropout=0.3):
    base_model.trainable = False  # Phase 1: freeze all

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x)
    x = layers.Dropout(dropout)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model

def get_callbacks(name, lr_patience=4):
    return [
        ModelCheckpoint(f'{name}_best.keras', monitor='val_accuracy',
                        save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                          patience=lr_patience, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=8,
                      restore_best_weights=True, verbose=1)
    ]

# Build all 3 models
print("Building DenseNet121...")
base_dense = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_densenet = build_model(base_dense)

print("Building InceptionV3...")
base_inc = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
model_inception = build_model(base_inc)

print("Building EfficientNetV2S...")
base_eff = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model_effnet = build_model(base_eff)

print("All 3 models built successfully!")
