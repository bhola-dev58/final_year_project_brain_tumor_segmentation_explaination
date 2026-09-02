# ============================================================
# BLOCK 8: Phase 3 - Full Backbone Fine-Tuning
# Deep fine-tuning to push accuracy above 98.0%.
# Key optimizations:
#   - Label smoothing: 0.02 (near-hard target distribution)
#   - Mixup DISABLED: direct augmented batches allow sharp boundary convergence
#   - ConvNeXtSmall LR=5e-6 (LayerNorm provides gradient stability)
#   - EarlyStopping patience=15: allows deep convergence without premature stopping
# ============================================================

def full_unfreeze(model, base_model, lr=3e-6):
    """Unfreeze all backbone layers with ultra-low learning rate."""
    base_model.trainable = True
    model.compile(
        optimizer=Adam(learning_rate=lr, amsgrad=True),
        loss=CategoricalCrossentropy(label_smoothing=0.02),
        metrics=['accuracy']
    )
    total_p = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"All layers unfrozen. Ultra-low LR = {lr} | Trainable params: {total_p:,}")
    return model

def get_callbacks_p3(name):
    """Phase 3 callbacks: higher patience for deeper convergence."""
    return [
        ModelCheckpoint(f'{name}_best.keras', monitor='val_accuracy',
                        save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.4,
                          patience=5, min_lr=1e-8, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=15,
                      restore_best_weights=True, verbose=1)
    ]

# Phase 3 uses direct augmented generators (Mixup disabled for sharp boundary learning)
# Independent generators per model prevent shared generator state issues
print("[INFO] Phase 3 uses direct augmented generators (no Mixup).")
train_224_p3, _ = make_generators((224, 224))
train_299_p3, _ = make_generators((299, 299))
train_224_cnx_p3, _ = make_generators((224, 224))

# Full unfreeze - DenseNet121
print("\n" + "=" * 60)
print("PHASE 3: DenseNet121 Full Unfreeze (Direct Batches, LR=3e-6)")
print("=" * 60)
model_densenet = full_unfreeze(model_densenet, base_dense, lr=3e-6)
h_dense_p3 = model_densenet.fit(
    train_224_p3, validation_data=val_224,
    steps_per_epoch=STEPS_224, epochs=EPOCHS_P3,
    callbacks=get_callbacks_p3('densenet_full')
)

# Full unfreeze - InceptionV3
print("\n" + "=" * 60)
print("PHASE 3: InceptionV3 Full Unfreeze (Direct Batches, LR=3e-6)")
print("=" * 60)
model_inception = full_unfreeze(model_inception, base_inc, lr=3e-6)
h_inc_p3 = model_inception.fit(
    train_299_p3, validation_data=val_299,
    steps_per_epoch=STEPS_299, epochs=EPOCHS_P3,
    callbacks=get_callbacks_p3('inception_full')
)

# Full unfreeze - ConvNeXtSmall
# LR=5e-6: ConvNeXt LayerNorm manages gradient variance internally
print("\n" + "=" * 60)
print("PHASE 3: ConvNeXtSmall Full Unfreeze (Direct Batches, LR=5e-6)")
print("  Expected standalone accuracy: 97.5% to 98.5%")
print("=" * 60)
model_convnext = full_unfreeze(model_convnext, base_convnext, lr=5e-6)
h_cnx_p3 = model_convnext.fit(
    train_224_cnx_p3, validation_data=val_224,
    steps_per_epoch=STEPS_224, epochs=EPOCHS_P3,
    callbacks=get_callbacks_p3('convnext_full')
)

print("\n[INFO] Phase 3 Complete. Models deeply tuned.")
print(f"  DenseNet121   best val_acc: {max(h_dense_p3.history['val_accuracy']):.4f}")
print(f"  InceptionV3   best val_acc: {max(h_inc_p3.history['val_accuracy']):.4f}")
print(f"  ConvNeXtSmall best val_acc: {max(h_cnx_p3.history['val_accuracy']):.4f}")
