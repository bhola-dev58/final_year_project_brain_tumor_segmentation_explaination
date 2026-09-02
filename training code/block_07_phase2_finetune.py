# ============================================================
# BLOCK 7: Phase 2 - Unfreeze top layers and fine-tune with Mixup
# Transitions model performance towards target accuracy
# ConvNeXtSmall: unfreeze top 100 layers (modular deep stages)
# ============================================================

def unfreeze_top(model, base_model, n_layers=60, lr=1e-5):
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    for layer in base_model.layers[-n_layers:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=lr, amsgrad=True),
        loss=CategoricalCrossentropy(label_smoothing=0.04),
        metrics=['accuracy']
    )
    total_trainable = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"Unfrozen last {n_layers} layers. Total Trainable Parameters: {total_trainable:,}")
    return model

# Fresh independent Mixup generators for Phase 2
dense_mix_p2 = mixup_generator(make_generators((224, 224))[0], alpha=0.2)
inc_mix_p2   = mixup_generator(make_generators((299, 299))[0], alpha=0.2)
cnx_mix_p2   = mixup_generator(make_generators((224, 224))[0], alpha=0.2)

# Fine-tune DenseNet121
print("=" * 60)
print("PHASE 2: DenseNet121 Fine-Tuning (Mixup Enabled)")
print("=" * 60)
model_densenet = unfreeze_top(model_densenet, base_dense, n_layers=60, lr=1e-5)
h_dense_p2 = model_densenet.fit(
    dense_mix_p2, validation_data=val_224,
    steps_per_epoch=STEPS_224, epochs=EPOCHS_P2,
    callbacks=get_callbacks('densenet_p2', lr_patience=4)
)

# Fine-tune InceptionV3
print("\n" + "=" * 60)
print("PHASE 2: InceptionV3 Fine-Tuning (Mixup Enabled)")
print("=" * 60)
model_inception = unfreeze_top(model_inception, base_inc, n_layers=60, lr=1e-5)
h_inc_p2 = model_inception.fit(
    inc_mix_p2, validation_data=val_299,
    steps_per_epoch=STEPS_299, epochs=EPOCHS_P2,
    callbacks=get_callbacks('inception_p2', lr_patience=4)
)

# Fine-tune ConvNeXtSmall
# ConvNeXtSmall has deeper modular blocks - unfreezing top 100 layers
print("\n" + "=" * 60)
print("PHASE 2: ConvNeXtSmall Fine-Tuning (Mixup Enabled)")
print("  Unfreezing top 100 layers - hierarchical depthwise stages")
print("=" * 60)
model_convnext = unfreeze_top(model_convnext, base_convnext, n_layers=100, lr=1e-5)
h_cnx_p2 = model_convnext.fit(
    cnx_mix_p2, validation_data=val_224,
    steps_per_epoch=STEPS_224, epochs=EPOCHS_P2,
    callbacks=get_callbacks('convnext_p2', lr_patience=4)
)

print("\n[INFO] Phase 2 Training Complete.")
print(f"  DenseNet121   best val_acc: {max(h_dense_p2.history['val_accuracy']):.4f}")
print(f"  InceptionV3   best val_acc: {max(h_inc_p2.history['val_accuracy']):.4f}")
print(f"  ConvNeXtSmall best val_acc: {max(h_cnx_p2.history['val_accuracy']):.4f}")
