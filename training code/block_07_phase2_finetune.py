# ============================================================
# BLOCK 7: Phase 2 — Unfreeze top 60 layers and fine-tune
# This is the KEY step that gets you from 78% to 95%+
# ============================================================

def unfreeze_top(model, base_model, n_layers=60, lr=1e-5):
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False
    for layer in base_model.layers[-n_layers:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    total_trainable = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"Unfrozen last {n_layers} layers. Total trainable params: {total_trainable:,}")
    return model

# Fine-tune DenseNet121
print("=" * 60)
print("PHASE 2 — DenseNet121 Fine-Tuning")
print("=" * 60)
model_densenet = unfreeze_top(model_densenet, base_dense, n_layers=60, lr=1e-5)
h_dense_p2 = model_densenet.fit(
    train_224, validation_data=val_224, epochs=EPOCHS_P2,
    class_weight=class_weight_dict,
    callbacks=get_callbacks('densenet_ft', lr_patience=5)
)

# Fine-tune InceptionV3
print("\n" + "=" * 60)
print("PHASE 2 — InceptionV3 Fine-Tuning")
print("=" * 60)
model_inception = unfreeze_top(model_inception, base_inc, n_layers=60, lr=1e-5)
h_inc_p2 = model_inception.fit(
    train_299, validation_data=val_299, epochs=EPOCHS_P2,
    class_weight=class_weight_dict,
    callbacks=get_callbacks('inception_ft', lr_patience=5)
)

# Fine-tune EfficientNetV2S
print("\n" + "=" * 60)
print("PHASE 2 — EfficientNetV2S Fine-Tuning")
print("=" * 60)
model_effnet = unfreeze_top(model_effnet, base_eff, n_layers=60, lr=1e-5)
h_eff_p2 = model_effnet.fit(
    train_224, validation_data=val_224, epochs=EPOCHS_P2,
    class_weight=class_weight_dict,
    callbacks=get_callbacks('effnet_ft', lr_patience=5)
)

print("\nPhase 2 Complete!")
