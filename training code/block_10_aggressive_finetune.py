# ============================================================
# BLOCK 10: Aggressive Fine-Tuning — Unfreeze ALL layers
# Run this AFTER Block 7 to push accuracy above 98%
# Then re-run Block 8 and Block 9 with updated model names
# ============================================================

def full_unfreeze(model, base_model, lr=3e-6):
    """Unfreeze ALL layers with very low learning rate"""
    base_model.trainable = True
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    print(f"All layers unfrozen. LR={lr}")
    return model

# Full unfreeze — DenseNet121
print("=" * 60)
print("PHASE 3 — DenseNet121 Full Unfreeze")
print("=" * 60)
model_densenet = full_unfreeze(model_densenet, base_dense, lr=3e-6)
h_dense_p3 = model_densenet.fit(
    train_224, validation_data=val_224, epochs=40,
    class_weight=class_weight_dict,
    callbacks=get_callbacks('densenet_full', lr_patience=6)
)

# Full unfreeze — InceptionV3
print("\n" + "=" * 60)
print("PHASE 3 — InceptionV3 Full Unfreeze")
print("=" * 60)
model_inception = full_unfreeze(model_inception, base_inc, lr=3e-6)
h_inc_p3 = model_inception.fit(
    train_299, validation_data=val_299, epochs=40,
    class_weight=class_weight_dict,
    callbacks=get_callbacks('inception_full', lr_patience=6)
)

# Full unfreeze — EfficientNetV2S
print("\n" + "=" * 60)
print("PHASE 3 — EfficientNetV2S Full Unfreeze")
print("=" * 60)
model_effnet = full_unfreeze(model_effnet, base_eff, lr=3e-6)
h_eff_p3 = model_effnet.fit(
    train_224, validation_data=val_224, epochs=40,
    class_weight=class_weight_dict,
    callbacks=get_callbacks('effnet_full', lr_patience=6)
)

print("\nPhase 3 Complete!")
print("Now run Block 8 with these model names:")
print("  densenet_full_best.keras")
print("  inception_full_best.keras")
print("  effnet_full_best.keras")
