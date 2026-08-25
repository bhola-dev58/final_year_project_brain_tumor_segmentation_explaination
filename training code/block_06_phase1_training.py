# ============================================================
# BLOCK 6: Phase 1 — Train classifier head only (frozen base)
# ============================================================

print("=" * 60)
print("PHASE 1 — DenseNet121 (frozen base)")
print("=" * 60)
h_dense_p1 = model_densenet.fit(
    train_224, validation_data=val_224, epochs=EPOCHS_P1,
    class_weight=class_weight_dict,
    callbacks=get_callbacks('densenet')
)

print("\n" + "=" * 60)
print("PHASE 1 — InceptionV3 (frozen base)")
print("=" * 60)
h_inc_p1 = model_inception.fit(
    train_299, validation_data=val_299, epochs=EPOCHS_P1,
    class_weight=class_weight_dict,
    callbacks=get_callbacks('inception')
)

print("\n" + "=" * 60)
print("PHASE 1 — EfficientNetV2S (frozen base)")
print("=" * 60)
h_eff_p1 = model_effnet.fit(
    train_224, validation_data=val_224, epochs=EPOCHS_P1,
    class_weight=class_weight_dict,
    callbacks=get_callbacks('effnet')
)

print("\nPhase 1 Complete!")
