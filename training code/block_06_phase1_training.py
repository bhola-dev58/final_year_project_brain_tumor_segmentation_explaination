# ============================================================
# BLOCK 6: Phase 1 - Train classifier head only (frozen backbone)
# All 3 CNN backbones frozen, only classification heads train
# Uses Mixup data augmentation for smooth decision boundaries
# ============================================================

STEPS_224 = train_224.samples // BATCH_SIZE
STEPS_299 = train_299.samples // BATCH_SIZE

# Independent Mixup generators per model to prevent state conflicts
dense_mix_p1  = mixup_generator(make_generators((224, 224))[0], alpha=0.2)
inc_mix_p1    = mixup_generator(make_generators((299, 299))[0], alpha=0.2)
cnx_mix_p1    = mixup_generator(make_generators((224, 224))[0], alpha=0.2)

print("=" * 60)
print("PHASE 1: DenseNet121 (Frozen Backbone, Mixup Enabled, LR=1e-3)")
print("=" * 60)
h_dense_p1 = model_densenet.fit(
    dense_mix_p1, validation_data=val_224,
    steps_per_epoch=STEPS_224, epochs=EPOCHS_P1,
    callbacks=get_callbacks('densenet_p1')
)

print("\n" + "=" * 60)
print("PHASE 1: InceptionV3 (Frozen Backbone, Mixup Enabled, LR=1e-3)")
print("=" * 60)
h_inc_p1 = model_inception.fit(
    inc_mix_p1, validation_data=val_299,
    steps_per_epoch=STEPS_299, epochs=EPOCHS_P1,
    callbacks=get_callbacks('inception_p1')
)

print("\n" + "=" * 60)
print("PHASE 1: ConvNeXtSmall (Frozen Backbone, Mixup Enabled, LR=1e-3)")
print("  CVPR 2022 - 50M params, 7x7 depthwise + GELU + LayerNorm")
print("=" * 60)
h_cnx_p1 = model_convnext.fit(
    cnx_mix_p1, validation_data=val_224,
    steps_per_epoch=STEPS_224, epochs=EPOCHS_P1,
    callbacks=get_callbacks('convnext_p1')
)

print("\n[INFO] Phase 1 Training Complete.")
print(f"  DenseNet121   best val_acc: {max(h_dense_p1.history['val_accuracy']):.4f}")
print(f"  InceptionV3   best val_acc: {max(h_inc_p1.history['val_accuracy']):.4f}")
print(f"  ConvNeXtSmall best val_acc: {max(h_cnx_p1.history['val_accuracy']):.4f}")

