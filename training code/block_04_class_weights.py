# ============================================================
# BLOCK 4: Compute Class Weights to handle imbalance
# ============================================================
labels_train = train_224.classes
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_train),
    y=labels_train
)
class_weight_dict = dict(enumerate(class_weights_arr))
print("Class weights:", class_weight_dict)
