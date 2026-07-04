"""
src/train_words_model.py

Trains an LSTM classifier on word-level ISL landmark sequences.
Includes landmark-space data augmentation (rotation, scale, translation
jitter, and left/right mirroring) to multiply the small dataset, since
raw clip counts per word are too low (~18-22) to train on directly.
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42
rng = np.random.default_rng(SEED)

X = np.load("data/words_X.npy")   # (num_videos, 30, 127)
y = np.load("data/words_y.npy")   # (num_videos,)

with open("models/word_label_map.json") as f:
    label_map = json.load(f)

num_classes = len(label_map)
print(f"[train_words] raw X: {X.shape}, y: {y.shape}, classes: {num_classes}")


def jitter_sequence(seq, rng):
    """Small random rotation + scale + translation applied to landmark x,y (per hand),
    leaving -1.0 sentinel (missing hand) frames untouched."""
    seq = seq.copy()
    angle = rng.uniform(-0.15, 0.15)  # ~ +/- 8.6 degrees
    scale = rng.uniform(0.9, 1.1)
    tx = rng.uniform(-0.03, 0.03)
    ty = rng.uniform(-0.03, 0.03)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    for offset in (1, 64):  # left hand starts at col 1, right hand at col 64
        hand = seq[:, offset:offset + 63].reshape(seq.shape[0], 21, 3)
        x, y_, z = hand[..., 0], hand[..., 1], hand[..., 2]
        mask = x != -1.0  # only touch frames where this hand was actually detected

        xr, yr = x - 0.5, y_ - 0.5
        x_new = (xr * cos_a - yr * sin_a) * scale + 0.5 + tx
        y_new = (xr * sin_a + yr * cos_a) * scale + 0.5 + ty

        hand[..., 0] = np.where(mask, x_new, x)
        hand[..., 1] = np.where(mask, y_new, y_)
        seq[:, offset:offset + 63] = hand.reshape(seq.shape[0], 63)

    return seq


def mirror_sequence(seq):
    """Flip left<->right hands and negate x, simulating the signer's mirror image."""
    seq = seq.copy()
    left = seq[:, 1:64].copy()
    right = seq[:, 64:127].copy()

    def flip_x(hand):
        pts = hand.reshape(hand.shape[0], 21, 3)
        x = pts[..., 0]
        mask = x != -1.0
        pts[..., 0] = np.where(mask, 1.0 - x, x)
        return pts.reshape(hand.shape[0], 63)

    seq[:, 1:64] = flip_x(right)
    seq[:, 64:127] = flip_x(left)
    return seq


def augment_dataset(X, y, jitter_copies=3, include_mirror=True):
    """Expands the dataset: keeps every original sample, adds `jitter_copies`
    randomly-perturbed versions of each, plus one mirrored version if requested."""
    aug_X, aug_y = [X], [y]

    for _ in range(jitter_copies):
        jittered = np.stack([jitter_sequence(seq, rng) for seq in X])
        aug_X.append(jittered)
        aug_y.append(y)

    if include_mirror:
        mirrored = np.stack([mirror_sequence(seq) for seq in X])
        aug_X.append(mirrored)
        aug_y.append(y)
        # one jittered mirror too, for extra variety
        jittered_mirror = np.stack([jitter_sequence(seq, rng) for seq in mirrored])
        aug_X.append(jittered_mirror)
        aug_y.append(y)

    return np.concatenate(aug_X, axis=0), np.concatenate(aug_y, axis=0)


X_aug, y_aug = augment_dataset(X, y, jitter_copies=3, include_mirror=True)
print(f"[train_words] augmented X: {X_aug.shape}, y: {y_aug.shape} (was {X.shape[0]} samples, now {X_aug.shape[0]})")

X_train, X_val, y_train, y_val = train_test_split(
    X_aug, y_aug, test_size=0.15, random_state=SEED, stratify=y_aug
)

classes, counts = np.unique(y_train, return_counts=True)
total = len(y_train)
class_weight = {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}

# Leaner model than before — fewer parameters, more regularization, to fit this data size
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.Masking(mask_value=0.0),
    tf.keras.layers.LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5),
]

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    class_weight=class_weight,
    callbacks=callbacks,
)

val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"[train_words] final validation accuracy: {val_acc:.4f}")

os.makedirs("models", exist_ok=True)
model.save("models/word_lstm.h5")
print("[train_words] saved models/word_lstm.h5")