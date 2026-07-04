"""
src/train_model.py

Trains a dense (MLP) classifier on ISL hand-landmark features.
No CNN / skeleton-image conversion needed — the data is already
numeric landmark coordinates, so a dense network is the right fit.
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

X = np.load("data/X.npy")
y = np.load("data/y.npy")

with open("models/label_map.json") as f:
    label_map = json.load(f)

num_classes = len(label_map)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
]

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=64,
    callbacks=callbacks,
)

val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"[train] final validation accuracy: {val_acc:.4f}")

os.makedirs("models", exist_ok=True)
model.save("models/gesture_cnn.h5")  # keeping this filename so app.py doesn't need edits yet
print("[train] saved models/gesture_cnn.h5")