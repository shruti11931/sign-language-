"""
src/train_asl_words_model.py
Trains an LSTM on word-level ASL landmark sequences (single hand, 30 frames, 63 features/frame).
"""
import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

X = np.load("data/asl_words_X.npy")
y = np.load("data/asl_words_y.npy")

with open("models/asl_word_label_map.json") as f:
    label_map = json.load(f)

num_classes = len(label_map)
print(f"X: {X.shape}, y: {y.shape}, classes: {num_classes}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.Masking(mask_value=0.0),
    tf.keras.layers.LSTM(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=6, factor=0.5),
]

classes, counts = np.unique(y_train, return_counts=True)
total = len(y_train)
class_weight = {int(c): total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}

model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=150, batch_size=16, callbacks=callbacks, class_weight=class_weight)

val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Final validation accuracy: {val_acc:.4f}")

os.makedirs("models", exist_ok=True)
model.save("models/asl_word_lstm.h5")
print("Saved models/asl_word_lstm.h5")