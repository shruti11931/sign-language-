# check_word_confusion.py
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

X = np.load("data/asl_words_X.npy")
y = np.load("data/asl_words_y.npy")

with open("models/asl_word_label_map.json") as f:
    label_map = json.load(f)

model = tf.keras.models.load_model("models/asl_word_lstm.h5")

_, X_val, _, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
preds = np.argmax(model.predict(X_val, verbose=0), axis=1)

cm = confusion_matrix(y_val, preds)
labels = [label_map[str(i)] for i in sorted(set(y_val) | set(preds))]
print("Confusion matrix (rows=actual, cols=predicted):")
print("       " + "  ".join(f"{l:>10}" for l in labels))
for i, row in zip(sorted(set(y_val) | set(preds)), cm):
    print(f"{label_map[str(i)]:>6} " + "  ".join(f"{v:>10}" for v in row))