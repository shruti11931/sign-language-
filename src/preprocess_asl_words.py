"""
src/preprocess_asl_words.py
Just validates and reports on the data collected by collect_asl_words.py —
no transformation needed since it's already saved in the right shape.
"""
import json
import numpy as np

X = np.load("data/asl_words_X.npy")
y = np.load("data/asl_words_y.npy")

with open("models/asl_word_label_map.json") as f:
    label_map = json.load(f)

print(f"X shape: {X.shape}, y shape: {y.shape}, classes: {len(label_map)}")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {label_map[str(u)]}: {c} clips")