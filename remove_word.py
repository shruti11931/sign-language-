# remove_word.py
import json
import numpy as np

WORD_TO_REMOVE = "yes"

X = np.load("data/asl_words_X.npy")
y = np.load("data/asl_words_y.npy")

with open("models/asl_word_label_map.json") as f:
    label_map = json.load(f)

name_to_idx = {v: int(k) for k, v in label_map.items()}
remove_idx = name_to_idx[WORD_TO_REMOVE]

# Drop all clips belonging to this word
mask = y != remove_idx
X = X[mask]
y = y[mask]

# Remove from label map and renumber remaining classes to a clean 0..N-1 range
remaining = sorted(i for i in name_to_idx.values() if i != remove_idx)
old_to_new = {old: new for new, old in enumerate(remaining)}

y = np.array([old_to_new[v] for v in y], dtype=np.int64)
new_label_map = {str(old_to_new[int(k)]): v for k, v in label_map.items() if v != WORD_TO_REMOVE}

np.save("data/asl_words_X.npy", X)
np.save("data/asl_words_y.npy", y)
with open("models/asl_word_label_map.json", "w") as f:
    json.dump(new_label_map, f, indent=2)

print(f"Removed '{WORD_TO_REMOVE}'. Remaining classes: {new_label_map}")
print(f"New X shape: {X.shape}")