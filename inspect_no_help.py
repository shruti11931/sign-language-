# inspect_no_help.py
import json
import numpy as np
import matplotlib.pyplot as plt

X = np.load("data/asl_words_X.npy")
y = np.load("data/asl_words_y.npy")

with open("models/asl_word_label_map.json") as f:
    label_map = json.load(f)

name_to_idx = {v: int(k) for k, v in label_map.items()}
no_idx = name_to_idx["no"]
help_idx = name_to_idx["help"]

no_clips = X[y == no_idx]
help_clips = X[y == help_idx]

# Plot wrist x-position (feature index 0) over time for several "no" clips,
# to check whether they look consistent with each other.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].set_title(f"'no' clips - wrist x over time (n={len(no_clips)})")
for i in range(min(15, len(no_clips))):
    axes[0].plot(no_clips[i][:, 0], alpha=0.5)
axes[0].set_xlabel("frame")
axes[0].set_ylabel("wrist x")

axes[1].set_title("'no' vs 'help' - average wrist trajectory")
no_mean = no_clips[:, :, :2].mean(axis=0)  # avg x,y over all no clips
help_mean = help_clips[:, :, :2].mean(axis=0)
axes[1].plot(no_mean[:, 0], no_mean[:, 1], label="no (avg)", marker="o", markersize=3)
axes[1].plot(help_mean[:, 0], help_mean[:, 1], label="help (avg)", marker="o", markersize=3)
axes[1].invert_yaxis()
axes[1].legend()
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

plt.tight_layout()
plt.savefig("no_help_inspection.png", dpi=120)
print("saved no_help_inspection.png")

# Also print numeric spread within "no" clips - high variance = inconsistent signing
print(f"\n'no' clips wrist-x std across clips (per frame, averaged): {no_clips[:, :, 0].std(axis=0).mean():.4f}")
print(f"'help' clips wrist-x std across clips (per frame, averaged): {help_clips[:, :, 0].std(axis=0).mean():.4f}")