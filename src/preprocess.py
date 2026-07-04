"""
src/preprocess.py

Loads the ISL hand-landmarks CSV and produces X.npy, y.npy, and
models/label_map.json for train_model.py.

CSV format (one row per sample):
  target          -> int, 0-25 = A-Z, 26-35 = 0-9 (digits appended later)
  uses_two_hands  -> float, 1.0 or 0.0
  left_hand_x_0..left_hand_z_20   (63 cols; -1.0 sentinel if that hand is absent)
  right_hand_x_0..right_hand_z_20 (63 cols; -1.0 sentinel if that hand is absent)

TARGET_TO_LABEL is the single source of truth for index -> class name.
If you add more classes later, extend this dict rather than relying on
chr() arithmetic — that only worked by coincidence for A-Z starting at 0.
"""

import json
import os
import numpy as np
import pandas as pd

CSV_PATH = "data/isl_landmarks.csv"
OUT_DIR = "data"

TARGET_TO_LABEL = {i: chr(ord("A") + i) for i in range(26)}          # 0-25  -> A-Z
TARGET_TO_LABEL.update({26 + i: str(i) for i in range(10)})           # 26-35 -> 0-9


def build_dataset(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)

    landmark_cols = [c for c in df.columns if c.startswith("left_hand") or c.startswith("right_hand")]
    feature_cols = ["uses_two_hands"] + landmark_cols  # 1 + 126 = 127 features

    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.int64)

    unique_targets = sorted(df["target"].unique().tolist())
    label_map = {}
    for t in unique_targets:
        if t not in TARGET_TO_LABEL:
            raise ValueError(
                f"target value {t} has no entry in TARGET_TO_LABEL — "
                f"add it there before running preprocess.py"
            )
        label_map[str(t)] = TARGET_TO_LABEL[t]

    return X, y, label_map


if __name__ == "__main__":
    X, y, label_map = build_dataset()
    print(f"[preprocess] X shape: {X.shape}, y shape: {y.shape}, classes: {len(label_map)}")
    print(f"[preprocess] label_map: {label_map}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save(f"{OUT_DIR}/X.npy", X)
    np.save(f"{OUT_DIR}/y.npy", y)
    with open("models/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("[preprocess] saved data/X.npy, data/y.npy, models/label_map.json")