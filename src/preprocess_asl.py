"""
src/preprocess_asl.py

Loads data/asl_landmarks.csv (single-hand, 64 columns: target + 63 landmark
values) and produces X.npy, y.npy for train_model.py.
"""

import json
import os
import numpy as np
import pandas as pd

CSV_PATH = "data/asl_landmarks.csv"
OUT_DIR = "data"
LABEL_MAP_PATH = "models/asl_label_map.json"


def build_dataset(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)

    landmark_cols = [c for c in df.columns if c.startswith("hand_")]
    X = df[landmark_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.int64)

    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    return X, y, label_map


if __name__ == "__main__":
    X, y, label_map = build_dataset()
    print(f"[preprocess_asl] X shape: {X.shape}, y shape: {y.shape}, classes: {len(label_map)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(f"{OUT_DIR}/asl_X.npy", X)
    np.save(f"{OUT_DIR}/asl_y.npy", y)

    print(f"[preprocess_asl] saved {OUT_DIR}/asl_X.npy, {OUT_DIR}/asl_y.npy")