"""
src/convert_digits_npy_to_landmarks.py

Converts the ardamavi Sign Language Digits Dataset (raw 64x64 pixel
images in X.npy / Y.npy) into landmark rows appended to
data/isl_landmarks.csv, matching its existing column layout exactly
(auto-detected, same approach as collect_digits_data.py).

IMPORTANT — run this, then open data/digit_preview.png BEFORE running
preprocess.py / train_model.py. This dataset's one-hot label order has
had reported mismatches in some releases — the preview shows 2 sample
images per decoded digit so you can visually confirm "3" actually
looks like 3, etc. If any look wrong, tell me which digit is
mislabeled and we'll fix the index mapping before training on it.

Expects:
    data/Sign-language-digits-dataset/X.npy   (images)
    data/Sign-language-digits-dataset/Y.npy   (one-hot labels)

Run from project root:
    python src/convert_digits_npy_to_landmarks.py
"""

import os
import re
import csv
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DATASET_DIR = "data/Sign-language-digits-dataset"
CSV_PATH = "data/isl_landmarks.csv"
MODEL_PATH = "hand_landmarker.task"
DIGIT_TARGET_OFFSET = 26
PREVIEW_PATH = "data/digit_preview.png"

MISSING_HAND = [-1.0] * 63


def build_column_maps(header):
    col_map = {}
    patterns = [
        re.compile(r"^(left|right)_hand_([xyz])_(\d+)$"),
        re.compile(r"^(left|right)_hand_(\d+)_([xyz])$"),
    ]
    for col in header:
        for pat in patterns:
            m = pat.match(col)
            if m:
                g = m.groups()
                if g[1] in ("x", "y", "z"):
                    side, axis, idx = g[0], g[1], int(g[2])
                else:
                    side, idx, axis = g[0], int(g[1]), g[2]
                col_map[(side, axis, idx)] = col
                break
    return col_map


def landmarks_to_row(col_map, header, left, right, target):
    row = {c: 0.0 for c in header}
    row["target"] = target
    row["uses_two_hands"] = 1.0 if (left != MISSING_HAND and right != MISSING_HAND) else 0.0
    for side, flat in (("left", left), ("right", right)):
        for idx in range(21):
            for ax_i, axis in enumerate(("x", "y", "z")):
                col = col_map.get((side, axis, idx))
                if col:
                    row[col] = flat[idx * 3 + ax_i]
    return row


def to_uint8_rgb(img):
    img = np.array(img, dtype=np.float32)
    if img.max() <= 1.5:
        img = img * 255.0
    img = img.astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def save_preview(images_by_digit):
    """2 samples per digit, labeled, so you can eyeball-check the mapping."""
    cell = 120
    grid = np.zeros((cell * 2, cell * 10, 3), dtype=np.uint8)
    for digit in range(10):
        samples = images_by_digit.get(digit, [])[:2]
        for row_i, img in enumerate(samples):
            resized = cv2.resize(img, (cell, cell))
            y0 = row_i * cell
            x0 = digit * cell
            grid[y0:y0 + cell, x0:x0 + cell] = resized
            cv2.putText(grid, str(digit), (x0 + 5, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.imwrite(PREVIEW_PATH, grid)
    print(f"[convert_digits] saved {PREVIEW_PATH} — open it and check each column actually shows that digit")


def main():
    X = np.load(os.path.join(DATASET_DIR, "X.npy"))
    Y = np.load(os.path.join(DATASET_DIR, "Y.npy"))
    print(f"[convert_digits] X shape: {X.shape}, Y shape: {Y.shape}")

    labels = np.argmax(Y, axis=1)  # one-hot -> digit index

    with open(CSV_PATH, "r", newline="") as f:
        header = next(csv.reader(f))
    col_map = build_column_maps(header)
    if not col_map:
        print("ERROR: could not detect landmark columns in CSV header:", header)
        return

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.4,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    images_by_digit = {}
    written = 0
    skipped_no_hand = 0

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)

        for i in range(len(X)):
            digit = int(labels[i])
            rgb_img = to_uint8_rgb(X[i])
            images_by_digit.setdefault(digit, []).append(rgb_img)

            upscaled = cv2.resize(rgb_img, (256, 256), interpolation=cv2.INTER_CUBIC)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=upscaled)
            result = landmarker.detect(mp_image)

            left, right = MISSING_HAND, MISSING_HAND
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                handedness = result.handedness[0][0].category_name
                flat = []
                for lm in landmarks:
                    flat.extend([lm.x, lm.y, lm.z])
                if handedness == "Left":
                    left = flat
                else:
                    right = flat

                row = landmarks_to_row(col_map, header, left, right, DIGIT_TARGET_OFFSET + digit)
                writer.writerow(row)
                written += 1
            else:
                skipped_no_hand += 1

            if (i + 1) % 200 == 0:
                print(f"  processed {i + 1}/{len(X)}...")

    save_preview(images_by_digit)

    print(f"\n[convert_digits] done. Rows written: {written}, skipped (no hand detected): {skipped_no_hand}")
    print(f"[convert_digits] CHECK {PREVIEW_PATH} before running preprocess.py — "
          f"confirm each column's digit label matches what the images actually show.")

    landmarker.close()


if __name__ == "__main__":
    main()