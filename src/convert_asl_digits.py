"""
src/convert_asl_digits.py

Converts the ASL Digits dataset (data/asl_digits_raw/<0-9>/*.jpg) into
landmark rows APPENDED to the existing data/asl_landmarks.csv (single-hand,
same 64-column format as the letters), and extends models/asl_label_map.json
with new digit classes continuing after the existing letter indices.
"""

import os
import csv
import json
import random
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DATASET_DIR = "data/asl_dataset_digits"
OUT_CSV = "data/asl_landmarks.csv"
LABEL_MAP_PATH = "models/asl_label_map.json"
MODEL_PATH = "hand_landmarker.task"
SAMPLES_PER_CLASS = 500
SEED = 42

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
)


def extract_hand_features(image_path, landmarker):
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    flat = []
    for lm in result.hand_landmarks[0]:
        flat.extend([lm.x, lm.y, lm.z])
    return flat


def main():
    random.seed(SEED)

    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    next_idx = max(int(k) for k in label_map.keys()) + 1

    digits = sorted(
        [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))],
        key=lambda x: int(x)
    )

    new_entries = {}
    for digit in digits:
        new_entries[str(next_idx)] = digit
        next_idx += 1

    with open(OUT_CSV, "a", newline="") as f_out, \
         vision.HandLandmarker.create_from_options(options) as landmarker:

        writer = csv.writer(f_out)

        for target_str, digit in new_entries.items():
            target = int(target_str)
            digit_dir = os.path.join(DATASET_DIR, digit)
            images = [f for f in os.listdir(digit_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            random.shuffle(images)
            images = images[:SAMPLES_PER_CLASS]

            written, skipped = 0, 0
            for img_name in images:
                path = os.path.join(digit_dir, img_name)
                feat = extract_hand_features(path, landmarker)
                if feat is None:
                    skipped += 1
                    continue
                writer.writerow([target] + feat)
                written += 1

            print(f"[convert_asl_digits] '{digit}' -> target {target}: {written} written, {skipped} skipped")

    label_map.update(new_entries)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"[convert_asl_digits] appended to {OUT_CSV}, updated {LABEL_MAP_PATH} with {len(new_entries)} new classes")


if __name__ == "__main__":
    main()