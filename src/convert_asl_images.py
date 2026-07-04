"""
src/convert_asl_images.py

Converts the ASL Alphabet image dataset (data/asl_alphabet_train/<letter>/*.jpg)
into a single-hand landmark CSV. ASL fingerspelling uses one hand only, so
each row is: target, hand_x_0, hand_y_0, hand_z_0, ..., hand_x_20, hand_y_20, hand_z_20
(1 + 63 = 64 columns total).

Only SAMPLES_PER_CLASS images per letter are used (dataset has ~3000 each,
far more than needed for training).
"""

import os
import csv
import json
import random
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DATASET_DIR = "data/asl_alphabet_train"
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

    landmarks = result.hand_landmarks[0]
    flat = []
    for lm in landmarks:
        flat.extend([lm.x, lm.y, lm.z])
    return flat


def build_dataset():
    random.seed(SEED)
    letters = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    label_map = {str(i): letter for i, letter in enumerate(letters)}

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)

    with open(OUT_CSV, "w", newline="") as f_out, \
         vision.HandLandmarker.create_from_options(options) as landmarker:

        writer = csv.writer(f_out)
        header = ["target"] + [f"hand_{axis}_{i}" for i in range(21) for axis in ("x", "y", "z")]
        writer.writerow(header)

        for idx, letter in enumerate(letters):
            letter_dir = os.path.join(DATASET_DIR, letter)
            images = [f for f in os.listdir(letter_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            random.shuffle(images)
            images = images[:SAMPLES_PER_CLASS]

            written, skipped = 0, 0
            for img_name in images:
                path = os.path.join(letter_dir, img_name)
                feat = extract_hand_features(path, landmarker)
                if feat is None:
                    skipped += 1
                    continue
                writer.writerow([idx] + feat)
                written += 1

            print(f"[convert_asl] {letter}: {written} written, {skipped} skipped (no hand detected)")

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"[convert_asl] saved {OUT_CSV} and {LABEL_MAP_PATH}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    build_dataset()