"""
src/convert_images_to_landmarks.py

Converts a public image dataset (class-per-folder, like the Kaggle
ASL Alphabet dataset) into the SAME landmark CSV format collect_data.py
produces:

    label, x0,y0,z0, x1,y1,z1, ..., x20,y20,z20

This appends to data/raw_landmarks/landmarks.csv — the same file your
own webcam samples go into — so public + own data end up merged in one
place, ready for preprocess.py.

Expects a folder structure like:
    data/public_dataset/asl_alphabet_train/
        A/  *.jpg
        B/  *.jpg
        ...
        Z/  *.jpg
        space/  del/  nothing/   <- automatically skipped, not real classes

Only a subsample of images per class is processed (default 600) —
MediaPipe on the full ~87,000 images would take a long time and you
don't need that many for a good result.

Run:
    python src/convert_images_to_landmarks.py
"""

import os
import csv
import random
import cv2
import mediapipe as mp

DATASET_DIR = "data/public_dataset/asl_alphabet_train"
OUTPUT_CSV = "data/raw_landmarks/landmarks.csv"
SAMPLES_PER_CLASS = 600
SKIP_FOLDERS = {"space", "del", "nothing"}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

total_written = 0
total_skipped_no_hand = 0

with open(OUTPUT_CSV, "a", newline="") as f:
    writer = csv.writer(f)

    class_folders = sorted(
        d for d in os.listdir(DATASET_DIR)
        if d not in SKIP_FOLDERS and os.path.isdir(os.path.join(DATASET_DIR, d))
    )

    for label in class_folders:
        folder_path = os.path.join(DATASET_DIR, label)
        images = os.listdir(folder_path)
        random.shuffle(images)
        images = images[:SAMPLES_PER_CLASS]

        written_for_class = 0
        for img_name in images:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                row = [label]
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                writer.writerow(row)
                written_for_class += 1
                total_written += 1
            else:
                total_skipped_no_hand += 1

        print(f"{label}: {written_for_class}/{len(images)} images had a detectable hand")

print(f"\nDone. Total landmark rows written: {total_written}")
print(f"Images skipped (no hand detected): {total_skipped_no_hand}")
print(f"Saved to: {OUTPUT_CSV}")