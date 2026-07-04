"""
src/collect_asl_digits.py

Live webcam collection tool for ASL digit signs. Appends samples to
data/asl_landmarks.csv (same single-hand, 63-feature format as your
letters) and extends models/asl_label_map.json with new digit classes.

Controls (webcam window must be focused):
  0-9   -> start/continue recording samples for that digit
  SPACE -> stop recording, reset counter (does not delete already-saved rows)
  ESC   -> quit

Only records a frame when a hand is actually detected with clean tracking.
Watch the orange skeleton overlay BEFORE pressing a digit key — if the
dots are scattered or not clearly on your hand, reposition first.
"""

import os
import csv
import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

OUT_CSV = "data/asl_landmarks.csv"
LABEL_MAP_PATH = "models/asl_label_map.json"
MODEL_PATH = "hand_landmarker.task"

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (61, 163, 232), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (61, 163, 232), -1)


def main():
    with open(LABEL_MAP_PATH) as f:
        label_map = json.load(f)

    name_to_idx = {v: int(k) for k, v in label_map.items()}
    next_idx = max(int(k) for k in label_map.keys()) + 1

    def get_or_create_target(digit_str):
        nonlocal next_idx
        if digit_str in name_to_idx:
            return name_to_idx[digit_str]
        idx = next_idx
        name_to_idx[digit_str] = idx
        label_map[str(idx)] = digit_str
        next_idx += 1
        return idx

    f_out = open(OUT_CSV, "a", newline="")
    writer = csv.writer(f_out)

    cap = cv2.VideoCapture(0)
    recording_digit = None
    sample_count = 0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        print("Press 0-9 to record that digit, SPACE to stop, ESC to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror for natural selfie view

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            hand_ok = False
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                draw_landmarks(frame, landmarks)
                hand_ok = True

                if recording_digit is not None:
                    flat = []
                    for lm in landmarks:
                        flat.extend([lm.x, lm.y, lm.z])
                    target = get_or_create_target(recording_digit)
                    writer.writerow([target] + flat)
                    sample_count += 1

            status = f"Recording digit: {recording_digit} ({sample_count})" if recording_digit else "Press 0-9 to start"
            color = (61, 163, 232) if recording_digit else (200, 200, 200)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if recording_digit and not hand_ok:
                cv2.putText(frame, "NO HAND DETECTED - not recording", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("ASL Digit Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                recording_digit = None
                sample_count = 0
            elif key < 256 and chr(key).isdigit():
                recording_digit = chr(key)
                sample_count = 0

    cap.release()
    cv2.destroyAllWindows()
    f_out.close()

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"[collect_asl_digits] done. Updated {LABEL_MAP_PATH}.")


if __name__ == "__main__":
    main()