"""
src/collect_digits_data.py

Records your own webcam samples for digits 0-9, appending rows to
data/isl_landmarks.csv in the EXACT same column layout the file
already uses — auto-detected from the file's own header, so this
works regardless of whether columns are named left_hand_x_0 or
left_hand_0_x.

target values: 26-35 map to digits 0-9 (26=0, 27=1, ..., 35=9) —
matches TARGET_TO_LABEL in the updated src/preprocess.py.

Uses the MediaPipe Tasks API (hand_landmarker.task) since mp.solutions
isn't available in this environment — same approach as preprocess_words.py.

Controls:
  Press 0-9  -> start recording that digit (every frame while active)
  SPACE      -> stop recording current digit
  ESC        -> quit

Run from project root:
    python src/collect_digits_data.py
"""

import os
import re
import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

CSV_PATH = "data/isl_landmarks.csv"
MODEL_PATH = "hand_landmarker.task"
DIGIT_TARGET_OFFSET = 26  # digit d -> target = 26 + d

MISSING_HAND = [-1.0] * 63


def build_column_maps(header):
    """Detects (side, axis, idx) -> column_name from whatever naming style
    the existing CSV header actually uses."""
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
    uses_two = 1.0 if (left != MISSING_HAND and right != MISSING_HAND) else 0.0
    row["uses_two_hands"] = uses_two

    for side, flat in (("left", left), ("right", right)):
        for idx in range(21):
            for ax_i, axis in enumerate(("x", "y", "z")):
                col = col_map.get((side, axis, idx))
                if col:
                    row[col] = flat[idx * 3 + ax_i]
    return row


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found — run this from the project root.")
        return

    with open(CSV_PATH, "r", newline="") as f:
        header = next(csv.reader(f))
    col_map = build_column_maps(header)
    if not col_map:
        print("ERROR: could not detect left_hand_*/right_hand_* columns in the CSV header.")
        print("Header was:", header)
        return
    print(f"[collect_digits] detected {len(col_map)} landmark columns, header has {len(header)} total columns")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    current_digit = None
    samples_collected = 0

    print("Press 0-9 to record that digit. SPACE to stop. ESC to quit.")

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            left, right = MISSING_HAND, MISSING_HAND
            if result.hand_landmarks:
                for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    flat = []
                    for lm in landmarks:
                        flat.extend([lm.x, lm.y, lm.z])
                    label = handedness[0].category_name  # "Left" | "Right"
                    if label == "Left":
                        left = flat
                    elif label == "Right":
                        right = flat

                for landmarks in result.hand_landmarks:
                    for lm in landmarks:
                        cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 3, (0, 165, 255), -1)

            if current_digit is not None and (left != MISSING_HAND or right != MISSING_HAND):
                row = landmarks_to_row(col_map, header, left, right, DIGIT_TARGET_OFFSET + current_digit)
                writer.writerow(row)
                samples_collected += 1

            status = f"Recording digit: {current_digit} ({samples_collected})" if current_digit is not None else "Idle"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Digit Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 32:
                current_digit = None
                samples_collected = 0
            elif chr(key).isdigit():
                current_digit = int(chr(key))
                samples_collected = 0

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()