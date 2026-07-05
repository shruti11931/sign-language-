"""
src/collect_asl_words.py

Live webcam collection tool for word-level ASL signs (motion, not static poses).
Saves incrementally after EVERY clip (not just at the end) so a crash or
accidental close never loses a session's data again.

Controls:
  type a word name + ENTER, then confirm  -> set the current word to record
  SPACE (hold)                            -> record a clip while held (30 frames)
  ESC                                     -> quit (everything is already saved)
"""

import os
import json
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

OUT_X = "data/asl_words_X.npy"
OUT_Y = "data/asl_words_y.npy"
LABEL_MAP_PATH = "models/asl_word_label_map.json"
MODEL_PATH = "hand_landmarker.task"
SEQ_LEN = 30
MISSING_HAND = [0.0] * 63

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


def extract_features(frame, landmarker):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None, None
    landmarks = result.hand_landmarks[0]
    flat = []
    for lm in landmarks:
        flat.extend([lm.x, lm.y, lm.z])
    return flat, landmarks


def load_existing():
    if os.path.exists(OUT_X) and os.path.exists(OUT_Y):
        return list(np.load(OUT_X)), list(np.load(OUT_Y))
    return [], []


def save_all(all_X, all_y, label_map):
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    np.save(OUT_X, np.array(all_X, dtype=np.float32))
    np.save(OUT_Y, np.array(all_y, dtype=np.int64))
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)


def get_confirmed_word():
    while True:
        word = input("Enter word to record: ").strip().lower()
        if not word:
            print("Empty input, try again.")
            continue
        confirm = input(f"Confirm word is '{word}'? (y/n): ").strip().lower()
        if confirm == "y":
            return word
        print("Okay, try again.")


def main():
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH) as f:
            label_map = json.load(f)
    else:
        label_map = {}

    name_to_idx = {v: int(k) for k, v in label_map.items()}
    next_idx = max([int(k) for k in label_map.keys()], default=-1) + 1

    all_X, all_y = load_existing()
    print(f"Loaded existing {len(all_X)} clips across {len(label_map)} words")

    current_word = None
    cap = cv2.VideoCapture(0)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        print("Press 'n' then click the camera window, hold SPACE to record, ESC to quit.")
        print("Data is saved to disk after every single clip - nothing is lost if you stop early.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            display = frame.copy()
            label = f"Word: {current_word or '(none set - press n in terminal)'}"
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (61, 163, 232), 2)
            cv2.putText(display, "Hold SPACE to record, ESC to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.imshow("ASL Word Collection", display)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            if key == 32 and current_word:  # SPACE -> record one clip
                clip_frames = []
                print(f"Recording '{current_word}'...")
                while len(clip_frames) < SEQ_LEN:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    feat, landmarks = extract_features(frame, landmarker)

                    live = frame.copy()
                    if landmarks:
                        draw_landmarks(live, landmarks)
                    cv2.putText(live, f"Recording {len(clip_frames)}/{SEQ_LEN}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("ASL Word Collection", live)
                    cv2.waitKey(1)

                    clip_frames.append(feat if feat else MISSING_HAND)

                if current_word not in name_to_idx:
                    name_to_idx[current_word] = next_idx
                    label_map[str(next_idx)] = current_word
                    next_idx += 1

                all_X.append(np.array(clip_frames, dtype=np.float32))
                all_y.append(name_to_idx[current_word])

                # Save immediately after every clip - crash-safe
                save_all(all_X, all_y, label_map)
                count_for_word = sum(1 for v in all_y if v == name_to_idx[current_word])
                print(f"Saved clip #{count_for_word} for '{current_word}' (written to disk)")

            if key == ord("n"):
                current_word = get_confirmed_word()

    cap.release()
    cv2.destroyAllWindows()

    save_all(all_X, all_y, label_map)
    print(f"Done. {len(all_X)} total clips across {len(label_map)} words (all saved).")


if __name__ == "__main__":
    main()