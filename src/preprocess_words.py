"""
src/preprocess_words.py (Tasks API version)

Requires a hand_landmarker.task model file in the project root
(already downloaded — hand_landmarker.task, 7.8MB).
"""

import os
import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

DATASET_DIR = "data/include_words"
OUT_DIR = "data"
SEQ_LEN = 30
MISSING_HAND = [-1.0] * 63
MIN_CLIPS_PER_WORD = 18  # skip words with fewer clips than this — not enough data to learn from
MODEL_PATH = "hand_landmarker.task"

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
)


def extract_frame_features(frame, landmarker):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    left = MISSING_HAND
    right = MISSING_HAND

    if result.hand_landmarks:
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            flat = []
            for lm in landmarks:
                flat.extend([lm.x, lm.y, lm.z])
            label = handedness[0].category_name  # "Left" or "Right"
            if label == "Left":
                left = flat
            elif label == "Right":
                right = flat

    uses_two_hands = 1.0 if (left != MISSING_HAND and right != MISSING_HAND) else 0.0
    return [uses_two_hands] + left + right


def video_to_sequence(video_path, landmarker):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, SEQ_LEN).astype(int)
    idx_set = set(indices.tolist())

    last_valid = [0.0] + MISSING_HAND + MISSING_HAND
    grabbed = {}
    frame_no = 0

    while cap.isOpened() and frame_no <= max(idx_set):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no in idx_set:
            feat = extract_frame_features(frame, landmarker)
            if feat[1:64] == MISSING_HAND and feat[64:] == MISSING_HAND:
                feat = last_valid
            else:
                last_valid = feat
            grabbed[frame_no] = feat
        frame_no += 1

    cap.release()

    if not grabbed:
        return None

    sequence = [grabbed.get(i, last_valid) for i in indices]
    return np.array(sequence, dtype=np.float32)


def build_dataset(dataset_dir=DATASET_DIR):
    all_words = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

    # Only keep words with enough clips to actually be learnable
    words = []
    for w in all_words:
        word_dir = os.path.join(dataset_dir, w)
        count = len([f for f in os.listdir(word_dir) if f.lower().endswith((".mov", ".mp4", ".avi"))])
        if count >= MIN_CLIPS_PER_WORD:
            words.append(w)
        else:
            print(f"[preprocess_words] skipping '{w}' — only {count} clips (< {MIN_CLIPS_PER_WORD})")

    print(f"[preprocess_words] keeping {len(words)}/{len(all_words)} words")

    label_map = {str(i): w for i, w in enumerate(words)}
    word_to_idx = {w: i for i, w in label_map.items()}

    X, y = [], []

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for word in words:
            word_dir = os.path.join(dataset_dir, word)
            video_files = [f for f in os.listdir(word_dir) if f.lower().endswith((".mov", ".mp4", ".avi"))]
            print(f"[preprocess_words] {word}: {len(video_files)} clips")

            for vf in video_files:
                path = os.path.join(word_dir, vf)
                seq = video_to_sequence(path, landmarker)
                if seq is None:
                    print(f"  skipped (unreadable/no frames): {vf}")
                    continue
                X.append(seq)
                y.append(int(word_to_idx[word]))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, label_map
    


if __name__ == "__main__":
    X, y, label_map = build_dataset()
    print(f"[preprocess_words] X shape: {X.shape}, y shape: {y.shape}, classes: {len(label_map)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save(f"{OUT_DIR}/words_X.npy", X)
    np.save(f"{OUT_DIR}/words_y.npy", y)
    with open("models/word_label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("[preprocess_words] saved data/words_X.npy, data/words_y.npy, models/word_label_map.json")