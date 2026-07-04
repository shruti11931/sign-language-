import cv2
import mediapipe as mp
import csv
import os
import time

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("0123456789")
SAVE_DIR = "data/raw_landmarks"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
current_label = None
samples_collected = 0

print("Press a key (A-Z, 0-9) to start recording that label.")
print("Press SPACE to stop recording. Press ESC to quit.")

with open(os.path.join(SAVE_DIR, "landmarks.csv"), "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if current_label is not None:
                row = [current_label]
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                writer.writerow(row)
                samples_collected += 1

        status = f"Recording: {current_label} ({samples_collected})" if current_label else "Idle"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            current_label = None
            samples_collected = 0
        elif chr(key).upper() in LABELS:
            current_label = chr(key).upper()
            samples_collected = 0

cap.release()
cv2.destroyAllWindows()