"""
backend/predict.py

Loads the trained ASL MLP model once at server startup and exposes
Predictor.predict(hand) for app.py to call on every incoming frame.

Feature format (63 values, single hand only):
  [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]  -- 21 landmarks * 3 = 63
"""

import json
import numpy as np
import tensorflow as tf


class Predictor:
    def __init__(self, model_path, label_map_path, conf_threshold=0.55):
        self.conf_threshold = conf_threshold
        self.is_loaded = False

        try:
            self.model = tf.keras.models.load_model(model_path)
            with open(label_map_path, "r") as f:
                self.label_map = json.load(f)
            self.is_loaded = True
            print(f"[predict] model loaded from {model_path}, {len(self.label_map)} classes")
        except Exception as e:
            print(f"[predict] WARNING: could not load model ({e}). Running in stub mode.")
            self.model = None
            self.label_map = {str(i): c for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}

    def predict(self, landmarks):
        """
        landmarks: list of 21 [x, y, z] points (single hand, MediaPipe normalized 0-1)
        returns: (letter: str, confidence: float)
        """
        if not landmarks:
            return None, 0.0

        if self.model is None:
            import random
            letter = random.choice(list(self.label_map.values()))
            return letter, round(random.uniform(0.5, 0.99), 2)

        flat = []
        for point in landmarks:
            flat.extend([float(point[0]), float(point[1]), float(point[2])])
        x = np.array(flat, dtype=np.float32).reshape(1, -1)

        preds = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        letter = self.label_map.get(str(idx), "?")
        return letter, round(confidence, 2)