"""
backend/predict_words.py

Loads the trained word-level LSTM (models/asl_word_lstm.h5) and exposes
WordPredictor.predict(sequence) for app.py to call.

Matches src/train_asl_words_model.py and src/collect_asl_words.py exactly:

    63 floats per frame = single hand, 21 landmarks * (x, y, z)
    Missing hand -> 63 values of 0.0 (zero-fill sentinel)
    30 frames per sequence

The frontend hook (useWordDetection.js) builds frames in this exact
shape and buffers 30 of them before sending.
"""

import json
import numpy as np
import tensorflow as tf

SEQ_LEN = 30
FEATURE_DIM = 63


class WordPredictor:
    def __init__(self, model_path, label_map_path, conf_threshold=0.4):
        self.conf_threshold = conf_threshold
        self.is_loaded = False

        try:
            self.model = tf.keras.models.load_model(model_path)
            with open(label_map_path, "r") as f:
                self.label_map = json.load(f)  # {"0": "hello", "1": "yes", "2": "no"}
            self.is_loaded = True
            print(f"[predict_words] model loaded from {model_path}, {len(self.label_map)} words")
        except Exception as e:
            print(f"[predict_words] WARNING: could not load word model ({e}). "
                  f"Word mode will return random words until "
                  f"models/asl_word_lstm.h5 and models/asl_word_label_map.json exist.")
            self.model = None
            self.label_map = {"0": "hello", "1": "yes", "2": "no"}

    def predict(self, sequence):
        """
        sequence: list of 30 frames, each a list of 63 floats
                  (exactly what useWordDetection.js buffers and sends)
        returns: (word: str, confidence: float) or (None, confidence) if below threshold
        """
        if self.model is None:
            import random
            word = random.choice(list(self.label_map.values()))
            return word, round(random.uniform(0.4, 0.95), 2)

        arr = np.array(sequence, dtype=np.float32)

        if arr.shape != (SEQ_LEN, FEATURE_DIM):
            raise ValueError(
                f"expected sequence shape ({SEQ_LEN}, {FEATURE_DIM}), got {arr.shape}"
            )

        x = arr.reshape(1, SEQ_LEN, FEATURE_DIM)
        preds = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

        if confidence < self.conf_threshold:
            return None, round(confidence, 2)

        word = self.label_map.get(str(idx), "?")
        return word, round(confidence, 2)