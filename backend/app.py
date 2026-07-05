"""
backend/app.py

Flask + Socket.IO server for the ASL fingerspelling translator.
Single hand, single frame per prediction.

Run with:
    python app.py
Server listens on http://localhost:5000
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO

from predict import Predictor
from predict_words import WordPredictor

app = Flask(__name__)
app.config["SECRET_KEY"] = "asl-translator-dev"
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    ping_timeout=60,      # how long to wait for a pong before giving up (default 20s)
    ping_interval=25,     # how often to ping (default 25s, fine to keep)
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "asl_model.h5")
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "asl_label_map.json")

predictor = Predictor(model_path=MODEL_PATH, label_map_path=LABEL_MAP_PATH)

WORD_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "asl_word_lstm.h5")
WORD_LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "asl_word_label_map.json")
word_predictor = WordPredictor(model_path=WORD_MODEL_PATH, label_map_path=WORD_LABEL_MAP_PATH)


def _validate_landmarks(landmarks):
    return isinstance(landmarks, list) and len(landmarks) == 21 and all(len(p) == 3 for p in landmarks)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": predictor.is_loaded})


@app.route("/predict", methods=["POST"])
def predict_rest():
    data = request.get_json(force=True)
    landmarks = data.get("landmarks")
    if not _validate_landmarks(landmarks):
        return jsonify({"error": "expected 'landmarks': 21 [x,y,z] points"}), 400

    letter, confidence = predictor.predict(landmarks)
    return jsonify({"letter": letter, "confidence": confidence})


@socketio.on("connect")
def handle_connect():
    print("[socket] client connected:", request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    print("[socket] client disconnected:", request.sid)


@socketio.on("frame")
def handle_frame(payload):
    # --- DEBUG: remove once fixed ---
   
    # --- end debug ---

    landmarks = payload.get("landmarks")
    if not _validate_landmarks(landmarks):
        socketio.emit("prediction_error", {"error": "bad landmarks payload"}, to=request.sid)
        return

    letter, confidence = predictor.predict(landmarks)
    socketio.emit("prediction", {"letter": letter, "confidence": confidence}, to=request.sid)
    
    
@socketio.on("word_sequence")
def handle_word_sequence(payload):
    """
    payload: { "sequence": [ [63 floats], ... x30 ] }
    Emits: "word_prediction" -> { "word": "hello", "confidence": 0.82 }
    """
    sequence = payload.get("sequence")
    if not sequence or len(sequence) != 30 or len(sequence[0]) != 63:
        socketio.emit("word_prediction_error", {"error": "expected 30-frame sequence of 63-value vectors"}, to=request.sid)
        return

    word, confidence = word_predictor.predict(sequence)
    socketio.emit("word_prediction", {"word": word, "confidence": confidence}, to=request.sid)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    
    socketio.run(app, host="0.0.0.0", port=port, debug=False, use_reloader=False)