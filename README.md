# 🤟 Sign to Speech — ASL Translator

A real-time American Sign Language (ASL) translator that turns hand gestures into live text and speech. The project now supports both fingerspelling and word-level recognition, with a polished React dashboard, real-time webcam tracking, and a Flask + Socket.IO backend.

---

## ✨ What’s New

- 🎥 A modern webcam-based UI with live camera feedback and a light/dark theme toggle
- 🔤 Two modes: "Letters & Digits" for fingerspelling and "Words" for sequence-based word prediction
- 🧠 Word-mode recognition using a rolling 30-frame landmark buffer sent to an LSTM model
- 📊 Live confidence indicators, buffer progress, recent logs, and sentence-level caption controls
- 🔊 Speak, undo, and clear actions for both letter and word captions
- 🌐 Real-time communication between the browser and backend through Socket.IO

---

## 🧩 Features

- Real-time hand tracking with MediaPipe Hands
- ASL alphabet and digit recognition for single-sign letters/digits
- Word-level prediction from short hand gesture sequences
- Confidence-based smoothing to reduce flickery predictions
- Auto-building caption with undo/clear support
- Text-to-speech playback for the current translated sentence
- Optional stub/demo mode when trained models are not yet available

---

## 🏗️ Architecture

The app uses a two-path pipeline:

1. The browser captures webcam frames and runs MediaPipe Hands locally.
2. Landmark coordinates are streamed to the Flask backend over Socket.IO.
3. The backend runs:
   - letter/digit inference through a dense classifier for single-frame predictions
   - word inference through a sequence model for 30-frame windows
4. The React dashboard updates the live prediction, confidence, caption, and recent activity log.

```text
Webcam → MediaPipe Hands → Socket.IO → Flask backend
                                      ├─ Letter/Digit MLP
                                      └─ Word LSTM
                                              ↓
                                    React dashboard UI
```

---

## 📁 Project Structure

```text
sign language/
├── backend/
│   ├── app.py                  # Flask + Socket.IO server
│   ├── predict.py              # Single-sign letter/digit inference
│   ├── predict_words.py        # Word-sequence inference with LSTM
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   └── src/
│       ├── components/
│       │   └── SignLanguageDashboard.jsx
│       └── hooks/
│           ├── useSignDetection.js
│           └── useWordDetection.js
│
├── src/
│   ├── collect_asl_digits.py
│   ├── collect_asl_words.py
│   ├── convert_asl_images.py
│   ├── convert_asl_digits.py
│   ├── preprocess_asl.py
│   ├── preprocess_words.py
│   ├── train_asl_model.py
│   └── train_asl_words_model.py
│
├── models/                     # trained model files
├── data/                       # datasets and landmark data
├── hand_landmarker.task        # MediaPipe hand model file
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- A webcam

### 1. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install frontend dependencies

```bash
cd frontend
npm install
```

### 3. Download the MediaPipe hand model

```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### 4. Train or prepare the models

The trained model files are not committed to the repo. You can build them locally.

#### Letter/digit model

```bash
python src/convert_asl_images.py
python src/collect_asl_digits.py   # optional, if you want your own digit dataset
python src/preprocess_asl.py
python src/train_asl_model.py
```

#### Word model (optional)

```bash
python src/collect_asl_words.py    # optional, for collecting word sign data
python src/preprocess_words.py
python src/train_asl_words_model.py
```

This produces the model files under the models folder.

### 5. Run the app

```bash
# Terminal 1
cd backend
python app.py
# -> http://localhost:5000

# Terminal 2
cd frontend
npm run dev
# -> http://localhost:5173
```

Open the frontend URL, allow camera access, and start signing.

---

## 🧠 How Recognition Works

1. The browser detects hand landmarks with MediaPipe and streams them to the backend.
2. For letter/digit mode, the backend classifies the current frame with a dense neural network.
3. For word mode, the frontend builds a rolling 30-frame sequence and sends it to the backend.
4. The backend uses a word-level LSTM model to predict a likely word from that sequence.
5. The UI shows confidence, buffer progress, and appends confirmed outputs to the caption.

---

## ⚠️ Notes and Limitations

- J and Z are motion-based signs in real ASL, so they are less reliable in this static-image-based setup.
- Some hand shapes such as M and N can be harder for MediaPipe to track clearly.
- This project focuses on fingerspelling and short word recognition, not full sentence-level sign language understanding.
- Model files, datasets, and the MediaPipe task file are expected to be generated locally.

---

## 🛠️ Tech Stack

- Frontend: React, Vite
- Hand tracking: MediaPipe Hands
- Realtime transport: Socket.IO
- Backend: Flask, Flask-SocketIO
- ML: TensorFlow / Keras
- Data processing: NumPy, pandas, OpenCV, scikit-learn

---

## 📄 License

This project is intended for educational and portfolio use.