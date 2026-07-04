# 🤟 Sign to Speech — ASL Fingerspelling Interpreter

A real-time American Sign Language (ASL) fingerspelling translator that turns hand signs into live text — and speaks them aloud. Built with a browser-based webcam pipeline, MediaPipe hand tracking, and a custom-trained neural network, all running end-to-end from camera to caption in real time.

---

## ✨ Features

- 🎥 **Real-time webcam hand tracking** via MediaPipe Hands, running entirely in the browser
- 🔤 **Full ASL alphabet (A–Z)** plus **digits (0–9)** and two control gestures: `space` and `del`
- 📊 **Live confidence score** displayed per prediction, with a stability filter to avoid flickery false triggers
- 💬 **Auto-building translated caption**, with undo/clear controls
- 🔊 **Text-to-speech playback** of the translated sentence
- 🌗 **Light/dark theme toggle**
- ✅ **~98% validation accuracy** on the trained model across 39 classes

---

## 🏗️ Architecture

**Pipeline:**

1. **Webcam** (browser) captures live video
2. **MediaPipe Hands** extracts 21 hand landmarks `(x, y, z)` per frame
3. **Socket.IO** streams the landmarks to the backend in real time
4. **Flask backend** runs the landmarks through a trained **MLP model**
5. Backend returns the **predicted letter/digit + confidence** over the same socket
6. **React UI** updates the live caption, confidence bar, and gesture log

```
Webcam → MediaPipe Hands → Socket.IO → Flask backend (MLP model)
                                              |
                                              v
                                   predicted letter + confidence
                                              |
                                              v
                          React UI (caption, confidence bar, gesture log)
```

**Why an MLP instead of a CNN?** Early prototypes converted hand landmarks into synthetic "skeleton images" to feed a CNN — but since MediaPipe already outputs precise numeric (x, y, z) coordinates, that conversion was an unnecessary, lossy detour. A small dense neural network trained directly on the 63 landmark values (21 points × x/y/z) is simpler, faster to train, and more accurate.

---

## 📁 Project Structure

```
sign language/
├── backend/
│   ├── app.py                  # Flask + Socket.IO server
│   ├── predict.py              # Model loading + inference logic
│   └── requirements.txt
│
├── frontend/
│   ├── index.html              # MediaPipe CDN <script> tags live here
│   └── src/
│       ├── components/
│       │   └── SignLanguageDashboard.jsx
│       └── hooks/
│           └── useSignDetection.js
│
├── src/
│   ├── convert_asl_images.py   # Kaggle ASL alphabet images -> landmarks CSV
│   ├── convert_asl_digits.py   # Kaggle ASL digits images -> landmarks CSV
│   ├── collect_asl_digits.py   # Live webcam digit collection tool
│   ├── preprocess_asl.py       # CSV -> X.npy / y.npy
│   └── train_asl_model.py      # Trains the MLP classifier
│
├── models/                     # asl_model.h5, asl_label_map.json (gitignored)
├── data/                       # datasets + landmark CSVs (gitignored)
├── hand_landmarker.task        # MediaPipe hand model file (gitignored)
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (tested on 3.13)
- Node.js 18+
- A webcam

### 1. Clone and set up the backend

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set up the frontend

```bash
cd frontend
npm install
```

### 3. Download the MediaPipe hand landmark model (one-time)

```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

### 4. Train the model

The trained model isn't included in this repo (see [Known Limitations](#-known-limitations--notes)). To build it yourself:

```bash
# 1. Download the ASL Alphabet dataset and place it at data/asl_alphabet_train/
#    https://www.kaggle.com/datasets/grassknoted/asl-alphabet
python src/convert_asl_images.py

# 2. Add digit support - either download a Kaggle digits dataset and adapt
#    convert_asl_digits.py to it, or record your own via webcam:
python src/collect_asl_digits.py

# 3. Build the training arrays and train
python src/preprocess_asl.py
python src/train_asl_model.py
```

This produces `models/asl_model.h5` and `models/asl_label_map.json`.

### 5. Run it

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

Open the frontend URL, allow camera access, and start signing. If no trained model is found yet, the backend runs in **stub mode** — returning random predictions so you can confirm the full pipeline works end-to-end before training.

---

## 🧠 How Recognition Works

1. The browser captures webcam frames and runs **MediaPipe Hands** locally (loaded via CDN `<script>` tags — the npm packages don't resolve cleanly through Vite's bundler, so this project deliberately avoids `import` for MediaPipe).
2. Each frame's 21 hand landmarks `(x, y, z)` are sent over a **Socket.IO** connection to the Flask backend as they're detected.
3. The backend flattens them into a 63-value feature vector and runs it through a trained **dense neural network** (`BatchNorm -> Dense(256) -> Dense(128) -> Dense(64) -> Softmax`).
4. Predictions are smoothed on the frontend using a **rolling majority-vote window** (not just "N identical frames in a row") — this tolerates brief flicker between visually similar signs (like O/C) without needing a long, sluggish hold time.
5. Once a sign is confirmed, it's appended to the caption (`space` and `del` trigger their respective actions instead of appending a character).

---

## ⚠️ Known Limitations & Notes

- **J and Z are motion signs** in real ASL (traced through the air), but this project's training data consists of static images. The model learns *a* fixed pose for these letters, not the true gesture — expect these two to be the least reliable.
- **M and N** involve tucked/occluded fingers, which reduces MediaPipe's hand-detection success rate on training images for these classes specifically.
- This is a **fingerspelling** translator, not a whole-word or sentence-level ASL recognizer. Word/sentence-level sign language is a fundamentally different (and much harder) problem requiring motion-sequence models — a possible future direction, not part of this project's current scope.
- Digit classes have less training data than letters, so digit recognition, while solid, is not quite as strong as letter recognition.
- Model files, datasets, and the MediaPipe `.task` file are **not committed to this repo** (see `.gitignore`) — they exceed GitHub's file size limits and are meant to be regenerated locally via the scripts in `src/`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite |
| Hand tracking | MediaPipe Hands (CDN) |
| Realtime transport | Socket.IO |
| Backend | Flask, Flask-SocketIO |
| ML | TensorFlow / Keras |
| Data processing | pandas, NumPy, scikit-learn, OpenCV |

---

## 📄 License

This project is for educational and portfolio purposes.