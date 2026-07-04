/**
 * frontend/src/hooks/useWordDetection.js
 *
 * Same CDN/window approach as useSignDetection.js — does NOT import
 * @mediapipe/hands or @mediapipe/camera_utils via npm, since those
 * packages break Vite's ESM pre-bundler. Requires the same two
 * <script> tags in index.html (you should already have them there
 * from fixing letter mode):
 *
 *   <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
 *   <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
 *
 * npm install socket.io-client   (only this one needs npm)
 *
 * Word mode is fundamentally different from letter mode:
 *   - tracks TWO hands (not one)
 *   - selfieMode is OFF, because preprocess_words.py processed raw
 *     (non-mirrored) video files — matching that at inference time
 *     matters, so screen-mirroring is done with CSS instead, never
 *     by flipping the actual landmark data sent to the model
 *   - doesn't stream every frame; the user presses and holds a
 *     "Record sign" button, frames are buffered locally for ~1.5s,
 *     then the whole 30-frame sequence is sent once
 */

import { useEffect, useRef, useState, useCallback } from "react";
import { io } from "socket.io-client";

const BACKEND_URL = "http://localhost:5000";
const SEQ_LEN = 30;
const MISSING_HAND = new Array(63).fill(-1.0);

/** Converts one MediaPipe Hands result into the same 127-value vector
 *  preprocess_words.py builds: [uses_two_hands] + left(63) + right(63) */
function resultsToFeatureVector(results) {
  let left = MISSING_HAND;
  let right = MISSING_HAND;

  const landmarksList = results.multiHandLandmarks || [];
  const handednessList = results.multiHandedness || [];

  landmarksList.forEach((landmarks, i) => {
    const flat = [];
    landmarks.forEach((lm) => flat.push(lm.x, lm.y, lm.z));
    const label = handednessList[i]?.label; // "Left" | "Right"
    if (label === "Left") left = flat;
    else if (label === "Right") right = flat;
  });

  const usesTwoHands = left !== MISSING_HAND && right !== MISSING_HAND ? 1.0 : 0.0;
  return [usesTwoHands, ...left, ...right];
}

export function useWordDetection({ onWord } = {}) {
  const videoRef = useRef(null);
  const socketRef = useRef(null);
  const bufferRef = useRef([]);
  const recordingRef = useRef(false);
  const lastFeatureRef = useRef([0.0, ...MISSING_HAND, ...MISSING_HAND]);

  const [connected, setConnected] = useState(false);
  const [recording, setRecording] = useState(false);
  const [bufferProgress, setBufferProgress] = useState(0); // 0-30
  const [predictedWord, setPredictedWord] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [status, setStatus] = useState("idle"); // idle | recording | processing | low_confidence
  const [mediapipeReady, setMediapipeReady] = useState(false);

  // 1. socket connection
  useEffect(() => {
    const socket = io(BACKEND_URL, { transports: ["websocket"] });
    socketRef.current = socket;

    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));

    socket.on("word_prediction", ({ word, confidence }) => {
      setConfidence(confidence);
      if (word) {
        setPredictedWord(word);
        setStatus("idle");
        onWord && onWord(word, confidence);
      } else {
        setStatus("low_confidence");
      }
    });

    socket.on("word_prediction_error", (err) => {
      console.error("[word] backend error:", err);
      setStatus("idle");
    });

    return () => socket.disconnect();
  }, [onWord]);

  // 2. MediaPipe Hands, two-hand tracking, selfieMode OFF
  useEffect(() => {
    if (!videoRef.current) return;

    if (typeof window.Hands === "undefined" || typeof window.Camera === "undefined") {
      console.error(
        "[useWordDetection] window.Hands / window.Camera not found. " +
        "Add the MediaPipe <script> tags to index.html — see the comment " +
        "at the top of this file."
      );
      return;
    }
    setMediapipeReady(true);

    const hands = new window.Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
    });
    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
      selfieMode: false, // must match how preprocess_words.py read the training videos
    });

    hands.onResults((results) => {
      const feature = resultsToFeatureVector(results);
      lastFeatureRef.current = feature;

      if (recordingRef.current) {
        bufferRef.current.push(feature);
        setBufferProgress(bufferRef.current.length);

        if (bufferRef.current.length >= SEQ_LEN) {
          finishRecording();
        }
      }
    });

    const camera = new window.Camera(videoRef.current, {
      onFrame: async () => {
        await hands.send({ image: videoRef.current });
      },
      width: 640,
      height: 480,
    });
    camera.start();

    return () => {
      camera.stop();
      hands.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const finishRecording = useCallback(() => {
    recordingRef.current = false;
    setRecording(false);
    setStatus("processing");

    // pad/trim to exactly SEQ_LEN in case the last frame arrived late
    let seq = bufferRef.current.slice(0, SEQ_LEN);
    while (seq.length < SEQ_LEN) seq.push(lastFeatureRef.current);

    if (socketRef.current?.connected) {
      socketRef.current.emit("word_sequence", { sequence: seq });
    } else {
      setStatus("idle");
    }

    bufferRef.current = [];
    setBufferProgress(0);
  }, []);

  const startRecording = useCallback(() => {
    bufferRef.current = [];
    setBufferProgress(0);
    setPredictedWord(null);
    recordingRef.current = true;
    setRecording(true);
    setStatus("recording");
  }, []);

  const stopRecording = useCallback(() => {
    // allows releasing the button early; pads out the rest of the sequence
    if (recordingRef.current) finishRecording();
  }, [finishRecording]);

  return {
    videoRef,
    connected,
    recording,
    bufferProgress,   // 0-30, drive a progress ring/bar in the UI
    predictedWord,
    confidence,
    status,
    mediapipeReady,
    startRecording,   // call on button mousedown/touchstart
    stopRecording,    // call on button mouseup/touchend
  };
}