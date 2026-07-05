// frontend/src/hooks/useWordDetection.js
//
// Continuous word-mode detection - no hold button. Maintains a rolling
// window of the last 30 frames of single-hand landmarks, and periodically
// sends that window for a word prediction, similar in spirit to how
// useSignDetection streams every frame for letters.

import { useRef, useState, useCallback } from "react";
import { io } from "socket.io-client";

const BACKEND_URL = "http://localhost:5000";
const SEQ_LEN = 30;
const PREDICT_EVERY_N_FRAMES = 3; // throttle how often we ask the backend
const STABLE_REPEATS_NEEDED = 2;  // keep at 2 - going to 1 risks single-frame misfiresword must repeat this many predictions in a row
const MIN_CONFIDENCE = 0.6;

export function useWordDetection() {
  const socketRef = useRef(null);
  const bufferRef = useRef([]);
  const frameCounterRef = useRef(0);
  const stableWordRef = useRef(null);
  const stableCountRef = useRef(0);
  const lastConfirmedRef = useRef(null);
  const cooldownRef = useRef(0);

  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(true);
  const [predictedWord, setPredictedWord] = useState(null);
  const [wordConfidence, setWordConfidence] = useState(0);
  const [bufferProgress, setBufferProgress] = useState(0);
  const [sentence, setSentence] = useState("");
  const [log, setLog] = useState([]);
  const [totalWords, setTotalWords] = useState(0);

  const ensureSocket = useCallback(() => {
    if (socketRef.current) return socketRef.current;
    const socket = io(BACKEND_URL, {
      transports: ["websocket"],
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 20000,
    });
    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));

    socket.on("word_prediction", ({ word, confidence }) => {
      setPredictedWord(word);
      setWordConfidence(confidence);

      if (!word || confidence < MIN_CONFIDENCE) {
        stableWordRef.current = null;
        stableCountRef.current = 0;
        return;
      }

      if (word === stableWordRef.current) {
        stableCountRef.current += 1;
      } else {
        stableWordRef.current = word;
        stableCountRef.current = 1;
      }

      // cooldown avoids re-confirming the same held sign repeatedly
      if (cooldownRef.current > 0) {
        cooldownRef.current -= 1;
        return;
      }

      if (stableCountRef.current >= STABLE_REPEATS_NEEDED && word !== lastConfirmedRef.current) {
        setSentence((s) => (s ? `${s} ${word}` : word));
        setTotalWords((n) => n + 1);
        setLog((prev) => [
          { time: new Date().toTimeString().slice(0, 8), sign: word, conf: confidence },
          ...prev,
        ].slice(0, 6));
        lastConfirmedRef.current = word;
        cooldownRef.current = 4; // ignore repeats for the next ~4 predictions
        stableCountRef.current = 0;
      }
    });

    socket.on("word_prediction_error", (err) => console.warn("[word_prediction_error]", err));
    socketRef.current = socket;
    return socket;
  }, []);

  // Called every frame from useSignDetection's MediaPipe loop via the
  // window.__wordModePushFrame bridge (see SignLanguageDashboard.jsx).
  const pushFrame = useCallback((landmarks) => {
    if (!running) return;

    const flat = landmarks
      ? landmarks.flatMap((p) => [p.x, p.y, p.z])
      : new Array(63).fill(0);

    bufferRef.current.push(flat);
    if (bufferRef.current.length > SEQ_LEN) {
      bufferRef.current.shift();
    }
    setBufferProgress(bufferRef.current.length);

    if (bufferRef.current.length === SEQ_LEN) {
      frameCounterRef.current += 1;
      if (frameCounterRef.current >= PREDICT_EVERY_N_FRAMES) {
        frameCounterRef.current = 0;

        // Skip prediction if any frame in this window has no hand
        // (all-zero) - that means the window spans a transition, not
        // a clean held sign, and would bias toward whatever training
        // class looks closest to "low motion."
        const hasGap = bufferRef.current.some((frame) => frame.every((v) => v === 0));
        if (hasGap) return;

        const socket = ensureSocket();
        if (socket.connected) {
          socket.emit("word_sequence", { sequence: bufferRef.current.slice() });
        }
      }
    }
  }, [running, ensureSocket]);

  const clearSentence = useCallback(() => setSentence(""), []);
  const undoLastWord = useCallback(() => {
    setSentence((s) => {
      const parts = s.trim().split(" ");
      parts.pop();
      return parts.join(" ");
    });
  }, []);
  const speak = useCallback(() => {
    if ("speechSynthesis" in window && sentence) {
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(sentence));
    }
  }, [sentence]);

  return {
    connected, running, setRunning,
    predictedWord, wordConfidence, bufferProgress,
    sentence, log, totalWords,
    clearSentence, undoLastWord, speak,
    pushFrame,
  };
}