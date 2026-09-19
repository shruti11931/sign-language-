// frontend/src/hooks/useSignDetection.js
//
// Single-hand ASL detection. Uses MediaPipe Hands loaded via CDN <script>
// tags (window.Hands/Camera/drawConnectors/drawLandmarks), NOT npm imports.

import { useEffect, useRef, useState, useCallback } from "react";
import { io } from "socket.io-client";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5000";
const MIN_CONFIDENCE = 0.7;
const ACCENT_COLOR = "#E8A33D";
const WINDOW_SIZE = 6;
const MAJORITY_NEEDED = 4;
const INFLIGHT_TIMEOUT_MS = 600;
const NO_HAND_RESET_FRAMES = 8;


const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

export function useSignDetection(canvasRef, mode) {
  const videoRef = useRef(document.createElement("video"));
  const socketRef = useRef(null);
  const recentPredictionsRef = useRef([]);
  const lastConfirmedRef = useRef(null);
  const frameTimesRef = useRef([]);
  const letterFrameCounterRef = useRef(0);
  const inflightSinceRef = useRef(0);
  const noHandFramesRef = useRef(0);

  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(true);
  const [handDetected, setHandDetected] = useState(false);
  const [currentSign, setCurrentSign] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [fps, setFps] = useState(0);
  const [sentence, setSentence] = useState("");
  const [log, setLog] = useState([]);
  const [totalSigns, setTotalSigns] = useState(0);

  useEffect(() => {
    const socket = io(BACKEND_URL, {
      transports: ["websocket"],
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 20000,
    });

    socketRef.current = socket;

    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));

    socket.on("prediction", ({ letter, confidence: conf }) => {
      inflightSinceRef.current = 0; // server answered, allow next frame
      
      if (!letter) return;

      recentPredictionsRef.current.push({ letter, conf });
      if (recentPredictionsRef.current.length > WINDOW_SIZE) {
        recentPredictionsRef.current.shift();
      }

      const counts = {};
      for (const p of recentPredictionsRef.current) {
        counts[p.letter] = (counts[p.letter] || 0) + 1;
      }
      const [topLetter, topCount] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];

      // Not enough agreement yet -> keep showing the previous stable sign
      if (topCount < MAJORITY_NEEDED) return;

      const agreeing = recentPredictionsRef.current.filter((p) => p.letter === topLetter);
      const avgConf = agreeing.reduce((s, p) => s + p.conf, 0) / agreeing.length;

      // Display only the stable sign, never the raw per-frame one
      setCurrentSign(topLetter);
      setConfidence(avgConf);

      if (avgConf > MIN_CONFIDENCE && topLetter !== lastConfirmedRef.current) {
        if (topLetter === "space") {
          setSentence((s) => (s + " ").slice(-40));
        } else if (topLetter === "del") {
          setSentence((s) => s.slice(0, -1));
        } else {
          setSentence((s) => (s + topLetter).slice(-40));
        }
        setTotalSigns((n) => n + 1);
        setLog((prev) => [
          { time: new Date().toTimeString().slice(0, 8), sign: topLetter, conf: avgConf },
          ...prev,
        ].slice(0, 6));

        lastConfirmedRef.current = topLetter;
        recentPredictionsRef.current = [];
      }
    });

    socket.on("prediction_error", (err) => {
      inflightSinceRef.current = 0;
      console.warn("[prediction_error]", err);
    });

    return () => socket.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const { Hands, Camera, drawConnectors, drawLandmarks } = window;
    if (!Hands || !Camera || !drawConnectors || !drawLandmarks) {
      console.error("[useSignDetection] MediaPipe globals not found — check index.html script tags");
      return;
    }

    const hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240/${file}`,
    });
    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 0,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
      selfieMode: true,
    });

    hands.onResults((results) => {
      const now = performance.now();
      frameTimesRef.current.push(now);
      frameTimesRef.current = frameTimesRef.current.filter((t) => now - t < 1000);
      setFps(frameTimesRef.current.length);

      ctx.save();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

      const rawHands = results.multiHandLandmarks || [];
      const landmarks = rawHands.length > 0 ? rawHands[0] : null;

      if (landmarks) {
        drawConnectors(ctx, landmarks, HAND_CONNECTIONS, { color: ACCENT_COLOR, lineWidth: 2 });
        drawLandmarks(ctx, landmarks, { color: ACCENT_COLOR, radius: 3 });
        setHandDetected(true);
        noHandFramesRef.current = 0;

        letterFrameCounterRef.current += 1;
        const inflight =
          inflightSinceRef.current && now - inflightSinceRef.current < INFLIGHT_TIMEOUT_MS;

        if (
          mode === "letter" &&
          socketRef.current?.connected &&
          running &&
          letterFrameCounterRef.current % 2 === 0 &&
          !inflight // drop frames while server is still busy
        ) {
          const points = landmarks.map((p) => [p.x, p.y, p.z]);
          inflightSinceRef.current = now;
          socketRef.current.emit("frame", { landmarks: points });
        }
      } else {
        setHandDetected(false);
        noHandFramesRef.current += 1;

        // Hand really gone (not just a dropped frame): reset so the same
        // letter can be signed again and the display clears.
        if (noHandFramesRef.current === NO_HAND_RESET_FRAMES) {
          recentPredictionsRef.current = [];
          lastConfirmedRef.current = null;
          inflightSinceRef.current = 0;
          setCurrentSign(null);
          setConfidence(0);
        }
      }

      // Fires every frame, hand or no hand — matches how collect_asl_words.py
      // recorded training clips (a no-hand frame still counts, isn't skipped).
      window.__wordModePushFrame?.(landmarks);

      ctx.restore();
    });

    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        await hands.send({ image: videoRef.current });
      },
      width: 480,
      height: 360,
    });
    camera.start();

    return () => {
      camera.stop();
      hands.close();
    };
  }, [canvasRef, running, mode]);

  const clearSentence = useCallback(() => setSentence(""), []);
  const undoLastLetter = useCallback(() => setSentence((s) => s.slice(0, -1)), []);
  const speak = useCallback(() => {
    if ("speechSynthesis" in window && sentence) {
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(sentence));
    }
  }, [sentence]);

  return {
    connected, running, setRunning, handDetected,
    currentSign, confidence, fps, sentence, log, totalSigns,
    clearSentence, undoLastLetter, speak,
  };
}