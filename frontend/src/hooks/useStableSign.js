import { useRef, useState, useCallback } from "react";

const WINDOW = 12;
const MIN_AGREE = 0.7;
const MIN_CONF = 0.85; // use 85 if your backend sends confidence as 0-100
const HOLD_MS = 800;
const NONE = { label: null, confidence: 0 };

export function useStableSign(onCommit) {
  const buf = useRef([]);
  const candidate = useRef(null);
  const holdStart = useRef(0);
  const armed = useRef(true);
  const [stable, setStable] = useState(NONE);

  const reset = useCallback(() => {
    buf.current = [];
    candidate.current = null;
    armed.current = true;
    setStable(NONE);
  }, []);

  const push = useCallback(
    (label, confidence) => {
      const valid = label && confidence >= MIN_CONF;
      buf.current.push(valid ? { label, confidence } : null);
      if (buf.current.length > WINDOW) buf.current.shift();

      const frames = buf.current.filter(Boolean);
      const counts = {};
      frames.forEach((f) => (counts[f.label] = (counts[f.label] || 0) + 1));
      const [top, n] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0] || [];

      const ok = top && buf.current.length >= WINDOW / 2 && n / buf.current.length >= MIN_AGREE;

      if (!ok) {
        if (!frames.length) armed.current = true;
        candidate.current = null;
        setStable(NONE);
        return;
      }

      const avg = frames.filter((f) => f.label === top).reduce((s, f) => s + f.confidence, 0) / n;
      const now = performance.now();

      if (candidate.current !== top) {
        candidate.current = top;
        holdStart.current = now;
        armed.current = true;
      }
      setStable({ label: top, confidence: avg });

      if (armed.current && now - holdStart.current >= HOLD_MS) {
        armed.current = false;
        onCommit?.(top, avg);
      }
    },
    [onCommit]
  );

  return { stable, push, reset };
}