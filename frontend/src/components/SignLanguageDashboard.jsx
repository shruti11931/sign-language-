import React, { useState, useRef, useEffect } from "react";
import { Hand, Volume2, Delete, RotateCcw, Radio, Sun, Moon } from "lucide-react";
import { useSignDetection } from "../hooks/useSignDetection";
import { useWordDetection } from "../hooks/useWordDetection";

const THEMES = {
  dark: {
    bg: "#14181F", surface: "#1E2430", surface2: "#262E3D", border: "#2A3140",
    text: "#F3F1EA", muted: "#8B93A3", faint: "#4B5566",
    accent: "#E8A33D", accentBg: "rgba(232,163,61,0.12)",
    success: "#4FB6A6", successBg: "rgba(79,182,166,0.12)",
    alert: "#D97771", alertBg: "rgba(217,119,113,0.12)",
    canvasBg: "#0E1116",
  },
  light: {
    bg: "#F6F4EF", surface: "#FFFFFF", surface2: "#EFEBE2", border: "#E2DCCC",
    text: "#211E19", muted: "#6B6558", faint: "#A69F8F",
    accent: "#C97A2B", accentBg: "rgba(201,122,43,0.10)",
    success: "#2E8C7A", successBg: "rgba(46,140,122,0.10)",
    alert: "#C25650", alertBg: "rgba(194,86,80,0.10)",
    canvasBg: "#EAE6DA",
  },
};

export default function SignLanguageDashboard() {
  const [theme, setTheme] = useState("dark");
  const t = THEMES[theme];
  const canvasRef = useRef(null);

  const [mode, setMode] = useState("letter"); // "letter" | "word"

  const sign = useSignDetection(canvasRef, mode);
  const word = useWordDetection();

  const [cursorOn, setCursorOn] = useState(true);

  useEffect(() => {
    const blink = setInterval(() => setCursorOn((c) => !c), 500);
    return () => clearInterval(blink);
  }, []);

  // Bridge: useSignDetection's MediaPipe loop calls window.__wordModePushFrame
  // on every frame (see useSignDetection.js), which feeds word mode's rolling buffer.
  useEffect(() => {
    window.__wordModePushFrame = mode === "word" ? word.pushFrame : null;
    return () => { window.__wordModePushFrame = null; };
  }, [word.pushFrame, mode]);

  const confColor = sign.confidence > 0.85 ? t.success : sign.confidence > 0.7 ? t.accent : t.alert;
  const avgConfidence = sign.log.length
    ? Math.round((sign.log.reduce((sum, r) => sum + r.conf, 0) / sign.log.length) * 100)
    : 0;

  const wordConfColor = word.wordConfidence > 0.85 ? t.success : word.wordConfidence > 0.7 ? t.accent : t.alert;

  const s = {
    page: { minHeight: "100vh", width: "100%", background: t.bg, fontFamily: "Inter, sans-serif", transition: "background 0.3s ease" },
    wrap: { maxWidth: 960, margin: "0 auto", padding: "32px 24px" },
    header: { display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20, flexWrap: "wrap", gap: 12 },
    headerLeft: { display: "flex", alignItems: "center", gap: 12 },
    logoCircle: { width: 42, height: 42, borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", background: t.surface2, flexShrink: 0 },
    title: { fontFamily: "'Space Grotesk', sans-serif", fontSize: 22, fontWeight: 600, color: t.text, margin: 0 },
    subtitle: { fontSize: 12.5, color: t.muted, margin: "2px 0 0" },
    headerBtns: { display: "flex", alignItems: "center", gap: 10 },
    themeBtn: { width: 36, height: 36, borderRadius: "50%", border: `1px solid ${t.border}`, background: t.surface, display: "flex", alignItems: "center", justifyContent: "center", cursor: "pointer" },
    connDot: { display: "flex", alignItems: "center", gap: 6, fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: t.muted },
    liveBtn: (running) => ({ display: "flex", alignItems: "center", gap: 7, padding: "8px 16px", borderRadius: 999, fontSize: 13, fontWeight: 500, border: "none", cursor: "pointer", background: running ? t.successBg : t.alertBg, color: running ? t.success : t.alert }),
    modeRow: { display: "flex", gap: 8, marginBottom: 20 },
    modeBtn: (active) => ({
      padding: "8px 18px", borderRadius: 999, fontSize: 13, fontWeight: 600, border: `1px solid ${active ? t.accent : t.border}`,
      cursor: "pointer", background: active ? t.accentBg : t.surface, color: active ? t.accent : t.muted,
    }),
    grid: { display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 16, marginBottom: 16 },
    card: { background: t.surface, border: `1px solid ${t.border}`, borderRadius: 16, padding: 16 },
    label: { fontFamily: "'JetBrains Mono', monospace", fontSize: 11, letterSpacing: 0.5, color: t.muted, margin: "0 0 12px", textTransform: "uppercase" },
    camBox: { position: "relative", borderRadius: 12, background: t.canvasBg, overflow: "hidden", lineHeight: 0 },
    camCanvas: { width: "100%", height: "auto", display: "block", borderRadius: 12 },
    camTag: { position: "absolute", bottom: 10, left: 10, fontFamily: "'JetBrains Mono', monospace", fontSize: 11, padding: "4px 8px", borderRadius: 6, background: theme === "dark" ? "rgba(20,24,31,0.85)" : "rgba(255,255,255,0.85)", color: t.muted },
    camFps: { position: "absolute", top: 10, right: 10, display: "flex", alignItems: "center", gap: 6, fontFamily: "'JetBrains Mono', monospace", fontSize: 11, padding: "4px 8px", borderRadius: 6, background: theme === "dark" ? "rgba(20,24,31,0.85)" : "rgba(255,255,255,0.85)", color: t.text },
    bufferTag: { position: "absolute", bottom: 10, right: 10, display: "flex", alignItems: "center", gap: 6, fontFamily: "'JetBrains Mono', monospace", fontSize: 11, padding: "4px 8px", borderRadius: 6, background: theme === "dark" ? "rgba(20,24,31,0.85)" : "rgba(255,255,255,0.85)", color: t.text },
    camHint: { position: "absolute", top: "50%", left: "50%", transform: "translate(-50%,-50%)", fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: t.muted, background: theme === "dark" ? "rgba(20,24,31,0.85)" : "rgba(255,255,255,0.85)", padding: "6px 12px", borderRadius: 8 },
    dot: (color) => ({ width: 6, height: 6, borderRadius: "50%", background: color }),
    rightCol: { display: "flex", flexDirection: "column", gap: 16 },
    predictCard: { background: t.surface, border: `1px solid ${t.border}`, borderRadius: 16, padding: 22, textAlign: "center" },
    bigSign: { fontFamily: "'Space Grotesk', sans-serif", fontSize: 52, fontWeight: 700, color: t.text, margin: "4px 0", minHeight: 62 },
    bigWord: { fontFamily: "'Space Grotesk', sans-serif", fontSize: 32, fontWeight: 700, color: t.text, margin: "4px 0", minHeight: 62, wordBreak: "break-word" },
    confBar: { height: 6, borderRadius: 999, background: t.canvasBg, overflow: "hidden", marginTop: 10 },
    confFill: { height: "100%", borderRadius: 999, transition: "width 0.4s ease, background 0.4s ease" },
    confLabel: { fontFamily: "'JetBrains Mono', monospace", fontSize: 12, marginTop: 8 },
    statsRow: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 },
    statBox: { background: t.surface, border: `1px solid ${t.border}`, borderRadius: 12, padding: 12 },
    statLabel: { fontFamily: "'JetBrains Mono', monospace", fontSize: 10.5, color: t.muted, margin: 0 },
    statVal: { fontFamily: "'Space Grotesk', sans-serif", fontSize: 20, fontWeight: 600, color: t.text, margin: "4px 0 0" },
    captionCard: { background: t.surface, border: `1px solid ${t.border}`, borderRadius: 16, padding: 20, marginBottom: 16 },
    captionHeader: { display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12, flexWrap: "wrap", gap: 10 },
    btnRow: { display: "flex", gap: 8 },
    pillBtn: (bg, color) => ({ display: "flex", alignItems: "center", gap: 6, padding: "7px 13px", borderRadius: 999, fontSize: 12.5, fontWeight: 500, border: "none", cursor: "pointer", background: bg, color }),
    captionBox: { position: "relative", borderRadius: 12, background: t.canvasBg, padding: "22px 20px", overflow: "hidden" },
    captionText: { fontFamily: "'JetBrains Mono', monospace", fontSize: 24, letterSpacing: 1, color: t.text, minHeight: 32, margin: 0, wordBreak: "break-all" },
    captionUnderline: { position: "absolute", bottom: 0, left: 0, height: 3, transition: "width 0.4s ease, background 0.4s ease" },
    logRow: (highlight) => ({ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 12px", borderRadius: 10, background: highlight ? t.surface2 : "transparent" }),
    logTime: { fontFamily: "'JetBrains Mono', monospace", fontSize: 11.5, color: t.muted },
    logSign: { fontFamily: "'Space Grotesk', sans-serif", fontWeight: 600, color: t.text, fontSize: 15 },
    logConf: (color, bg) => ({ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, padding: "2px 8px", borderRadius: 6, color, background: bg }),
    empty: { fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: t.faint, padding: "8px 4px" },
    footer: { textAlign: "center", fontFamily: "'JetBrains Mono', monospace", fontSize: 11, color: t.faint, marginTop: 24 },
  };

  const activeConnected = mode === "letter" ? sign.connected : word.connected;
  const activeHandDetected = sign.handDetected; // shared MediaPipe loop drives both modes

  return (
    <div style={s.page}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');
        * { box-sizing: border-box; }
        @media (max-width: 720px) {
          .sl-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>

      <div style={s.wrap}>
        <div style={s.header}>
          <div style={s.headerLeft}>
            <div style={s.logoCircle}><Hand size={19} color={t.accent} /></div>
            <div>
              <p style={s.title}>Sign to Speech</p>
              <p style={s.subtitle}>
                {mode === "letter" ? "Fingerspelling translator — alphabet & digits" : "Word sign translator"}
              </p>
            </div>
          </div>
          <div style={s.headerBtns}>
            <span style={s.connDot}>
              <span style={s.dot(activeConnected ? t.success : t.alert)} />
              {activeConnected ? "Backend connected" : "Backend offline"}
            </span>
            <button style={s.themeBtn} onClick={() => setTheme(theme === "dark" ? "light" : "dark")} aria-label="Toggle theme">
              {theme === "dark" ? <Sun size={16} color={t.muted} /> : <Moon size={16} color={t.muted} />}
            </button>
            {mode === "letter" ? (
              <button style={s.liveBtn(sign.running)} onClick={() => sign.setRunning((r) => !r)}>
                <Radio size={13} />
                {sign.running ? "Live" : "Paused"}
              </button>
            ) : (
              <button style={s.liveBtn(word.running)} onClick={() => word.setRunning((r) => !r)}>
                <Radio size={13} />
                {word.running ? "Live" : "Paused"}
              </button>
            )}
          </div>
        </div>

        <div style={s.modeRow}>
          <button style={s.modeBtn(mode === "letter")} onClick={() => setMode("letter")}>Letters &amp; Digits</button>
          <button style={s.modeBtn(mode === "word")} onClick={() => setMode("word")}>Words</button>
        </div>

        <div style={s.grid} className="sl-grid">
          <div style={s.card}>
            <p style={s.label}>Camera feed</p>
            <div style={s.camBox}>
              <canvas ref={canvasRef} width={480} height={360} style={s.camCanvas} />
              {!activeHandDetected && (
                <span style={s.camHint}>Show your hand to the camera</span>
              )}
              <span style={s.camTag}>21 landmarks tracked</span>
              <span style={s.camFps}>
                <span style={s.dot(sign.running ? t.success : t.alert)} />
                {sign.fps} fps
              </span>

              {mode === "word" && (
                <span style={s.bufferTag}>
                  <span style={s.dot(word.bufferProgress === 30 ? t.success : t.accent)} />
                  buffer {word.bufferProgress}/30
                </span>
              )}
            </div>
          </div>

          <div style={s.rightCol}>
            {mode === "letter" ? (
              <>
                <div style={s.predictCard}>
                  <p style={s.label}>Detected sign</p>
                  <p style={s.bigSign}>{sign.currentSign || "—"}</p>
                  <div style={s.confBar}>
                    <div style={{ ...s.confFill, width: `${sign.confidence * 100}%`, background: confColor }} />
                  </div>
                  <p style={{ ...s.confLabel, color: confColor }}>
                    {sign.currentSign ? `${Math.round(sign.confidence * 100)}% confidence` : "waiting for hand..."}
                  </p>
                </div>

                <div style={s.statsRow}>
                  <div style={s.statBox}>
                    <p style={s.statLabel}>AVG CONFIDENCE</p>
                    <p style={s.statVal}>{avgConfidence}%</p>
                  </div>
                  <div style={s.statBox}>
                    <p style={s.statLabel}>SIGNS LOGGED</p>
                    <p style={s.statVal}>{sign.totalSigns}</p>
                  </div>
                </div>
              </>
            ) : (
              <>
                <div style={s.predictCard}>
                  <p style={s.label}>Detected word</p>
                  <p style={s.bigWord}>{word.predictedWord || "—"}</p>
                  <div style={s.confBar}>
                    <div style={{ ...s.confFill, width: `${word.wordConfidence * 100}%`, background: wordConfColor }} />
                  </div>
                  <p style={{ ...s.confLabel, color: wordConfColor }}>
                    {word.predictedWord ? `${Math.round(word.wordConfidence * 100)}% confidence` : "show your hand and sign a word..."}
                  </p>
                </div>

                <div style={s.statsRow}>
                  <div style={s.statBox}>
                    <p style={s.statLabel}>WORDS LOGGED</p>
                    <p style={s.statVal}>{word.totalWords}</p>
                  </div>
                  <div style={s.statBox}>
                    <p style={s.statLabel}>BUFFER</p>
                    <p style={s.statVal}>{word.bufferProgress}/30</p>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        <div style={s.captionCard}>
          <div style={s.captionHeader}>
            <p style={{ ...s.label, margin: 0 }}>Translated caption</p>
            <div style={s.btnRow}>
              {mode === "letter" ? (
                <>
                  <button style={s.pillBtn(t.accentBg, t.accent)} onClick={sign.speak}>
                    <Volume2 size={13} /> Speak
                  </button>
                  <button style={s.pillBtn(t.surface2, t.muted)} onClick={sign.undoLastLetter}>
                    <Delete size={13} /> Undo
                  </button>
                  <button style={s.pillBtn(t.surface2, t.muted)} onClick={sign.clearSentence}>
                    <RotateCcw size={13} /> Clear
                  </button>
                </>
              ) : (
                <>
                  <button style={s.pillBtn(t.accentBg, t.accent)} onClick={word.speak}>
                    <Volume2 size={13} /> Speak
                  </button>
                  <button style={s.pillBtn(t.surface2, t.muted)} onClick={word.undoLastWord}>
                    <Delete size={13} /> Undo
                  </button>
                  <button style={s.pillBtn(t.surface2, t.muted)} onClick={word.clearSentence}>
                    <RotateCcw size={13} /> Clear
                  </button>
                </>
              )}
            </div>
          </div>
          <div style={s.captionBox}>
            {mode === "letter" ? (
              <p style={s.captionText}>
                {sign.sentence || <span style={{ color: t.faint }}>Start signing to see text appear...</span>}
                {sign.sentence && <span style={{ opacity: cursorOn ? 1 : 0, color: t.accent }}>|</span>}
              </p>
            ) : (
              <p style={s.captionText}>
                {word.sentence || <span style={{ color: t.faint }}>Show your hand and sign a word...</span>}
                {word.sentence && <span style={{ opacity: cursorOn ? 1 : 0, color: t.accent }}>|</span>}
              </p>
            )}
            <div style={{
              ...s.captionUnderline,
              width: `${(mode === "letter" ? sign.confidence : word.wordConfidence) * 100}%`,
              background: mode === "letter" ? confColor : wordConfColor,
            }} />
          </div>
        </div>

        <div style={s.card}>
          <p style={s.label}>{mode === "letter" ? "Recent gesture log" : "Recent word log"}</p>
          <div>
            {mode === "letter" ? (
              <>
                {sign.log.length === 0 && <p style={s.empty}>No signs confirmed yet — hold a sign steady for it to register.</p>}
                {sign.log.map((row, i) => (
                  <div key={i} style={s.logRow(i === 0)}>
                    <span style={s.logTime}>{row.time}</span>
                    <span style={s.logSign}>{row.sign}</span>
                    <span style={s.logConf(row.conf > 0.85 ? t.success : t.accent, row.conf > 0.85 ? t.successBg : t.accentBg)}>
                      {Math.round(row.conf * 100)}%
                    </span>
                  </div>
                ))}
              </>
            ) : (
              <>
                {word.log.length === 0 && <p style={s.empty}>No words confirmed yet — show a sign to the camera and hold it steady.</p>}
                {word.log.map((row, i) => (
                  <div key={i} style={s.logRow(i === 0)}>
                    <span style={s.logTime}>{row.time}</span>
                    <span style={s.logSign}>{row.sign}</span>
                    <span style={s.logConf(row.conf > 0.85 ? t.success : t.accent, row.conf > 0.85 ? t.successBg : t.accentBg)}>
                      {Math.round(row.conf * 100)}%
                    </span>
                  </div>
                ))}
              </>
            )}
          </div>
        </div>

        <p style={s.footer}>
          {activeConnected ? "Connected to backend — live predictions" : "Backend not connected — start the Flask server on port 5000"}
        </p>
      </div>
    </div>
  );
}