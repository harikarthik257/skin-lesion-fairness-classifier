/* Evidence Garden: clinical editorial surfaces, warm precision, visible evidence and honest caveats. */
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDownRight, ArrowUpRight, CircleAlert, FileText, Gauge, Info, Loader2, Moon, Play, ScanLine, ShieldCheck, Sun, Upload, Waves, X } from "lucide-react";
import { useTheme } from "@/contexts/ThemeContext";

const CLASS_ORDER = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"] as const;
type ClassCode = (typeof CLASS_ORDER)[number];

const CLASS_META: Record<ClassCode, { label: string; color: string }> = {
  MEL: { label: "Melanoma", color: "#e47c66" },
  NV: { label: "Melanocytic nevus", color: "#2769d8" },
  BCC: { label: "Basal cell carcinoma", color: "#6f8fb9" },
  AKIEC: { label: "Actinic keratosis", color: "#b3a47d" },
  BKL: { label: "Benign keratosis", color: "#7f9e8d" },
  DF: { label: "Dermatofibroma", color: "#927b8f" },
  VASC: { label: "Vascular lesion", color: "#b46a7a" },
};

const GROUP_ORDER = ["Light (I-II)", "Medium (III-IV)", "Dark (V-VI)"];

interface PredictionResult {
  predicted_class: string;
  predicted_label: string;
  description: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  gradcam_heatmap: string;
  estimated_skin_tone: string;
  ita_value: number;
  uncertainty: { entropy: number; level: string } | null;
  mode: string;
}

interface VariantMetric {
  key: string;
  name: string;
  best: boolean;
  overall_accuracy: number;
  melanoma_recall: number | null;
  group_accuracy: Record<string, number | null>;
  fairness_gap: number | null;
  n: number;
}

function EvidenceStamp({ children, tone = "blue" }: { children: React.ReactNode; tone?: "blue" | "coral" | "sage" }) {
  return <span className={`stamp stamp-${tone}`}><span className="stamp-dot" />{children}</span>;
}

function LogoMark() {
  return (
    <svg viewBox="0 0 28 28" width="100%" height="100%" fill="none" aria-hidden="true">
      <circle cx="14" cy="14" r="10.5" stroke="#2769d8" strokeWidth="2" />
      <circle cx="14" cy="14" r="4" fill="#2769d8" />
      <circle cx="14" cy="14" r="12.5" stroke="#e47c66" strokeWidth="1" strokeDasharray="1.5 3" opacity="0.55" />
    </svg>
  );
}

function LensArt() {
  return (
    <svg viewBox="0 0 320 320" width="100%" height="100%" fill="none" aria-hidden="true">
      <defs>
        <radialGradient id="lensGrad" cx="46%" cy="42%" r="62%">
          <stop offset="0%" stopColor="#e9f0fb" />
          <stop offset="100%" stopColor="#2769d8" stopOpacity="0.4" />
        </radialGradient>
      </defs>
      <circle cx="160" cy="160" r="130" fill="url(#lensGrad)" />
      <circle cx="160" cy="160" r="130" stroke="#2769d8" strokeWidth="1.5" opacity="0.45" />
      <circle cx="160" cy="160" r="88" stroke="#e47c66" strokeWidth="1" opacity="0.5" strokeDasharray="3 6" />
      <circle cx="160" cy="160" r="46" fill="#2769d8" opacity="0.9" />
      <circle cx="160" cy="160" r="46" fill="none" stroke="#fff" strokeWidth="2" opacity="0.35" />
      <circle cx="160" cy="160" r="18" fill="#fff" opacity="0.5" />
    </svg>
  );
}

function ProbabilityBars({ probs }: { probs: Record<string, number> }) {
  return (
    <div className="probability-list">
      {CLASS_ORDER.map((code, index) => {
        const value = probs[code] ?? 0;
        const meta = CLASS_META[code];
        return (
          <div className="prob-row" key={code}>
            <div className="prob-meta"><span>{meta.label}</span><strong>{value.toFixed(1)}%</strong></div>
            <div className="prob-track"><div className="prob-fill" style={{ width: `${Math.max(value, 1.3)}%`, background: meta.color, animationDelay: `${index * 55}ms` }} /></div>
          </div>
        );
      })}
    </div>
  );
}

function DermalLens() {
  return (
    <div className="lens-scene" aria-label="Abstract dermal lens visualization">
      <div className="lens-halo" />
      <div className="lens-ring lens-ring-a" />
      <div className="lens-ring lens-ring-b" />
      <div className="lens-orb"><LensArt /></div>
      <div className="lens-orbit orbit-a" />
      <div className="lens-orbit orbit-b" />
      <div className="lens-caption"><span>DERMAL LENS / 07 CLASSES</span><span>MODEL FOCUS MAP</span></div>
    </div>
  );
}

const API_BASE = "";

export default function Home() {
  const { theme, toggleTheme } = useTheme();
  const [mode, setMode] = useState<"fast" | "uncertainty">("fast");
  const [selectedExample, setSelectedExample] = useState<ClassCode | null>("MEL");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(`/examples/MEL.jpg`);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [variants, setVariants] = useState<VariantMetric[] | null>(null);
  const [metricsError, setMetricsError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/metrics`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((data) => setVariants(data.variants))
      .catch(() => setMetricsError("Could not load evaluation results from the backend. Is it running (uvicorn backend.main:app --port 8000)?"));
  }, []);

  const baseline = variants?.find((v) => v.key === "baseline") ?? null;
  const bestVariant = variants?.find((v) => v.best) ?? null;
  const lowestGapVariant = useMemo(() => {
    if (!variants) return null;
    const withGap = variants.filter((v) => v.fairness_gap !== null);
    if (!withGap.length) return null;
    return withGap.reduce((min, v) => (v.fairness_gap! < min.fairness_gap! ? v : min));
  }, [variants]);
  const maxGroupAcc = useMemo(() => {
    if (!variants) return 100;
    return Math.max(...variants.map((v) => v.group_accuracy[GROUP_ORDER[0]] ?? 0), 1);
  }, [variants]);

  function handleExampleSelect(code: ClassCode) {
    setSelectedExample(code);
    setUploadedFile(null);
    setPreviewUrl(`/examples/${code}.jpg`);
    setResult(null);
    setErrorMsg(null);
  }

  function handleFileChange(file: File | null) {
    if (!file) return;
    setUploadedFile(file);
    setSelectedExample(null);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setErrorMsg(null);
  }

  async function handleDiagnose() {
    setLoading(true);
    setErrorMsg(null);
    try {
      let res: Response;
      if (uploadedFile) {
        const form = new FormData();
        form.append("file", uploadedFile);
        res = await fetch(`${API_BASE}/api/predict?mode=${mode}`, { method: "POST", body: form });
      } else if (selectedExample) {
        res = await fetch(`${API_BASE}/api/predict_example`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ code: selectedExample, mode }),
        });
      } else {
        setErrorMsg("Choose an example or upload an image first.");
        setLoading(false);
        return;
      }
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const data: PredictionResult = await res.json();
      setResult(data);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Something went wrong running the model.");
    } finally {
      setLoading(false);
    }
  }

  const diagnosed = result !== null;

  return <div className="site-shell">
    <header className="site-header"><a className="brand" href="#top"><span className="brand-mark"><LogoMark /></span><span>Derm<span>Lens</span></span></a><nav><a href="#demo">Diagnose</a><a href="#fairness">Fairness</a><a href="#method">Method</a></nav><div className="header-actions"><span className="header-status"><i /> Research prototype</span><button className="icon-button" aria-label="Toggle color theme" onClick={toggleTheme}>{theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}</button></div></header>

    <main id="top">
      <section className="hero container"><div className="hero-copy"><EvidenceStamp>RESEARCH / TRANSPARENCY FIRST</EvidenceStamp><h1>See the signal.<br /><em>Inspect the gap.</em></h1><p className="hero-lead">A skin lesion classifier designed to show its work—and where it is still learning to see fairly across skin tones.</p><div className="hero-ctas"><a className="button button-primary" href="#demo">Try the demo <ArrowDownRight size={16} /></a><a className="text-link" href="#fairness">Explore the evidence <ArrowUpRight size={15} /></a></div><div className="hero-footnote"><ShieldCheck size={15} /><span>Built with HAM10000 + ISIC2019</span><span className="divider" /><span>7 lesion classes</span></div></div><div className="hero-visual"><DermalLens /><div className="hero-note note-one"><span className="note-index">01</span><span>Prediction is a starting point, not a conclusion.</span></div><div className="hero-note note-two"><span className="note-index">02</span><span>Skin tone group = algorithmic estimate</span></div></div></section>

      <section className="notice-band"><div className="container notice-inner"><CircleAlert size={20} /><div><strong>Research & educational prototype — not a medical device.</strong><span>Never use this tool to diagnose, treat, or rule out a condition. Always speak with a qualified dermatologist.</span></div><a href="#method">Read limitations <ArrowUpRight size={14} /></a></div></section>

      <section className="story container"><div className="section-kicker"><span>01 / WHY THIS EXISTS</span><span className="kicker-line" /></div><div className="story-grid"><h2>Dermatology AI has a<br /><em>visibility problem.</em></h2><div className="story-body"><p>Most public image datasets overrepresent lighter skin. That means a model can look accurate overall while quietly missing more on darker skin.</p><p>DermLens makes that imbalance visible. The point is not to promise a perfect answer—it is to make performance, uncertainty, and tradeoffs inspectable.</p><a className="text-link" href="#fairness">See performance by skin tone <ArrowUpRight size={15} /></a></div></div><div className="signal-strip"><div><span className="signal-label">OVERALL ACCURACY</span><strong>{baseline ? baseline.overall_accuracy.toFixed(1) : "—"}<span>%</span></strong><small>baseline model</small></div><div><span className="signal-label">MELANOMA RECALL</span><strong>{baseline?.melanoma_recall != null ? baseline.melanoma_recall.toFixed(1) : "—"}<span>%</span></strong><small>the miss that matters</small></div><div className="signal-callout"><Waves size={19} /><span>Accuracy is one number.<br /><b>Context is the evidence.</b></span></div></div></section>

      <section id="demo" className="demo-section"><div className="container"><div className="section-kicker"><span>02 / INTERACTIVE DEMO</span><span className="kicker-line" /><EvidenceStamp tone="coral">RESEARCH ONLY</EvidenceStamp></div><div className="demo-header"><div><h2>What does the model<br /><em>see in a photo?</em></h2><p>Upload a lesion image, or try one of the curated examples below — real held-out test images, one per class.</p></div><div className="mode-switch"><span className="switch-label">INFERENCE MODE</span><div className="switch-buttons"><button className={mode === "fast" ? "active" : ""} onClick={() => setMode("fast")}>Fast</button><button className={mode === "uncertainty" ? "active" : ""} onClick={() => setMode("uncertainty")}>Uncertainty-aware <Info size={13} /></button></div><small>{mode === "uncertainty" ? "MC-Dropout adds a confidence range and runs ~30× slower." : "A quick TTA-averaged pass. No uncertainty estimate included."}</small></div></div><div className="demo-grid"><div className="upload-card"><input ref={fileInputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={(e) => handleFileChange(e.target.files?.[0] ?? null)} /><div className={`upload-zone ${previewUrl ? "has-preview" : ""}`} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); handleFileChange(e.dataTransfer.files?.[0] ?? null); }}>{previewUrl ? <><img className="upload-preview-img" src={previewUrl} alt="Selected lesion" /><span className="upload-preview-label">{uploadedFile ? uploadedFile.name : `Example: ${selectedExample ? CLASS_META[selectedExample].label : ""}`}</span></> : <><div className="upload-icon"><Upload size={19} /></div><strong>Drop a photo here</strong><span>or choose a file from your device</span></>}<button className="button button-secondary" style={{ marginTop: 14 }} onClick={() => fileInputRef.current?.click()}><FileText size={15} /> Choose image</button><small>JPG, PNG · Max 10 MB · No images are stored</small></div><div className="example-header"><span>TRY AN EXAMPLE</span><span>7 CLASSES</span></div><div className="example-strip">{CLASS_ORDER.map((code) => <button key={code} className={`example-chip ${selectedExample === code ? "selected" : ""}`} onClick={() => handleExampleSelect(code)}><span className="example-dot" style={{ background: CLASS_META[code].color }} /><span>{code}</span></button>)}</div><button className="button button-primary diagnose-button" disabled={loading || (!uploadedFile && !selectedExample)} onClick={handleDiagnose}>{loading ? <Loader2 size={17} className="spin" style={{ animation: "spin 1s linear infinite" }} /> : <ScanLine size={17} />} {loading ? "Running model…" : diagnosed ? "Run again" : "Diagnose"}</button>{errorMsg && <div className="error-banner"><CircleAlert size={14} /><span>{errorMsg}</span></div>}</div><div className={`result-card ${diagnosed ? "is-diagnosed" : ""}`}>{!diagnosed ? <div className="result-empty"><div className="empty-orb"><Play size={17} /></div><span className="stamp stamp-blue"><span className="stamp-dot" />AWAITING IMAGE</span><h3>Your result will appear here.</h3><p>Choose an example or upload a photo to see the full probability spread, model focus, and plain-language context.</p></div> : <div className="result-content"><div className="result-top"><EvidenceStamp tone={result!.confidence < 50 ? "coral" : "sage"}>{result!.confidence < 50 ? "LOW CONFIDENCE — VERIFY WITH A SPECIALIST" : "MODEL OUTPUT"}</EvidenceStamp><span className="result-time">Live inference · {result!.mode === "uncertainty" ? "MC-Dropout" : "TTA (fast)"}</span></div><div className="prediction"><div><span className="prediction-label">PREDICTED CLASS</span><h3>{result!.predicted_label}</h3></div><div className="confidence"><strong>{result!.confidence.toFixed(1)}<small>%</small></strong><span>confidence</span></div></div><p className="result-description">{result!.description}</p><div className="heatmap-row"><figure><img src={previewUrl ?? ""} alt="Uploaded lesion" /><figcaption>Your image</figcaption></figure><figure><img src={result!.gradcam_heatmap} alt="Grad-CAM++ model focus heatmap" /><figcaption>Model focus (Grad-CAM++)</figcaption></figure></div><div className="result-divider" /><div className="prob-section"><div className="prob-title"><span>ALL CLASS PROBABILITIES</span><button onClick={() => setShowDetails(!showDetails)}>{showDetails ? "Hide detail" : "Why all seven?"} <Info size={13} /></button></div><ProbabilityBars probs={result!.class_probabilities} />{showDetails && <div className="detail-note"><Info size={14} /><span>A high top score does not erase a meaningful second possibility. Melanoma and nevus confusion is real; the full spread is part of the result.</span></div>}</div><div className="result-meta"><div><span>EST. SKIN-TONE GROUP</span><strong>{result!.estimated_skin_tone}</strong><small>ITA={result!.ita_value.toFixed(1)}° · algorithmic estimate, not a clinical fact</small></div><div><span>UNCERTAINTY</span><strong className={result!.uncertainty && result!.uncertainty.level === "high" ? "text-coral" : ""}>{result!.uncertainty ? `${result!.uncertainty.level === "high" ? "High" : "Low"} · ${result!.uncertainty.entropy.toFixed(3)}` : "Not run"}</strong><small>{result!.uncertainty ? "Entropy score (30 stochastic passes)" : "Enable uncertainty-aware mode to estimate"}</small></div></div></div>}</div></div></div></section>

      <section id="fairness" className="fairness-section container"><div className="section-kicker"><span>03 / FAIRNESS & PERFORMANCE</span><span className="kicker-line" /><EvidenceStamp tone="sage">NO SPIN</EvidenceStamp></div><div className="fairness-heading"><div><h2>Better for whom?<br /><em>Measure it.</em></h2><p>These results are mixed — and not in the direction we hoped. The fairness-corrected models don't reliably narrow the skin-tone gap; on this data, plain baseline plus test-time augmentation currently has the smallest gap of any variant. The resolution-tuned fairness model has the best overall accuracy, but its melanoma recall is slightly lower than baseline's. We show all of it, including the parts that didn't work as intended.</p></div><div className="metric-note"><Gauge size={19} /><span>Primary metric<br /><b>max − min group accuracy</b></span></div></div>{metricsError ? <div className="error-banner">{metricsError}</div> : !variants ? <div className="metrics-loading"><Loader2 size={18} style={{ animation: "spin 1s linear infinite" }} /><div style={{ marginTop: 10 }}>Loading real evaluation results…</div></div> : <><div className="table-wrap"><table><thead><tr><th>MODEL VARIANT</th><th>OVERALL ACC.</th><th>MELANOMA RECALL</th><th>LIGHT</th><th>MEDIUM</th><th>DARK</th><th>FAIRNESS GAP</th></tr></thead><tbody>{variants.map((v) => <tr key={v.key} className={lowestGapVariant?.key === v.key ? "highlight-row" : ""}><td><span className="variant-dot" />{v.name}{lowestGapVariant?.key === v.key && <EvidenceStamp tone="sage">LOWEST GAP</EvidenceStamp>}{v.best && <EvidenceStamp tone="blue">BEST ACCURACY</EvidenceStamp>}</td><td><strong>{v.overall_accuracy.toFixed(1)}%</strong></td><td>{v.melanoma_recall != null ? `${v.melanoma_recall.toFixed(1)}%` : "—"}</td><td>{v.group_accuracy[GROUP_ORDER[0]]?.toFixed(1) ?? "—"}%</td><td>{v.group_accuracy[GROUP_ORDER[1]]?.toFixed(1) ?? "—"}%</td><td>{v.group_accuracy[GROUP_ORDER[2]]?.toFixed(1) ?? "—"}%</td><td className={lowestGapVariant?.key === v.key ? "good-gap" : ""}>{v.fairness_gap != null ? `${v.fairness_gap.toFixed(1)}%` : "—"}</td></tr>)}</tbody></table></div><div className="chart-card"><div className="chart-head"><div><span className="chart-eyebrow">GROUP ACCURACY BY VARIANT</span><h3>The full spread,<br /><em>not just the headline.</em></h3></div><div className="legend"><span><i className="legend-light" />Light</span><span><i className="legend-medium" />Medium</span><span><i className="legend-dark" />Dark</span></div></div><div className="bar-chart">{variants.map((v) => <div className="bar-group" key={v.key}><div className="bars"><span style={{ height: `${(v.group_accuracy[GROUP_ORDER[0]] ?? 0) / maxGroupAcc * 100}%`, background: "#7a9fdf" }} /><span style={{ height: `${(v.group_accuracy[GROUP_ORDER[1]] ?? 0) / maxGroupAcc * 100}%`, background: "#4f7dca" }} /><span style={{ height: `${(v.group_accuracy[GROUP_ORDER[2]] ?? 0) / maxGroupAcc * 100}%`, background: "#2769d8" }} /></div><small>{v.name.replace("Fairness + ", "F+").replace("Baseline + ", "B+").replace("Fairness-corrected", "Fair")}</small></div>)}</div><div className="axis-labels"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div></div></>}</section>

      <section id="method" className="method-section"><div className="container"><div className="section-kicker"><span>04 / METHOD & LIMITATIONS</span><span className="kicker-line" /></div><div className="method-grid"><div><h2>Trust is built in<br /><em>the footnotes.</em></h2><p className="method-intro">The most important result was the one that made our number smaller.</p><div className="bug-card"><div className="bug-top"><span className="bug-icon"><X size={17} /></span><EvidenceStamp tone="coral">BUG FOUND + FIXED</EvidenceStamp></div><h3>Lesion-level leakage inflated our first score.</h3><p>Naive train/test splits let near-duplicate photos of the same physical lesion appear in both sets. Removing those duplicates dropped accuracy from the low-80s to the mid-70s—the honest number.</p><div className="bug-numbers"><div><strong>80s</strong><span>inflated score</span></div><ArrowDownRight size={18} /><div className="honest-number"><strong>{baseline ? `${baseline.overall_accuracy.toFixed(1)}%` : "~75%"}</strong><span>lesion-level split</span></div></div></div></div><div className="method-list"><div className="method-item"><span className="method-num">01</span><div><h3>Datasets</h3><p>HAM10000 and ISIC2019, seven classes, with a lesion-level split to prevent near-duplicate leakage.</p></div></div><div className="method-item"><span className="method-num">02</span><div><h3>What the score means</h3><p>Confidence is the model's probability distribution—not a medical likelihood. The heatmap is a focus cue, not an explanation.</p></div></div><div className="method-item"><span className="method-num">03</span><div><h3>Known limitations</h3><p>Melanoma / nevus confusion, small rare-class samples, ITA-based tone estimation, and one Grad-CAM architectural quirk.</p></div></div></div></div></div></section>

      <section className="closing-section container"><div className="closing-card"><div><EvidenceStamp>THE TAKEAWAY</EvidenceStamp><h2>A careful model is<br /><em>more useful than a confident one.</em></h2></div><div><p>Use DermLens to inspect the behavior of a research prototype—not to replace the person who can examine you, ask questions, and notice what the camera cannot.</p><a className="button button-primary" href="#demo">Inspect a sample <ArrowUpRight size={16} /></a></div></div></section>
    </main>
    <footer className="site-footer"><div className="container footer-inner"><a className="brand" href="#top"><span className="brand-mark"><LogoMark /></span><span>Derm<span>Lens</span></span></a><span>Research prototype · v0.1</span><span>Made to show the gaps</span></div></footer>
  </div>;
}
