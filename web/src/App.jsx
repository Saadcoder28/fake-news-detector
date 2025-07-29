// src/App.jsx
import React, { useState, useEffect } from 'react';
import './App.css';

// ── helpers — your existing API functions ─────────────────────────
const get  = url      => fetch(url).then(r => r.json());
const post = (url, d) => fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(d),
}).then(r => r.json());

// ── Custom doughnut chart from your existing code ────────────────
function CustomDoughnut({ real, fake }) {
  const total = real + fake;
  const realPercentage = total ? (real / total) * 100 : 0;
  const fakePercentage = total ? (fake / total) * 100 : 0;
  const radius = 45, center = 55;

  // real arc
  const realAngle = (realPercentage / 100) * 360;
  const realStart = -90;
  const realEnd = realStart + realAngle;
  const realStartX = center + radius * Math.cos((realStart * Math.PI) / 180);
  const realStartY = center + radius * Math.sin((realStart * Math.PI) / 180);
  const realEndX   = center + radius * Math.cos((realEnd   * Math.PI) / 180);
  const realEndY   = center + radius * Math.sin((realEnd   * Math.PI) / 180);
  const realLarge  = realAngle > 180 ? 1 : 0;
  const realPath   = `M ${center} ${center} L ${realStartX} ${realStartY} A ${radius} ${radius} 0 ${realLarge} 1 ${realEndX} ${realEndY} Z`;

  // fake arc
  const fakeStart   = realEnd;
  const fakeAngle   = (fakePercentage / 100) * 360;
  const fakeEnd     = fakeStart + fakeAngle;
  const fakeStartX  = center + radius * Math.cos((fakeStart * Math.PI) / 180);
  const fakeStartY  = center + radius * Math.sin((fakeStart * Math.PI) / 180);
  const fakeEndX    = center + radius * Math.cos((fakeEnd   * Math.PI) / 180);
  const fakeEndY    = center + radius * Math.sin((fakeEnd   * Math.PI) / 180);
  const fakeLarge   = fakeAngle > 180 ? 1 : 0;
  const fakePath    = `M ${center} ${center} L ${fakeStartX} ${fakeStartY} A ${radius} ${radius} 0 ${fakeLarge} 1 ${fakeEndX} ${fakeEndY} Z`;

  return (
    <svg width="110" height="110" viewBox="0 0 110 110">
      <circle cx={center} cy={center} r={radius}
        fill="none" stroke="#333" strokeWidth="2" />
      {real > 0 && <path d={realPath} fill="#4ade80" opacity="0.8" />}
      {fake > 0 && <path d={fakePath} fill="#f87171" opacity="0.8" />}
    </svg>
  );
}

// ── HowItWorks page component ─────────────────────────────────────
function HowItWorksPage({ onBack }) {
  return (
    <div className="page-container">
      <nav className="nav">
        <div className="nav-inner">
          <div className="nav-links">
            <a href="#" onClick={onBack}>← Back to Home</a>
          </div>
        </div>
      </nav>
      
      <div className="how-it-works-page">
        <section className="how-it-works">
          <h2>How It Works</h2>
          <p>
            Our Fake‑News Detector uses a fine‑tuned <strong>RoBERTa</strong> 
            transformer from Hugging Face. Incoming text is tokenized with 
            <code>AutoTokenizer</code>, passed through a sequence‑classification 
            head, and returns a probability for two classes: <em>real</em> vs. <em>fake</em>.
          </p>
          <h3>Model &amp; Data</h3>
          <ul>
            <li><strong>Base:</strong> <code>roberta-base</code></li>
            <li><strong>Training set:</strong> ≈10 000 mixed real‑news & disinfo headlines</li>
            <li><strong>Fine‑tuning:</strong> 3 epochs, AdamW, max length 512 tokens</li>
          </ul>
          <h3>Request Flow</h3>
          <ol>
            <li>Client → <code>POST /predict</code> (FastAPI + Gunicorn + Uvicorn)</li>
            <li>Server tokenizes & runs inference on CPU/GPU → returns <code>label</code> + <code>prob</code></li>
            <li>Frontend displays verdict; "Why?" triggers a token‑masking explanation</li>
          </ol>
          <p className="small">
            Live stats (last 1 000 queries) are kept in‑memory—no external database.
          </p>
        </section>
      </div>
    </div>
  );
}

// ── Main Home page component ──────────────────────────────────────
function HomePage({ onShowDemo, onShowHowItWorks, stats }) {
  return (
    <>
      <nav className="nav">
        <div className="nav-inner">
          <div className="nav-links-left">
            <a href="#" onClick={onShowHowItWorks}>How&nbsp;it&nbsp;works</a>
          </div>
          <div className="nav-links-right">
            <a href="#" onClick={onShowDemo}>Try&nbsp;it</a>
          </div>
        </div>
      </nav>

      <header className="hero">
        <div className="hero-content">
          <h1 className="headline">AI Fake‑News Detector</h1>
          <p className="sub">
            Paste any headline or paragraph and our model will tell you whether it's likely&nbsp;
            <strong>real</strong> or <strong>fake</strong>.
          </p>
          <button className="cta" onClick={onShowDemo}>
            Try the live demo
          </button>
        </div>
      </header>

      {stats && <StatsBox {...stats} />}
    </>
  );
}

export default function App() {
  const [currentPage, setCurrentPage] = useState('home'); // 'home' or 'how-it-works'
  const [showDemo, setShowDemo] = useState(false);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [explain, setExplain] = useState(null);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);

  // poll /stats every 15s
  useEffect(() => {
    let t;
    const loop = async () => {
      try { setStats(await get('/stats')); } catch {}
      t = setTimeout(loop, 15000);
    };
    loop();
    return () => clearTimeout(t);
  }, []);

  async function handleDetect(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setExplain(null);
    try {
      setResult(await post('/predict', { text }));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleExplain() {
    try {
      setExplain(await get(`/explain/${result.id}`));
    } catch (e) {
      setError(e.message);
    }
  }

  const handleShowDemo = () => setShowDemo(true);
  const handleShowHowItWorks = () => setCurrentPage('how-it-works');
  const handleBackToHome = () => setCurrentPage('home');

  return (
    <>
      {currentPage === 'home' && (
        <HomePage 
          onShowDemo={handleShowDemo}
          onShowHowItWorks={handleShowHowItWorks}
          stats={stats}
        />
      )}
      
      {currentPage === 'how-it-works' && (
        <HowItWorksPage onBack={handleBackToHome} />
      )}

      {showDemo && (
        <div className="overlay" onClick={() => setShowDemo(false)}>
          <div className="card" onClick={e => e.stopPropagation()}>
            <h2>Analyse text</h2>
            <form onSubmit={handleDetect}>
              <textarea
                rows="5"
                placeholder="Paste text…"
                value={text}
                onChange={e => setText(e.target.value)}
                required
              />
              <button disabled={!text || loading}>
                {loading ? 'Checking…' : 'Detect'}
              </button>
            </form>
            {error && <p className="error">⚠ {error}</p>}
            {result && (
              <>
                <p className={`verdict ${result.label}`}>
                  {result.label.toUpperCase()} ({(result.prob * 100).toFixed(1)}%)
                </p>
                <button className="explain-btn" onClick={handleExplain}>
                  Why?
                </button>
              </>
            )}
            {explain && (
              <ul className="explain">
                {explain.words.map((w, i) => {
                  const cls = explain.contrib[i] >= 0 ? 'good' : 'bad';
                  return (
                    <li key={i}>
                      <span className={cls}>
                        {explain.contrib[i] >= 0 ? '+' : ''}
                        {(explain.contrib[i] * 100).toFixed(1)}
                      </span>
                      <span className="token">{w}</span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>
      )}
    </>
  );
}

// ── existing StatsBox & Bar components ─────────────────────────
function StatsBox({ n, real, fake, avg }) {
  return (
    <div className="stats">
      <h3>Live stats (last 1 000)</h3>
      <CustomDoughnut real={real} fake={fake} />
      <div className="bars" style={{ marginTop: '.7rem' }}>
        <Bar label="Real" value={real} total={n} color="#4ade80" />
        <Bar label="Fake" value={fake} total={n} color="#f87171" />
      </div>
      <p className="count">
        {n} analysed • avg confidence {(avg * 100).toFixed(0)}%
      </p>
    </div>
  );
}

function Bar({ label, value, total, color }) {
  const pct = total ? (value / total) * 100 : 0;
  return (
    <div className="bar">
      <span>{label}</span>
      <div className="track">
        <div className="fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span>{value}</span>
    </div>
  );
}