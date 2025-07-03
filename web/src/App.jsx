import { useState } from 'react';
import './App.css';

async function predict(text) {
  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export default function App() {
  const [showDemo, setShowDemo] = useState(false);
  const [text, setText]         = useState('');
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState(null);

  async function handleDetect(e) {
    e.preventDefault();
    setLoading(true); setError(null); setResult(null);
    try       { setResult(await predict(text)); }
    catch (e) { setError(e.message);            }
    finally   { setLoading(false);              }
  }

  return (
    <>
      {/* ── top bar ───────────────────────────── */}
      <nav className="nav">
        <div className="nav-inner">
          <div className="nav-links">
            <a href="#how">How&nbsp;it&nbsp;works</a>
            <a href="#demo" onClick={() => setShowDemo(true)}>Try&nbsp;it</a>
          </div>
        </div>
      </nav>

      {/* ── hero ─────────────────────────────── */}
      <header className="hero" id="how">
        <div className="hero-content">
          <h1 className="headline">AI Fake-News Detector</h1>
          <p className="sub">
            Paste any headline or paragraph and our model will tell you whether
            it’s likely <strong>real</strong> or <strong>fake</strong>.
          </p>
          <button className="cta" onClick={() => setShowDemo(true)}>
            Try the live demo
          </button>
        </div>
      </header>

      {/* ── floating analyser ─────────────────── */}
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

            {error  && <p className="error">⚠ {error}</p>}
            {result && (
              <p className={`verdict ${result.label}`}>
                {result.label.toUpperCase()} ({(result.prob * 100).toFixed(1)} %)
              </p>
            )}
          </div>
        </div>
      )}
    </>
  );
}
