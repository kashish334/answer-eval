import React, { useState, useEffect, useCallback } from "react";
import { evaluateSheet, checkHealth } from "./api.js";

const CONFIDENCE_LABEL = {
  high: "High confidence",
  medium: "Medium confidence",
  low: "Needs review",
};

function ConfidenceTag({ level }) {
  return (
    <span className={`tag tag--${level}`}>
      {CONFIDENCE_LABEL[level] || level}
    </span>
  );
}

function ResultRow({ result, index }) {
  const [open, setOpen] = useState(false);
  const pct = result.max_score ? Math.round((result.score / result.max_score) * 100) : 0;

  return (
    <article className="result">
      <button className="result__head" onClick={() => setOpen((o) => !o)}>
        <div className="result__head-left">
          <span className="result__qno">Q{result.question_no}</span>
          <span className="result__qtext">{result.question_text}</span>
        </div>
        <div className="result__head-right">
          <ConfidenceTag level={result.confidence} />
          <span className="result__score">
            {result.score}<span className="result__score-max">/{result.max_score}</span>
          </span>
          <span className={`result__chevron ${open ? "is-open" : ""}`}>›</span>
        </div>
      </button>

      {open && (
        <div className="result__body">
          {result.spans_multiple_pages && (
            <p className="result__note">
              Extracted answer spans pages {result.source_pages.join(" \u2192 ")}.
            </p>
          )}

          <div className="result__block">
            <h4>Extracted answer</h4>
            <p className="result__answer">
              {result.extracted_answer_text || (
                <em className="muted">No text extracted.</em>
              )}
            </p>
          </div>

          <div className="result__grid">
            <div>
              <h4>Matched rubric points</h4>
              <ul className="result__points result__points--matched">
                {result.matched_points?.length
                  ? result.matched_points.map((p, i) => <li key={i}>{p}</li>)
                  : <li className="muted">None</li>}
              </ul>
            </div>
            <div>
              <h4>Missed rubric points</h4>
              <ul className="result__points result__points--missed">
                {result.missed_points?.length
                  ? result.missed_points.map((p, i) => <li key={i}>{p}</li>)
                  : <li className="muted">None</li>}
              </ul>
            </div>
          </div>

          <div className="result__block">
            <h4>Score rationale</h4>
            <p className="result__rationale">{result.score_vs_answer_key}</p>
          </div>

          <div className="result__block">
            <h4>Confidence reason</h4>
            <p className="result__rationale">{result.confidence_reason}</p>
          </div>
        </div>
      )}
    </article>
  );
}

function Summary({ results }) {
  const total = results.reduce((s, r) => s + r.score, 0);
  const max = results.reduce((s, r) => s + r.max_score, 0);
  const flagged = results.filter((r) => r.confidence === "low").length;

  return (
    <div className="summary">
      <div className="summary__stat">
        <span className="summary__value">{total}<span className="summary__value-max">/{max}</span></span>
        <span className="summary__label">Total score</span>
      </div>
      <div className="summary__divider" />
      <div className="summary__stat">
        <span className="summary__value">{results.length}</span>
        <span className="summary__label">Questions evaluated</span>
      </div>
      <div className="summary__divider" />
      <div className="summary__stat">
        <span className="summary__value">{flagged}</span>
        <span className="summary__label">Flagged for review</span>
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [backendUp, setBackendUp] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    checkHealth().then(setBackendUp);
  }, []);

  const handleFile = useCallback((f) => {
    if (f && f.type === "application/pdf") {
      setFile(f);
      setError(null);
    } else {
      setError("Please choose a PDF file.");
    }
  }, []);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      handleFile(e.dataTransfer.files?.[0]);
    },
    [handleFile]
  );

  const runEvaluation = async () => {
    if (!file) return;
    setStatus("loading");
    setError(null);
    try {
      const data = await evaluateSheet(file);
      setResults(data.results);
      setStatus("done");
    } catch (e) {
      setError(e.message);
      setStatus("error");
    }
  };

  return (
    <div className="page">
      <header className="header">
        <div className="header__inner">
          <p className="eyebrow">CNV AI Labs</p>
          <h1>Answer Sheet Evaluation</h1>
          <p className="header__desc">
            Upload a scanned answer sheet. The pipeline extracts each answer,
            scores it against the rubric by meaning, and flags what it's unsure of.
          </p>
        </div>
        <div className={`status-dot ${backendUp ? "status-dot--up" : "status-dot--down"}`}>
          <span className="status-dot__mark" />
          {backendUp === null ? "Checking backend…" : backendUp ? "Backend connected" : "Backend unreachable"}
        </div>
      </header>

      <main className="main">
        <section
          className={`dropzone ${dragOver ? "dropzone--active" : ""} ${file ? "dropzone--filled" : ""}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        >
          <input
            id="file-input"
            type="file"
            accept="application/pdf"
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
          {file ? (
            <>
              <p className="dropzone__filename">{file.name}</p>
              <p className="dropzone__hint">
                <label htmlFor="file-input" className="link">Choose a different file</label>
              </p>
            </>
          ) : (
            <>
              <p className="dropzone__title">Drop the answer sheet PDF here</p>
              <p className="dropzone__hint">
                or <label htmlFor="file-input" className="link">browse files</label>
              </p>
            </>
          )}
        </section>

        <div className="actions">
          <button
            className="button"
            disabled={!file || status === "loading"}
            onClick={runEvaluation}
          >
            {status === "loading" ? "Evaluating…" : "Run evaluation"}
          </button>
          {error && <p className="error-text">{error}</p>}
        </div>

        {status === "loading" && (
          <p className="loading-text">
            Running extraction, structuring, scoring, and confidence flagging — this can take a little while.
          </p>
        )}

        {status === "done" && results.length > 0 && (
          <section className="results">
            <Summary results={results} />
            <div className="results__list">
              {results.map((r, i) => (
                <ResultRow key={r.question_no} result={r} index={i} />
              ))}
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        Extract → Structure → Score → Confidence
      </footer>
    </div>
  );
}
