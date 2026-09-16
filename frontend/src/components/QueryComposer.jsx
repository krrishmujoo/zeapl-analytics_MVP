import { useId, useState } from "react";

const HISTORY_KEY = "ledger-os:query-history";
const HISTORY_LIMIT = 8;
const GRANULARITIES = ["day", "week", "month"];

// Mirrors request_analyser.py's EXPLANATION_TERMS on the backend: those are
// exactly the words that make the backend's own `requires_llm` flag true on
// the legacy V1 path. The V2 dynamic endpoint only calls the LLM when the
// caller explicitly passes use_llm_fallback=true, so a "why did X happen"
// question would otherwise get a flat trend sentence instead of a real
// explanation unless the person remembers to check "Allow LLM fallback"
// under Advanced options. This detects that intent automatically.
const EXPLANATION_TERMS = /\b(why|explain|reason|reasons|cause|caused|interpret)\b/i;

// --- persisted question history -----------------------------------------
// Small, best-effort localStorage helpers. Any storage failure (private
// browsing quota, disabled storage, etc.) degrades to "no history" rather
// than throwing -- history is a convenience, not a requirement.

export function loadHistory() {
  if (typeof window === "undefined" || !window.localStorage) return [];
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((q) => typeof q === "string") : [];
  } catch {
    return [];
  }
}

export function pushHistory(question) {
  const trimmed = (question || "").trim();
  if (!trimmed) return loadHistory();
  const next = [trimmed, ...loadHistory().filter((q) => q !== trimmed)].slice(0, HISTORY_LIMIT);
  try {
    window.localStorage?.setItem(HISTORY_KEY, JSON.stringify(next));
  } catch {
    // storage unavailable -- not fatal, just don't persist
  }
  return next;
}

export function clearHistory() {
  try {
    window.localStorage?.removeItem(HISTORY_KEY);
  } catch {
    // Storage unavailable — local React state will still be cleared.
  }

  return [];
}


export default function QueryComposer({
  suggestions = [],
  history = [],
  isLoading = false,
  onSubmit,
  onCancel,
  forecastableMetrics = [],
}) {
  const [value, setValue] = useState("");
  const [metric, setMetric] = useState("");
  const [granularity, setGranularity] = useState("");
  const [horizon, setHorizon] = useState("");
  const [useLlmFallback, setUseLlmFallback] = useState(false);
  const inputId = useId();

  function submit(question) {
    const q = (question ?? value).trim();
    if (!q || isLoading) return;
    // The checkbox can force it on for anything; an explanation-shaped
    // question forces it on too, even if the checkbox was never touched.
    const wantsExplanation = EXPLANATION_TERMS.test(q);
    onSubmit?.(q, {
      metric: metric || undefined,
      granularity: granularity || undefined,
      horizon: horizon ? Number(horizon) : undefined,
      useLlmFallback: useLlmFallback || wantsExplanation,
    });
    setValue("");
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <section className="glass command-hero">
      <span className="command-eyebrow">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 3v3m0 12v3m9-9h-3M6 12H3m14.5-6.5-2 2m-9 9-2 2m0-13 2 2m9 9 2 2" strokeLinecap="round" />
        </svg>
        Ask the dataset
      </span>
      <h2 className="command-heading">
        Ask a question in plain language &mdash; totals, breakdowns, comparisons, or a forecast.
      </h2>

      <label className="sr-only" htmlFor={inputId}>
        Ask a question about your dataset
      </label>
      <div className="command-row">
        <div className="command-field">
          <textarea
            id={inputId}
            rows={1}
            placeholder="e.g. Forecast invoice value for the next 4 weeks"
            value={value}
            disabled={isLoading}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={handleKeyDown}
          />
        </div>
        {isLoading ? (
          <button type="button" className="btn btn-ghost" onClick={onCancel}>
            Cancel request
          </button>
        ) : (
          <button type="button" className="btn btn-primary" onClick={() => submit()}>
            Ask
          </button>
        )}
      </div>

      {suggestions.length > 0 && (
        <div className="chip-row">
          {suggestions.map((s) => {
            const label = typeof s === "string" ? s : s.label;
            const question = typeof s === "string" ? s : s.question || s.label;
            return (
              <button key={label} type="button" className="chip" onClick={() => submit(question)}>
                {label}
              </button>
            );
          })}
        </div>
      )}

      {history.length > 0 && (
        <div className="history-row">
          <span className="history-label">Recent</span>
          {history.map((q) => (
            <button key={q} type="button" className="chip chip-ghost" onClick={() => submit(q)}>
              {q}
            </button>
          ))}
        </div>
      )}

      <details className="advanced-disclosure">
        <summary>Advanced options</summary>
        <div className="advanced-grid">
          <label className="advanced-field">
            Metric
            <select value={metric} onChange={(event) => setMetric(event.target.value)}>
              <option value="">Auto-detect</option>
              {forecastableMetrics.map((m) => (
                <option key={m.key} value={m.key}>
                  {m.displayName}
                </option>
              ))}
            </select>
          </label>
          <label className="advanced-field">
            Granularity
            <select value={granularity} onChange={(event) => setGranularity(event.target.value)}>
              <option value="">Auto-detect</option>
              {GRANULARITIES.map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
          </label>
          <label className="advanced-field">
            Horizon
            <input
              type="number"
              min="1"
              placeholder="e.g. 4"
              value={horizon}
              onChange={(event) => setHorizon(event.target.value)}
            />
          </label>
          <label className="advanced-field advanced-field-checkbox">
            <input
              type="checkbox"
              checked={useLlmFallback}
              onChange={(event) => setUseLlmFallback(event.target.checked)}
            />
            Allow LLM fallback
          </label>
        </div>
      </details>
    </section>
  );
}
