// Central API client for the FastAPI backend.
//
//   GET /api/data                -> { records, record_count }              [V1]
//   GET /api/analyze?question=   -> { request, result, insights,
//                                      chart_spec, chart_data, prediction,
//                                      explanation, errors }               [V1, legacy]
//   GET /api/v2/profile          -> DatasetRuntimeConfig                   [V2]
//   GET /api/v2/analyze          -> unified analysis/forecast response     [V2, primary]
//   GET /api/v2/forecast         -> forecast-only response                 [V2, diagnostic]
//
// V1 functions (fetchOverview/analyze) are kept for backward compatibility
// with any code still exercising the legacy path, but the app's primary
// flow uses only the V2 functions below.

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function toError(res) {
  let detail = `Request failed (${res.status})`;
  try {
    const body = await res.json();
    if (body?.detail) detail = body.detail;
  } catch {
    // response wasn't JSON -- keep the generic message
  }
  const error = new Error(detail);
  error.status = res.status;
  return error;
}

// One reusable GET helper: builds a query string from a plain params
// object (omitting undefined/null/empty-string values), parses JSON,
// throws a structured Error with a `.status` on non-2xx responses, and
// supports cancellation via AbortSignal.
async function request(path, params = {}, { signal } = {}) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    search.set(key, String(value));
  }
  const qs = search.toString();
  const url = `${API_BASE}${path}${qs ? `?${qs}` : ""}`;

  let res;
  try {
    res = await fetch(url, { signal });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    const wrapped = new Error(
      `Couldn't reach the backend at ${API_BASE}. Is the server running?`
    );
    wrapped.cause = err;
    throw wrapped;
  }

  if (!res.ok) throw await toError(res);

  try {
    return await res.json();
  } catch (err) {
    const wrapped = new Error("Backend returned a non-JSON response.");
    wrapped.cause = err;
    throw wrapped;
  }
}

// --- V2 (primary) ---------------------------------------------------------

export function getDatasetProfile({ dataset } = {}, { signal } = {}) {
  return request("/api/v2/profile", { dataset }, { signal });
}

export function analyzeDataset(
  question,
  {
    dataset,
    metric,
    granularity,
    horizon,
    useLlmFallback,
    conversationId,
  } = {},
  { signal } = {}
) {
  return request(
    "/api/v2/analyze",
    {
      question,
      dataset,
      conversation_id: conversationId,
      metric,
      granularity,
      horizon,
      use_llm_fallback: useLlmFallback === true ? true : undefined,
    },
    { signal }
  );
}

// Diagnostic/compatibility helper only -- the app's primary flow never
// calls this directly; forecast questions go through analyzeDataset()
// above, which the backend routes internally.
export function forecastDataset(
  question,
  { dataset, metric, granularity, horizon, useLlmFallback } = {},
  { signal } = {}
) {
  return request(
    "/api/v2/forecast",
    {
      question,
      dataset,
      metric,
      granularity,
      horizon,
      use_llm_fallback: useLlmFallback === true ? true : undefined,
    },
    { signal }
  );
}

// --- V1 (legacy, kept for backward compatibility only) --------------------

export async function fetchOverview() {
  const res = await fetch(`${API_BASE}/api/data`);
  if (!res.ok) throw await toError(res);
  return res.json();
}

export async function analyze(question) {
  const url = `${API_BASE}/api/analyze?question=${encodeURIComponent(question)}`;
  const res = await fetch(url);
  if (!res.ok) throw await toError(res);
  return res.json();
}

export { API_BASE };
