"""
metric_resolver.py -- Phase 5

Resolves a user's query (plus optionally already-extracted candidate
metric names from a request analyzer) against the CURRENT dataset's
discovered metrics. Replaces hardcoded forecast-metric resolution in
router.py, and its unconditional "default to revenue" fallback.

Never returns a metric the caller didn't earn through matching. Ambiguity
is returned explicitly as a candidate list, not silently resolved.
"""
import re

STOPWORDS = {"the", "a", "an", "of", "for", "in", "on", "to", "and", "next",
             "please", "show", "me", "what", "was", "is", "our", "give"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and t not in STOPWORDS}


def _normalize(text: str) -> str:
    return re.sub(r"[_\-\s]+", " ", text.lower()).strip()


def resolve_metric(
    query: str,
    config: dict,
    requested_metrics: list[str] | None = None,
) -> dict:
    """
    Resolution order: exact column-name match -> exact display-name match ->
    alias match -> token-overlap fuzzy match -> a single forecastable metric
    (only if nothing else in the dataset could match) -> ambiguous candidates.

    Returns:
      resolved dataset  -> {"resolved": True, "metric_key", "field",
                             "resolution_method", "resolution_confidence"}
      ambiguous dataset -> {"resolved": False, "candidates": [...],
                             "resolution_method": "ambiguous"}
      nothing matched   -> {"resolved": False, "candidates": [],
                             "resolution_method": "no_match"}
    """
    metrics = config.get("metrics", {})
    if not metrics:
        return {"resolved": False, "candidates": [], "resolution_method": "no_match",
                "resolution_confidence": 0.0, "message": "no metrics discovered for this dataset"}

    query_norm = _normalize(query or "")
    query_tokens = _tokens(query or "")

    # 1. exact column-name match (word-boundary, so "sale" inside "sales_amount" doesn't misfire)
    for key, spec in metrics.items():
        if re.search(rf"\b{re.escape(spec['field'].lower())}\b", query_norm):
            return _result(spec, "exact_column_match", 1.0)

    # 2. exact normalized display-name match
    for key, spec in metrics.items():
        if _normalize(spec["display_name"]) in query_norm:
            return _result(spec, "display_name_match", 0.95)

    # 3. alias match
    for key, spec in metrics.items():
        for alias in spec.get("aliases", []):
            if re.search(rf"\b{re.escape(alias)}\b", query_norm):
                return _result(spec, "alias_match", 0.85)

    # 4. token-overlap fuzzy match
    best_key, best_score = None, 0.0
    scored = []
    for key, spec in metrics.items():
        candidate_tokens = _tokens(spec["field"]) | _tokens(spec["display_name"])
        for alias in spec.get("aliases", []):
            candidate_tokens |= _tokens(alias)
        if not candidate_tokens or not query_tokens:
            continue
        overlap = len(query_tokens & candidate_tokens)
        if overlap == 0:
            continue
        score = overlap / len(candidate_tokens | query_tokens)  # jaccard
        scored.append((key, score))
        if score > best_score:
            best_key, best_score = key, score

    if scored:
        scored.sort(key=lambda kv: kv[1], reverse=True)
        top = [k for k, s in scored if s >= best_score * 0.8]  # near-ties = ambiguous
        if len(top) == 1 and best_score >= 0.2:
            return _result(metrics[best_key], "fuzzy_token_match", round(best_score, 2))
        if len(top) > 1 and best_score >= 0.2:
            return {
                "resolved": False, "resolution_method": "ambiguous",
                "resolution_confidence": round(best_score, 2),
                "candidates": [_candidate(metrics[k]) for k in top],
                "message": f"query matched {len(top)} metrics about equally well",
            }

    # 5. metrics already extracted by a request-analyzer pass, mapped
    #    through the SAME discovered metrics (never a fixed vocabulary)
    if requested_metrics:
        matched = [m for m in requested_metrics if m in metrics]
        if len(matched) == 1:
            return _result(metrics[matched[0]], "request_analyzer_match", 0.7)
        if len(matched) > 1:
            return {
                "resolved": False, "resolution_method": "ambiguous",
                "resolution_confidence": 0.5,
                "candidates": [_candidate(metrics[k]) for k in matched],
                "message": "multiple metrics were extracted from the query",
            }

    # 6. exactly one forecastable metric exists in the whole dataset -- safe
    #    to select it, but say so explicitly (never silent).
    forecastable = [k for k, s in metrics.items() if s.get("forecastable")]
    if len(forecastable) == 1:
        return _result(metrics[forecastable[0]], "only_forecastable_metric", 0.6)

    # 7. nothing resolved -- return candidates (all forecastable metrics) for
    #    the caller to present, rather than guessing.
    return {
        "resolved": False,
        "resolution_method": "no_match",
        "resolution_confidence": 0.0,
        "candidates": [_candidate(metrics[k]) for k in forecastable] or
                       [_candidate(s) for s in metrics.values()],
        "message": "no metric in the query matched any discovered column, alias, or display name",
    }


def resolve_all_mentioned_metrics(query: str, config: dict) -> list[dict]:
    """
    Like resolve_metric(), but returns EVERY discovered metric the query
    plausibly mentions (exact column/display-name/alias match), instead of
    stopping at the single best match. Used for "compare X and Y" style
    queries where more than one metric is expected. Longest/most-specific
    field name is checked first so a short alias can't shadow a more
    specific metric that's also present.
    """
    metrics = config.get("metrics", {})
    if not metrics:
        return []

    query_norm = _normalize(query or "")
    matched = {}

    # longest field name first, so e.g. "invoice_value" is checked before a
    # shorter, more generic metric name that might otherwise false-match.
    ordered = sorted(metrics.items(), key=lambda kv: len(kv[1]["field"]), reverse=True)

    for key, spec in ordered:
        if key in matched:
            continue
        if re.search(rf"\b{re.escape(spec['field'].lower())}\b", query_norm):
            matched[key] = spec
            continue
        if _normalize(spec["display_name"]) in query_norm:
            matched[key] = spec
            continue
        for alias in spec.get("aliases", []):
            if re.search(rf"\b{re.escape(alias)}\b", query_norm):
                matched[key] = spec
                break

    return [_candidate(spec) for spec in matched.values()]


def _result(spec: dict, method: str, confidence: float) -> dict:
    return {
        "resolved": True,
        "metric_key": spec["metric_key"],
        "field": spec["field"],
        "display_name": spec["display_name"],
        "resolution_method": method,
        "resolution_confidence": confidence,
    }


def _candidate(spec: dict) -> dict:
    return {"metric_key": spec["metric_key"], "field": spec["field"], "display_name": spec["display_name"]}