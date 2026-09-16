"""
llm_metric_resolver.py -- Phase 11 (metric-disambiguation part)

The ONLY thing an LLM is used for here: picking which already-discovered
metric a genuinely ambiguous query refers to, when deterministic matching
(metric_resolver.py) couldn't decide. It never sees raw data -- only the
query text and each candidate metric's key/field/display_name/semantic_type/
aliases. Its output is validated against the actual candidate list before
being trusted; an invalid or failed response changes nothing (the caller's
deterministic "ambiguous" response is used as-is).
"""
import json
import logging

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-5"


def resolve_ambiguous_metric_with_llm(query: str, candidates: list[dict]) -> dict | None:
    """
    Returns {"metric_key": ..., "reasoning": ...} on a validated success, or
    None if the LLM is unavailable, errors, or returns anything that isn't
    one of the actual candidate metric_keys. Callers MUST treat None as
    "stay with the deterministic ambiguous/no_match response" -- never as
    an error to surface to the user.
    """
    if not candidates:
        return None

    try:
        from anthropic import Anthropic
    except ImportError:
        logger.info("anthropic package unavailable; skipping LLM metric disambiguation.")
        return None

    candidate_payload = [
        {
            "metric_key": c["metric_key"],
            "field": c["field"],
            "display_name": c["display_name"],
        }
        for c in candidates
    ]

    prompt = f"""A user asked an analytics question about a dataset. Given the question and
the list of metrics that actually exist in this dataset, pick the ONE metric_key
the question most likely refers to. Only choose from the provided list -- never
invent a metric_key that isn't in it. If none of them plausibly match, say so.

User question: {query!r}

Available metrics (JSON):
{json.dumps(candidate_payload)}

Respond with only one raw JSON object, no markdown fences, no preamble:
{{"metric_key": "<one of the metric_key values above, or null if none plausibly match>", "reasoning": "<one short sentence>"}}
"""

    try:
        client = Anthropic()
        response = client.messages.create(
            model=MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = "".join(b.text for b in response.content if b.type == "text").strip()
        raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw_text)

        metric_key = parsed.get("metric_key")
        valid_keys = {c["metric_key"] for c in candidates}
        if metric_key not in valid_keys:
            logger.warning("LLM metric resolution returned an invalid/null metric_key (%r); ignoring.", metric_key)
            return None

        return {"metric_key": metric_key, "reasoning": parsed.get("reasoning", "")}

    except Exception as error:
        # Any failure here (network, invalid JSON, API error, etc.) must be
        # silent to the caller -- the deterministic response still works.
        logger.warning("LLM metric disambiguation failed (%s); falling back to deterministic result.", error)
        return None