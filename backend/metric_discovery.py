"""
metric_discovery.py -- Phase 3

Turns a DatasetProfile's candidate_metrics into a dict of MetricSpecs:
field, display name, semantic type, inferred aggregation, non-negative/
additive/forecastable flags, aliases, and a discovery confidence score.
Replaces registries.PredictionMetrics / METRIC_TO_FIELD -- nothing here is
tied to any particular dataset's column names.

Results are cached to disk per dataset fingerprint (schemas/<fingerprint>.json)
so re-analyzing the same dataset doesn't re-run discovery every request.
"""
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_CACHE_DIR = Path(__file__).parent / "schemas"
DISCOVERY_VERSION = 3

# ---------------------------------------------------------------------------
# Aggregation-inference keyword hints (weak signals, combined with semantic
# type and distribution -- never the sole basis for a decision).
# ---------------------------------------------------------------------------
SUM_HINTS = ("revenue", "sales", "quantity", "units", "cost", "spend", "usage",
             "minutes", "points", "amount", "transactions", "downtime", "energy")
MEAN_HINTS = ("temperature", "rating", "score", "percentage", "pct", "rate",
              "conversion", "utilization", "latency", "price", "average", "aov", "margin")
COUNT_HINTS = ("event", "transaction", "order", "scan", "ticket", "incident")
DISTINCT_HINTS = ("customer_id", "user_id", "order_id", "device_id", "invoice_id", "session_id")

STOPWORDS = {"the", "a", "an", "of", "per", "total", "amount"}


def _tokenize(name: str) -> list[str]:
    return [t for t in re.split(r"[_\s\-]+", name.lower()) if t and t not in STOPWORDS]


def _display_name(field: str) -> str:
    return " ".join(w.capitalize() for w in _tokenize(field))


def _aliases(field: str) -> list[str]:
    tokens = _tokenize(field)
    aliases = {field.lower(), field.lower().replace("_", " "), " ".join(tokens)}
    # cheap singular/plural variants
    for t in list(aliases):
        if t.endswith("s") and len(t) > 3:
            aliases.add(t[:-1])
        else:
            aliases.add(t + "s")
    aliases.discard("")
    return sorted(aliases)

def _identifier_aliases(field: str) -> list[str]:
    """
    Generate natural-language aliases for identifier count metrics.

    Examples:
        scan_id    -> scan, scans, scan count, total scans
        user_id    -> user, users, user count, total users
        product_id -> product, products, product count
    """
    normalized = field.lower().strip()

    if normalized.endswith("_id"):
        base = normalized[:-3]
    elif normalized.endswith(" id"):
        base = normalized[:-3]
    else:
        base = normalized

    base = base.replace("_", " ").strip()

    if not base:
        return _aliases(field)

    if base.endswith("s"):
        singular = base[:-1]
        plural = base
    else:
        singular = base
        plural = f"{base}s"

    aliases = {
        field.lower(),
        field.lower().replace("_", " "),
        singular,
        plural,
        f"{singular} count",
        f"{plural} count",
        f"total {plural}",
        f"number of {plural}",
        f"distinct {singular}",
        f"distinct {plural}",
        f"unique {singular}",
        f"unique {plural}",
    }

    aliases.discard("")
    return sorted(aliases)


def _infer_aggregation(field: str, col_profile: dict) -> tuple[str, list[str], float]:
    """Returns (default_aggregation, allowed_aggregations, confidence)."""
    name = field.lower()
    semantic_type = col_profile["semantic_type"]

    if semantic_type == "boolean":
        return "sum", ["sum", "mean", "count"], 0.75  # sum = true-count, mean = rate

    if semantic_type == "percentage":
        return "mean", ["mean", "min", "max"], 0.9

    if any(h in name for h in MEAN_HINTS):
        return "mean", ["mean", "min", "max", "median"], 0.85

    if semantic_type == "monetary":
        return "sum", ["sum", "mean", "min", "max"], 0.9

    if any(h in name for h in SUM_HINTS):
        return "sum", ["sum", "mean", "min", "max"], 0.85

    if semantic_type == "discrete_numeric":
        return "sum", ["sum", "mean", "min", "max"], 0.55

    # continuous_numeric with no naming hint at all -- distribution-based
    # fallback: a column that's mostly non-zero and spread out behaves more
    # like a rate/level (mean-worthy) than an additive flow; low confidence
    # either way, flagged for possible confirmation downstream.
    zero_ratio = col_profile.get("zero_ratio") or 0.0
    if zero_ratio < 0.05:
        return "mean", ["mean", "sum", "min", "max"], 0.4
    return "sum", ["sum", "mean", "min", "max"], 0.4


SIGNED_NAME_HINTS = ("profit", "margin", "change", "delta", "temperature", "balance",
                     "variance", "deviation", "net", "growth", "return")

# Below this fraction, negative values look like sporadic data-entry errors
# rather than a legitimately signed quantity.
LIKELY_ERROR_NEGATIVE_RATIO = 0.1


def _infer_non_negative(field: str, col_profile: dict) -> bool:
    negative_ratio = col_profile.get("negative_ratio") or 0.0
    if negative_ratio == 0.0:
        return True

    # A handful of negative values in a column that isn't semantically signed
    # (e.g. a few negative "quantity" rows) look like data errors, not a
    # legitimate negative range -- treat as non-negative so the cleaning
    # pipeline strips them. A substantial negative fraction, or a name that
    # suggests a genuinely signed quantity (profit, temperature, ...), means
    # negatives are real and must NOT be scrubbed.
    looks_signed_by_name = any(h in field.lower() for h in SIGNED_NAME_HINTS)
    if not looks_signed_by_name and negative_ratio <= LIKELY_ERROR_NEGATIVE_RATIO:
        return True
    return False


def _infer_additive(default_aggregation: str, semantic_type: str) -> bool:
    if semantic_type == "percentage":
        return False
    return default_aggregation == "sum"


def _fingerprint_dir(dataset_fingerprint: str | None) -> Path | None:
    if not dataset_fingerprint:
        return None
    return SCHEMA_CACHE_DIR / f"{dataset_fingerprint}.json"


def _load_cached(dataset_fingerprint: str | None) -> dict | None:
    path = _fingerprint_dir(dataset_fingerprint)
    if not path or not path.exists():
        return None
    try:
        with path.open("r") as f:
            cached = json.load(f)
        if cached.get("discovery_version") != DISCOVERY_VERSION:
            return None
        return cached.get("metrics")
    except (json.JSONDecodeError, OSError) as error:
        logger.warning("Discovered-metric cache %s is corrupted (%s); rediscovering.", path.name, error)
        return None


def _save_cache(dataset_fingerprint: str | None, metrics: dict) -> None:
    path = _fingerprint_dir(dataset_fingerprint)
    if not path:
        return
    SCHEMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump({"discovery_version": DISCOVERY_VERSION, "metrics": metrics}, f, indent=2, default=str)
    tmp.replace(path)


def discover_metrics(
    data,
    profile: dict,
    dataset_fingerprint: str | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Returns {metric_key: MetricSpec}. `data`/`profile` are only needed for a
    cache miss; a cache hit skips touching `data` entirely.
    """
    if use_cache:
        cached = _load_cached(dataset_fingerprint)
        if cached is not None:
            logger.info("Loaded discovered metrics for fingerprint %s from cache.", dataset_fingerprint)
            return cached

    metrics = {}

    for field in profile.get("candidate_metrics", []):
        col_profile = profile["columns"][field]
        default_agg, allowed_aggs, agg_confidence = _infer_aggregation(field, col_profile)
        non_negative = _infer_non_negative(field, col_profile)
        additive = _infer_additive(default_agg, col_profile["semantic_type"])
        forecastable = col_profile.get("is_forecast_candidate", False)

        metric_key = field  # column name doubles as the key -- always resolvable exactly

        metrics[metric_key] = {
            "metric_key": metric_key,
            "field": field,
            "display_name": _display_name(field),
            "semantic_type": col_profile["semantic_type"],
            "default_aggregation": default_agg,
            "allowed_aggregations": allowed_aggs,
            "non_negative": non_negative,
            "zero_valid": True,
            "additive": additive,
            "forecastable": forecastable,
            "forecast_rejection_reasons": col_profile.get("forecast_rejection_reasons", []),
            "supported_granularities": [],  # filled in by forecastability.py
            "discovery_confidence": round(agg_confidence, 2),
            "aliases": _aliases(field),
            "needs_confirmation": agg_confidence < 0.6,
        }

    # Distinct-count metrics for identifier-like columns (Phase 3: "may be
    # used for distinct-count metrics when appropriate").
    for col, col_profile in profile.get("columns", {}).items():
        if not col_profile.get("is_identifier"):
            continue
        metric_key = f"{col}_distinct_count"
        metrics[metric_key] = {
            "metric_key": metric_key,
            "field": col,
            "display_name": f"Distinct {_display_name(col.removesuffix('_id'))} Count",
            "semantic_type": "count",
            "default_aggregation": "nunique",
            "allowed_aggregations": ["nunique"],
            "non_negative": True,
            "zero_valid": True,
            "additive": False,
            "forecastable": bool(profile.get("primary_date_column")),
            "forecast_rejection_reasons": [] if profile.get("primary_date_column") else ["no reliable date column"],
            "supported_granularities": [],
            "discovery_confidence": 0.6,
            "aliases": _identifier_aliases(col),
            "needs_confirmation": True,
        }

    _save_cache(dataset_fingerprint, metrics)
    return metrics