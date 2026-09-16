"""
dataset_runtime_config.py -- Phase 4

Replaces the global, fixed registries.DatasetConfig with a runtime
configuration generated per dataset (keyed by dataset fingerprint), built
from dataset_profiler + metric_discovery + forecastability. Persisted as
JSON with atomic writes; invalidated automatically when the fingerprint,
profiler version, or discovery version changes.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from dataset_profiler import PROFILER_VERSION, profile_dataset
from metric_discovery import DISCOVERY_VERSION, discover_metrics
from forecastability import assess_forecastability
from model_manager import compute_dataset_fingerprint, compute_newest_timestamp

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "configs"
CONFIG_VERSION = 2


def _config_path(dataset_fingerprint: str) -> Path:
    return CONFIG_DIR / f"{dataset_fingerprint}.json"


def _load_cached_config(dataset_fingerprint: str) -> dict | None:
    path = _config_path(dataset_fingerprint)
    if not path.exists():
        return None
    try:
        with path.open("r") as f:
            cached = json.load(f)
    except (json.JSONDecodeError, OSError) as error:
        logger.warning("Runtime config cache %s is corrupted (%s); rebuilding.", path.name, error)
        return None

    if (
        cached.get("config_version") != CONFIG_VERSION
        or cached.get("profiler_version") != PROFILER_VERSION
        or cached.get("discovery_version") != DISCOVERY_VERSION
    ):
        logger.info("Runtime config cache %s is stale (version mismatch); rebuilding.", path.name)
        return None

    return cached


def _save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = _config_path(config["dataset_fingerprint"])
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(config, f, indent=2, default=str)
    tmp.replace(path)


def build_runtime_config(
    data,
    dataset_id: str | None = None,
    source_path: str | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Builds (or loads a cached) DatasetRuntimeConfig for `data`. Two datasets
    with different schemas/content never collide -- everything is keyed and
    persisted by dataset_fingerprint, which already factors in columns,
    dtypes, row count, and value distributions (model_manager.compute_dataset_fingerprint).
    """
    fingerprint = compute_dataset_fingerprint(data)

    if use_cache:
        cached = _load_cached_config(fingerprint)
        if cached is not None:
            logger.info("Loaded runtime config for dataset %s (fingerprint %s) from cache.",
                        dataset_id or source_path, fingerprint)
            return cached

    profile = profile_dataset(data, dataset_id=dataset_id)
    metrics = discover_metrics(data, profile, dataset_fingerprint=fingerprint, use_cache=use_cache)

    date_column = profile.get("primary_date_column")
    newest_timestamp = compute_newest_timestamp(data, date_column) if date_column else None

    # Forecastability per metric, folded into each metric's spec so callers
    # don't need a second lookup.
    warnings = list(profile.get("warnings", []))
    if date_column:
        for metric_key, spec in metrics.items():
            if not spec.get("forecastable"):
                continue
            fc = assess_forecastability(data, spec, date_column)
            spec["forecastable"] = fc["forecastable"]
            spec["supported_granularities"] = fc["supported_granularities"]
            spec["recommended_granularity"] = fc["recommended_granularity"]
            spec["max_horizon"] = fc["max_horizon"]
            spec["forecastability_score"] = fc["score"]
            if not fc["forecastable"]:
                spec["forecast_rejection_reasons"] = fc["reasons"]
            warnings.extend(f"{metric_key}: {w}" for w in fc.get("warnings", []))
    else:
        for spec in metrics.values():
            spec["forecastable"] = False
            spec["forecast_rejection_reasons"] = ["no reliable date column in dataset"]
        warnings.append("no primary date column detected -- no metric is forecastable")

    time_range = None
    if date_column:
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if len(dates):
            time_range = {
                "start": dates.min().isoformat(),
                "end": dates.max().isoformat(),
                "days": int((dates.max() - dates.min()).total_seconds() / 86400),
            }

    config = {
        "config_version": CONFIG_VERSION,
        "profiler_version": PROFILER_VERSION,
        "discovery_version": DISCOVERY_VERSION,
        "dataset_id": dataset_id,
        "dataset_fingerprint": fingerprint,
        "source_path": source_path,
        "date_column": date_column,
        "valid_column": None,  # no generic way to detect a validity flag; left for explicit override
        "metrics": metrics,
        "dimensions": profile.get("candidate_dimensions", []),
        "time_range": time_range,
        "newest_timestamp": newest_timestamp,
        "row_count": profile.get("row_count", 0),
        "column_profiles": profile.get("columns", {}),
        "warnings": warnings,
        "built_at": datetime.now().isoformat(),
    }

    _save_config(config)
    return config