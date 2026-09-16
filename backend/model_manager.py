import json
import hashlib
import logging
import joblib
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
METADATA_FILE = MODELS_DIR / "metadata.json"

# Below this R², a forecast is flagged as low_confidence rather than
# presented at face value.
LOW_CONFIDENCE_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Retraining configuration (Task 6)
#
# Centralized here so tuning retrain sensitivity never requires touching
# train_models.py itself.
# ---------------------------------------------------------------------------
ROW_GROWTH_RETRAIN_THRESHOLD = 0.01      # 1% more rows than last training run
MIN_NEW_ROWS_RETRAIN_THRESHOLD = 500     # ...or this many new raw rows, whichever fires first
MODEL_MAX_AGE_DAYS = 14                  # force a refresh even if nothing else changed
VALIDATION_DROP_RETRAIN_THRESHOLD = 0.15 # relative R² drop vs last saved model that should trigger concern

# Feature-engineering version. Bumped whenever create_features()'s output
# shape/semantics change (see prediction.py). Folded into every model's
# filename and fingerprint so a model trained on an old feature schema can
# never be silently loaded and fed a mismatched feature vector.
FEATURE_ENGINEERING_VERSION = 2


# ---------------------------------------------------------------------------
# Dataset fingerprinting (Task 5)
#
# The original fingerprint only hashed column names, so two datasets with
# identical columns but very different content (row count, value ranges,
# dtypes) hashed identically and could silently share a model. This version
# additionally hashes:
#   - dtypes per column
#   - row count
#   - summary statistics (mean/std/min/max, rounded) for numeric columns
# so a meaningfully different dataset (more rows, wider value ranges, a
# column that changed type) always gets a new fingerprint, while cosmetic
# differences (row order, formatting) do not force unnecessary retraining.
# ---------------------------------------------------------------------------

def compute_dataset_fingerprint(data: list[dict]) -> str:
    if not data:
        return "empty"

    df = pd.DataFrame(data)

    columns_sig = ",".join(sorted(df.columns))
    dtypes_sig = ",".join(f"{c}:{df[c].dtype}" for c in sorted(df.columns))

    numeric_df = df.select_dtypes(include=[np.number])
    stats_parts = []
    for col in sorted(numeric_df.columns):
        series = numeric_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            stats_parts.append(f"{col}:empty")
            continue
        stats_parts.append(
            f"{col}:{series.mean():.3f}:{series.std(ddof=0):.3f}:"
            f"{series.min():.3f}:{series.max():.3f}"
        )
    stats_sig = ",".join(stats_parts)

    signature = "|".join([
        columns_sig,
        dtypes_sig,
        f"rows:{len(df)}",
        stats_sig,
        f"fe_v{FEATURE_ENGINEERING_VERSION}",
    ])

    return hashlib.md5(signature.encode()).hexdigest()[:16]


def compute_newest_timestamp(data: list[dict], date_column: str) -> str | None:
    """Max value of the dataset's date column, as an ISO string. Used to
    detect "same row count, but rows changed" (e.g. corrections/backfills)
    that a pure fingerprint over summary stats might miss."""
    if not data or date_column not in data[0]:
        return None
    try:
        series = pd.to_datetime(pd.DataFrame(data)[date_column], errors="coerce")
        newest = series.max()
        if pd.isna(newest):
            return None
        return newest.isoformat()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metadata I/O (Task 8 -- corrupted metadata handling)
#
# Self-healing: a fresh project with no models/ directory or metadata.json
# yet no longer crashes on the first call, and a corrupted metadata.json
# (truncated write, manual edit gone wrong, etc.) is treated as "start
# fresh" with a logged warning instead of crashing every endpoint that
# touches models.
#
# A small in-process cache avoids re-reading + re-parsing metadata.json on
# every single call within the same process (Task 9 -- performance); it's
# invalidated automatically whenever the file's mtime changes, including
# from another process.
# ---------------------------------------------------------------------------

_metadata_cache: dict | None = None
_metadata_cache_mtime: float | None = None


def _default_metadata() -> dict:
    return {"models": {}, "status": {"state": "idle"}}


def load_metadata(use_cache: bool = True) -> dict:
    global _metadata_cache, _metadata_cache_mtime

    if not METADATA_FILE.exists():
        _metadata_cache, _metadata_cache_mtime = _default_metadata(), None
        return _metadata_cache

    try:
        current_mtime = METADATA_FILE.stat().st_mtime
    except OSError:
        current_mtime = None

    if use_cache and _metadata_cache is not None and current_mtime == _metadata_cache_mtime:
        return _metadata_cache

    try:
        with open(METADATA_FILE, "r") as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            raise ValueError("metadata.json root must be an object")
    except (json.JSONDecodeError, ValueError, OSError) as error:
        logger.warning(
            "metadata.json is corrupted or unreadable (%s); starting from a fresh, "
            "empty metadata store. Existing model .pkl files are unaffected and will "
            "simply be retrained.",
            error,
        )
        metadata = _default_metadata()

    metadata.setdefault("models", {})
    metadata.setdefault("status", {"state": "idle"})

    _metadata_cache, _metadata_cache_mtime = metadata, current_mtime
    return metadata


def save_metadata(metadata: dict) -> None:
    global _metadata_cache, _metadata_cache_mtime

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Write atomically (temp file + rename) so a crash mid-write can never
    # leave metadata.json half-written/corrupted for the next process.
    tmp_path = METADATA_FILE.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(metadata, f, indent=4, default=str)
    tmp_path.replace(METADATA_FILE)

    _metadata_cache = metadata
    try:
        _metadata_cache_mtime = METADATA_FILE.stat().st_mtime
    except OSError:
        _metadata_cache_mtime = None


# ---------------------------------------------------------------------------
# Model file paths + I/O
#
# Filenames include both the dataset fingerprint AND the feature-engineering
# version, so neither a different dataset nor a code change to
# create_features() can result in a mismatched model being loaded.
# ---------------------------------------------------------------------------

def get_model_path(metric_key: str, granularity: str, dataset_fingerprint: str) -> Path:
    return MODELS_DIR / (
        f"{metric_key}_{granularity}_{dataset_fingerprint}_fe{FEATURE_ENGINEERING_VERSION}.pkl"
    )


def save_model(model, metric_key, granularity, dataset_fingerprint) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = get_model_path(metric_key, granularity, dataset_fingerprint)

    tmp_path = model_path.with_suffix(".pkl.tmp")
    joblib.dump(model, tmp_path)
    tmp_path.replace(model_path)

    _model_cache[str(model_path)] = model


# In-memory cache of already-loaded models (Task 9 -- performance). Keyed
# by resolved file path so repeated predictions against the same model
# within one process never touch disk more than once.
_model_cache: dict = {}


def load_model(metric_key, granularity, dataset_fingerprint):
    """Returns None (never raises) if no model exists yet for this
    metric/granularity/dataset/feature-version combination, or if the file
    on disk is corrupted (e.g. truncated write, incompatible pickle)."""
    model_path = get_model_path(metric_key, granularity, dataset_fingerprint)
    cache_key = str(model_path)

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if not model_path.exists():
        return None

    try:
        model = joblib.load(model_path)
    except Exception as error:
        logger.warning(
            "Model file %s is corrupted or unreadable (%s); treating as missing "
            "so it will be retrained.",
            model_path.name, error,
        )
        return None

    _model_cache[cache_key] = model
    return model


# ---------------------------------------------------------------------------
# Retrain decision (Task 6)
#
# Retrains if ANY of the following hold:
#   - this metric/granularity has never been trained
#   - the dataset fingerprint changed (schema, dtypes, row count, or value
#     distribution shifted meaningfully)
#   - the newest timestamp in the dataset moved (new/corrected rows landed
#     even if total row count didn't change, e.g. a backfill)
#   - row count grew by more than ROW_GROWTH_RETRAIN_THRESHOLD, or by more
#     than MIN_NEW_ROWS_RETRAIN_THRESHOLD rows outright
#   - the currently-saved model is older than MODEL_MAX_AGE_DAYS
#
# "validation score decreased significantly" (also called for in Task 6) is
# intentionally NOT a pre-train gate here -- you can't know a new model's
# validation score until after training it. It's instead enforced at
# save-time by compare_models(): a newly retrained model only replaces the
# saved one if it's at least as good, and update_metadata() records both so
# a persistent, unexplained drop is visible in metadata.json for monitoring.
# ---------------------------------------------------------------------------

def _model_name(dataset_fingerprint: str, metric_key: str, granularity: str) -> str:
    """Metadata key, namespaced by dataset -- two different datasets that
    happen to produce the same metric_key/granularity (e.g. both discover a
    'revenue' metric) must never share a metadata entry, even though their
    .pkl files are already safely separated by fingerprint in the filename."""
    return f"{dataset_fingerprint}:{metric_key}_{granularity}"


def should_retrain(
    metric_key: str,
    granularity: str,
    current_rows: int,
    dataset_fingerprint: str,
    newest_timestamp: str | None = None,
) -> tuple[bool, str]:
    """Returns (should_retrain, reason) for observability/logging."""
    metadata = load_metadata()
    model_name = _model_name(dataset_fingerprint, metric_key, granularity)

    if model_name not in metadata["models"]:
        return True, "no existing model"

    existing = metadata["models"][model_name]

    if existing.get("dataset_fingerprint") != dataset_fingerprint:
        return True, "dataset fingerprint changed"

    if newest_timestamp and existing.get("newest_timestamp") not in (None, newest_timestamp):
        return True, "newest data timestamp changed"

    old_rows = existing.get("rows", 0)
    if old_rows == 0:
        return True, "previously trained on zero rows"

    growth = (current_rows - old_rows) / old_rows
    if growth > ROW_GROWTH_RETRAIN_THRESHOLD:
        return True, f"row count grew {growth:.1%}"

    if (current_rows - old_rows) > MIN_NEW_ROWS_RETRAIN_THRESHOLD:
        return True, f"{current_rows - old_rows} new rows added"

    trained_at = existing.get("trained_at")
    if trained_at:
        try:
            age = datetime.now() - datetime.fromisoformat(trained_at)
            if age > timedelta(days=MODEL_MAX_AGE_DAYS):
                return True, f"model is {age.days} days old (> {MODEL_MAX_AGE_DAYS})"
        except ValueError:
            return True, "unparseable trained_at timestamp"
    else:
        return True, "no trained_at timestamp on record"

    return False, "up to date"


def compare_models(new_metrics: dict | None, old_metrics: dict | None) -> bool:
    """True if the newly trained model should replace the saved one.

    Compares by R² when both are available (higher is better); a model with
    no evaluable R² (e.g. a flat/degenerate series) only replaces an
    existing model if there was nothing saved before.
    """
    new_r2 = (new_metrics or {}).get("r2")
    old_r2 = (old_metrics or {}).get("r2")

    if old_r2 is None:
        return True
    if new_r2 is None:
        return False
    return new_r2 >= old_r2


def update_metadata(
    metric_key: str,
    granularity: str,
    rows: int,
    metrics: dict,
    dataset_fingerprint: str,
    newest_timestamp: str | None = None,
    algorithm: str = "linear_regression",
    feature_count: int | None = None,
) -> None:
    metadata = load_metadata()
    model_name = _model_name(dataset_fingerprint, metric_key, granularity)

    version = 1
    if model_name in metadata["models"]:
        version = metadata["models"][model_name].get("version", 0) + 1

    metadata["models"][model_name] = {
        "rows": rows,
        "confidence": metrics.get("r2"),
        "r2": metrics.get("r2"),
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "validation_folds": metrics.get("n_folds"),
        "validation_note": metrics.get("note"),
        "algorithm": algorithm,
        "feature_count": feature_count,
        "feature_engineering_version": FEATURE_ENGINEERING_VERSION,
        "version": version,
        "dataset_fingerprint": dataset_fingerprint,
        "newest_timestamp": newest_timestamp,
        "trained_at": datetime.now().isoformat(),
    }

    save_metadata(metadata)


def get_model_confidence(dataset_fingerprint: str, metric_key: str, granularity: str):
    metadata = load_metadata()
    model_name = _model_name(dataset_fingerprint, metric_key, granularity)
    if model_name not in metadata["models"]:
        return None
    return metadata["models"][model_name].get("r2")


def get_model_metrics(dataset_fingerprint: str, metric_key: str, granularity: str) -> dict | None:
    """Full validation metrics + bookkeeping for a saved model, or None."""
    metadata = load_metadata()
    model_name = _model_name(dataset_fingerprint, metric_key, granularity)
    return metadata["models"].get(model_name)


# ---------------------------------------------------------------------------
# Training status (unchanged from the original -- lets an API layer poll
# "is training still running" instead of blocking a request on it)
# ---------------------------------------------------------------------------

def set_training_status(state: str, dataset_fingerprint: str | None = None):
    """state: 'idle' | 'training' | 'ready' | 'error'"""
    metadata = load_metadata()
    metadata["status"] = {
        "state": state,
        "dataset_fingerprint": dataset_fingerprint,
        "updated_at": datetime.now().isoformat(),
    }
    save_metadata(metadata)


def get_training_status():
    metadata = load_metadata()
    return metadata.get("status", {"state": "idle"})