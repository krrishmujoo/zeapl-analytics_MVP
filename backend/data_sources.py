import json
import threading
from pathlib import Path

import pandas as pd
from fastapi import HTTPException


DATA_FILE = Path(__file__).parent / "fact_scan_activity.csv"


def load_json(file_path: Path) -> list[dict]:
    try:
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail="JSON data file is invalid.",
        ) from error

    if not isinstance(data, list):
        raise HTTPException(
            status_code=500,
            detail="JSON data must contain a list of records.",
        )

    return data


def convert_csv_value(value: str):
    """Kept for backward compatibility with anything importing it directly
    (e.g. tests). load_csv() no longer calls this per-cell -- see its
    docstring for why."""
    cleaned_value = value.strip()

    if cleaned_value == "":
        return None

    try:
        return int(cleaned_value)

    except ValueError:
        pass

    try:
        return float(cleaned_value)

    except ValueError:
        return cleaned_value


def load_csv(file_path: Path) -> list[dict]:
    """
    Same output shape/semantics as the original row-by-row implementation
    (empty cell -> None, whole numbers -> int, other numerics -> float,
    everything else -> str) but parses with pandas' C engine instead of
    csv.DictReader + a per-cell try/except int()/float(). On a 300k-row
    file this cut load time from ~11s to well under 1s -- the per-cell
    Python-level exception handling was the bottleneck, not disk I/O.
    """
    try:
        df = pd.read_csv(
            file_path,
            encoding="utf-8-sig",
            keep_default_na=True,
            na_values=[""],
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise HTTPException(
            status_code=500,
            detail="CSV data file is invalid.",
        ) from error

    # Whole-number float columns (e.g. "5.0" parsed as float64 because the
    # column also has blanks/NaNs) get demoted to nullable Int64, matching
    # convert_csv_value()'s original "prefer int over float" behavior.
    for col in df.columns:
        if df[col].dtype == "float64":
            non_null = df[col].dropna()
            if not non_null.empty and (non_null % 1 == 0).all():
                df[col] = df[col].astype("Int64")

    columns = df.columns.tolist()
    # .tolist() on each column converts numpy scalars (np.int64/np.float64)
    # to native Python int/float in one vectorized C-level pass, instead of
    # a Python-level isinstance/convert check per cell. zip(*...) then
    # builds rows without repeated per-column dict indexing.
    column_lists = [df[col].tolist() for col in columns]
    nullable_columns = {col for col in columns if df[col].isna().any()}

    records = [dict(zip(columns, row)) for row in zip(*column_lists)]

    if nullable_columns:
        for record in records:
            for col in nullable_columns:
                value = record[col]
                # pandas leaves gaps as NaN (float) or pd.NA (nullable
                # Int64); normalize both to None like the original did
                # for blank cells.
                if value is pd.NA or (isinstance(value, float) and value != value):
                    record[col] = None

    return records



def load_excel(file_path: Path) -> list[dict]:
    try:
        import pandas as pd

        dataframe = pd.read_excel(file_path)

        dataframe = dataframe.where(
            dataframe.notna(),
            None,
        )

        return dataframe.to_dict(orient="records")

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read Excel file: {error}",
        )


# ---------------------------------------------------------------------------
# In-process data cache
#
# app.py/router.py call load_data() once per incoming request with no
# caching of their own, which meant every single question asked of the
# dashboard re-parsed the whole CSV/JSON file from disk -- ~11s per request
# on a 300k-row file, dwarfing everything else in the pipeline (prediction
# itself runs in well under a second once data is in memory).
#
# Cached by (resolved path, mtime, size), so an updated data file is picked
# up automatically on its next request without needing a server restart or
# manual cache-clear, while unchanged files are served from memory.
# Thread-safe for FastAPI's threaded request handling.
# ---------------------------------------------------------------------------

_data_cache: dict[str, tuple[float, int, list[dict]]] = {}
_cache_lock = threading.Lock()


def clear_data_cache() -> None:
    """Manual escape hatch -- e.g. call this after uploading a replacement
    data file if you want the next request to reload immediately rather
    than waiting on the mtime/size check (which will already catch a
    normal file overwrite on its own)."""
    with _cache_lock:
        _data_cache.clear()


def load_data(
    file_path: Path | None = None,
) -> list[dict]:
    selected_file = Path(file_path) if file_path else DATA_FILE

    if not selected_file.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Data file not found: {selected_file.name}",
        )

    stat = selected_file.stat()
    cache_key = str(selected_file.resolve())

    with _cache_lock:
        cached = _data_cache.get(cache_key)
        if (
            cached is not None
            and cached[0] == stat.st_mtime
            and cached[1] == stat.st_size
        ):
            return cached[2]

    file_extension = selected_file.suffix.lower()

    if file_extension == ".json":
        records = load_json(selected_file)

    elif file_extension == ".csv":
        records = load_csv(selected_file)

    elif file_extension in {".xlsx", ".xls"}:
        records = load_excel(selected_file)

    else:
        raise HTTPException(
            status_code=500,
            detail=f"Unsupported data format: {file_extension}",
        )

    with _cache_lock:
        _data_cache[cache_key] = (
            stat.st_mtime,
            stat.st_size,
            records,
        )

    return records