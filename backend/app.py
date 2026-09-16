from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from request_analyser import analyze_request
from router import execute_request
from data_sources import load_data
from dataset_runtime_config import build_runtime_config
from dynamic_router import run_dynamic_forecast
from conversation_orchestrator import analyze_conversation
app = FastAPI(
    title="Zeapl AI Dashboard API",
    version="2.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home() -> dict:
    return {
        "message": "Zeapl AI Dashboard API is running.",
        "analyze_endpoint": "/api/analyze",
        "dynamic_endpoints": ["/api/v2/profile", "/api/v2/forecast"],
    }
@app.get("/api/data")
def get_data() -> dict:
    data = load_data()
    return {
        "records": data,
        "record_count": len(data),
    }
@app.get("/api/analyze")
def analyze(question: str) -> dict:
    data = load_data()

    try:
        request = analyze_request(
            question,
            data=data,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return execute_request(
        data=data,
        request=request,
    )


# ---------------------------------------------------------------------------
# Dataset-agnostic endpoints (new).
#
# `dataset` is an OPTIONAL path to any CSV/JSON file this server can read,
# defaulting to fact_scan_activity.csv so nothing here breaks existing
# behavior if the caller omits it. This is the "how does a dataset get
# selected" answer from the Phase 1 audit -- smallest surface change,
# additive only. To point at a different dataset (sales/IoT/customer/etc.),
# pass e.g. ?dataset=/path/to/your_file.csv.
# ---------------------------------------------------------------------------

DEFAULT_DYNAMIC_DATASET = Path(__file__).parent / "campaign master dump.csv"

def _resolve_dataset_path(dataset: str | None) -> Path:
    return Path(dataset) if dataset else DEFAULT_DYNAMIC_DATASET


@app.get("/api/v2/profile")
def profile(dataset: str | None = None) -> dict:
    """Inspect what the system can automatically discover about a dataset --
    no forecasting, just the profile + discovered metrics/dimensions."""
    path = _resolve_dataset_path(dataset)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"dataset not found: {path}")
    data = load_data(path)
    config = build_runtime_config(data, dataset_id=path.stem, source_path=str(path))
    return config


@app.get("/api/v2/forecast")
def forecast(
    question: str,
    dataset: str | None = None,
    metric: str | None = None,
    granularity: str | None = None,
    horizon: int | None = None,
    use_llm_fallback: bool = False,
) -> dict:
    """Dataset-agnostic forecasting: works for any tabular CSV/JSON with a
    detectable date column and at least one forecastable numeric metric --
    no registries.py editing required for a new dataset."""
    path = _resolve_dataset_path(dataset)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"dataset not found: {path}")
    data = load_data(path)
    return run_dynamic_forecast(
        data, query=question, dataset_id=path.stem, source_path=str(path),
        metric_key_override=metric, granularity_override=granularity, horizon_override=horizon,
        enable_llm_fallback=use_llm_fallback,
    )


@app.get("/api/v2/analyze")
def analyze_v2(
    question: str,
    dataset: str | None = None,
    metric: str | None = None,
    granularity: str | None = None,
    horizon: int | None = None,
    use_llm_fallback: bool = False,
    conversation_id: str | None = None,
):
    path = _resolve_dataset_path(dataset)

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"dataset not found: {path}",
        )

    data = load_data(path)

    return analyze_conversation(
        data=data,
        question=question,
        conversation_id=conversation_id,
        dataset_id=path.stem,
        source_path=str(path),
        metric_override=metric,
        granularity_override=granularity,
        horizon_override=horizon,
        use_llm_fallback=use_llm_fallback,
    )