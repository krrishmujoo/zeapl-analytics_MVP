import re
from pathlib import Path

from aggregation_dispatcher import dispatch_aggregation
from aggregation import rollup_scan_activity_to_monthly
from insight import get_basic_insight
from chart import decide_chart,prepare_chart_data
from LLM import analyze_data
from response_builder import build_response
from prediction import run_prediction
from data_sources import load_data
from registries import (
    PredictionMetrics,
    PredictionGranularity,
    METRIC_CATEGORY_TO_PREDICTION_METRIC,
    DatasetConfig,
)


DEFAULT_PREDICTION_HORIZON = {"day": 7, "week": 4, "month": 3}

# prediction.py forecasts off raw scan-level rows (scan_date, invoice_value,
# etc.) -- a different shape from data.json's monthly rollups that
# aggregation/chart/insights use. Whichever data source app.py loaded for
# the rest of the request, the prediction step always loads this file
# directly so "forecast ..." questions work regardless of what /api/data
# is currently pointed at. Cached by data_sources.load_data(), so this is
# effectively free after the first call.
RAW_PREDICTION_DATA_PATH = Path(__file__).parent / "fact_scan_activity.csv"


def resolve_prediction_metric(request: dict) -> tuple[str, str, str]:
    """Turns a request into (metric_key, dataframe_field, agg_kind) using the
    keyword map in registries.PredictionMetrics. Defaults to revenue."""
    query = (request.get("query") or "").lower()

    for key, spec in PredictionMetrics.items():
        if any(keyword in query for keyword in spec["keywords"]):
            return key, spec["field"], spec["agg"]

    for category in request.get("metrics", []):
        mapped_key = METRIC_CATEGORY_TO_PREDICTION_METRIC.get(category)
        if mapped_key:
            spec = PredictionMetrics[mapped_key]
            return mapped_key, spec["field"], spec["agg"]

    default = PredictionMetrics["revenue"]
    return "revenue", default["field"], default["agg"]


def resolve_prediction_granularity(request: dict) -> str:
    """Turns a request into "day"/"week"/"month" using registries.PredictionGranularity."""
    query = (request.get("query") or "").lower()
    dimension = request.get("dimension")

    if dimension == "week":
        return "week"
    if dimension in ("month", "quarter", "year"):
        return "month"

    for granularity, keywords in PredictionGranularity.items():
        if any(keyword in query for keyword in keywords):
            return granularity

    return "day"


def resolve_prediction_horizon(request: dict, granularity: str) -> int:
    """Pulls a horizon like "next 30 days" out of the query, else a default per granularity."""
    query = (request.get("query") or "").lower()
    match = re.search(r"next\s+(\d+)\s*(day|week|month)", query)
    if match and match.group(2) == granularity.rstrip("s"):
        return int(match.group(1))
    return DEFAULT_PREDICTION_HORIZON[granularity]


AGGREGATION_EXECUTION_PRIORITY = [
    "growth",
    "top_users",
    "bottom_users",
    "unique_users",
    "average",
    "total",
     "distinct_count",
    "maximum",
    "minimum",
]


def find_aggregation_operation(
    operations: list[str],
) -> str | None:

    for operation in AGGREGATION_EXECUTION_PRIORITY:
        if operation in operations:
            return operation

    return None

def apply_filters(
    data: list[dict],
    filters: dict,
) -> list[dict]:
    if not filters or not data:
        return data

    months = filters.get("month", [])

    if not months:
        return data

    normalized_months = {
        month.lower()
        for month in months
    }

    if "month" in data[0]:
        return [
            record
            for record in data
            if str(
                record.get("month", "")
            ).lower() in normalized_months
        ]

    if "scan_date" in data[0]:
        import pandas as pd
        return [
            record
            for record in data
            if pd.to_datetime(record["scan_date"]).strftime("%B").lower() in normalized_months
        ]

    return data

def build_execution_plan(request: dict) -> dict:
    execution_plan = []

    aggregation_operation = request.get("primary_operation")
    if aggregation_operation not in AGGREGATION_EXECUTION_PRIORITY:
        aggregation_operation = None

    if aggregation_operation:
     execution_plan.append({
        "module": "aggregation",
        "metrics": request.get("metrics", []),
        "operation": aggregation_operation,
        "field": request.get("field"),
    })

    if request.get("requires_chart"):
        execution_plan.append({
            "module": "chart",
        })

    if request.get("requires_insights"):
        execution_plan.append({
            "module": "insights",
        })

    if request.get("requires_prediction"):
        execution_plan.append({
            "module": "prediction",
        })

    if request.get("requires_llm"):
        execution_plan.append({
            "module": "llm",
            "query": request["query"],
        })

    return {
        "execution_plan": execution_plan,
    }


def execute_request(
    data: list[dict],
    request: dict,
) -> dict:
    plan = build_execution_plan(request)

    filtered_data = apply_filters(
        data,
        request.get("filters", {}),
    )

    # aggregation/chart/insights/llm all expect one-record-per-month data
    # (data.json's shape). prediction works directly off raw scan rows, so
    # it uses filtered_data as-is below, never monthly_data. This is a
    # no-op if filtered_data is already monthly-shaped.
    monthly_data = rollup_scan_activity_to_monthly(filtered_data)

    results = {}
    errors = []
    

    
   

    for step in plan["execution_plan"]:
        module = step["module"]

        if module == "aggregation":
            aggregation_request = {
                "metrics": step["metrics"],
                "primary_operation": step["operation"],
                "top_n": request.get("top_n", 5),
                "field": step.get("field"),
                "location": request.get("location"),
            }

            try:
                # User-based operations need raw scan data
                if step["operation"] in [
                    "top_users",
                    "bottom_users",
                    "unique_users",
                    "distinct_count",
                ]:
                    aggregation_data = filtered_data
                else:
                    aggregation_data = monthly_data

                aggregation_result = dispatch_aggregation(
                    aggregation_data,
                    aggregation_request,
                )

                results["aggregation"] = aggregation_result

            except Exception as error:
                errors.append(
                    f"Aggregation failed: {error}"
                )

        elif module == "chart":
            try:
                chart_result = decide_chart(request)

                chart_data = prepare_chart_data(
                    monthly_data,
                    request,
                )

                results["chart"] = chart_result
                results["chart_data"] = chart_data

            except Exception as error:
                 import traceback
                 traceback.print_exc()   # Prints the full traceback in the terminal

                 errors.append(
                    f"Chart generation failed: {error}"
                )

        elif module == "insights":
            try:
                insights_result = get_basic_insight(
                request["query"],
                monthly_data,
            )
                results["insights"] = insights_result

            except Exception as error:
                errors.append(
                    f"Insight generation failed: {error}"
                )

        elif module == "prediction":
            try:
                metric_key, field, agg = resolve_prediction_metric(request)
                granularity = resolve_prediction_granularity(request)
                horizon = resolve_prediction_horizon(request, granularity)

                raw_data = load_data(RAW_PREDICTION_DATA_PATH)
                raw_filtered = apply_filters(raw_data, request.get("filters", {}))

                prediction_result = run_prediction(
                    raw_filtered,
                    metric_key=metric_key,
                    field=field,
                    agg=agg,
                    granularity=granularity,
                    horizon=horizon,
                    date_column=DatasetConfig["date_column"],
                    valid_column=DatasetConfig.get("valid_column"),
                )
                results["prediction"] = prediction_result

            except Exception as error:
                errors.append(
                    f"Prediction failed: {error}"
                )

        elif module == "llm":
            try:
                llm_result = analyze_data(
                monthly_data,
                step["query"],
                reason="llm",
            )
                results["llm"] = llm_result

            except Exception as error:
                errors.append(
                    f"LLM analysis failed: {error}"
                )

    return build_response(
        request=request,
        result=results.get("aggregation"),
        insights=results.get("insights"),
        chart_spec=results.get("chart", {}).get("chart_spec"),
        chart_data=results.get("chart_data"),
        prediction=results.get("prediction"),
        explanation=results.get("llm"),
        errors=errors,
    )