from typing import Dict, List, Optional
from registries import METRIC_TO_FIELD

# -------------------------------------------------------
# Metrics supported by the dashboard
# -------------------------------------------------------

SUPPORTED_METRICS = {
    "revenue",
    "profit",
    "expenses",
    "customers",
    "sales",
    "churn",
}


# -------------------------------------------------------
# Time dimensions
# -------------------------------------------------------

TIME_DIMENSIONS = {
    "month",
    "months",
    "quarter",
    "quarters",
    "year",
    "years",
    "week",
    "weeks",
    "day",
    "days",
    "date",
}


# -------------------------------------------------------
# Categorical dimensions
# -------------------------------------------------------

CATEGORY_DIMENSIONS = {
    "region",
    "product",
    "department",
    "segment",
    "city",
    "country",
    "category",
}



# -------------------------------------------------------
# Explicit chart keywords
# -------------------------------------------------------

CHART_KEYWORDS = {
    "chart",
    "graph",
    "plot",
    "visualize",
    "visualisation",
    "visualization",
}


# -------------------------------------------------------
# Query Helpers
# -------------------------------------------------------



def detect_dimension_type(dimension: Optional[str]) -> Optional[str]:

    if dimension is None:
        return None

    if dimension in TIME_DIMENSIONS:
        return "time"

    if dimension in CATEGORY_DIMENSIONS:
        return "category"

    return None



# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def metric_fields(metrics: List[str], metric_field_map: Optional[dict] = None) -> List[str]:

    if metric_field_map is not None:
        return [metric_field_map[metric] for metric in metrics if metric in metric_field_map]

    fields = []

    for metric in metrics:
        if metric in METRIC_TO_FIELD:
            fields.append(METRIC_TO_FIELD[metric])

    return fields


def is_time_dimension(dimension: Optional[str]) -> bool:

    return detect_dimension_type(dimension) == "time"


def is_category_dimension(dimension: Optional[str]) -> bool:

    return detect_dimension_type(dimension) == "category"


def is_single_metric(metrics: List[str]) -> bool:

    return len(metrics) == 1


def is_multi_metric(metrics: List[str]) -> bool:

    return len(metrics) > 1


def supports_rule_engine(metrics: List[str], metric_field_map: Optional[dict] = None) -> bool:

    if not metrics:
        return False

    if metric_field_map is not None:
        return all(metric in metric_field_map for metric in metrics)

    for metric in metrics:
        if metric not in SUPPORTED_METRICS:
            return False

    return True


# -------------------------------------------------------
# Chart Selection Engine
# -------------------------------------------------------

def choose_chart_type(
    metrics: List[str],
    operations: List[str],
    dimension: Optional[str],
    query: str,
) -> str:
    """
    Rule-based chart selector.

    Possible return values:
        bar
        line
        grouped_bar
        multi_line
        pie
        scatter
        none
    """

    dimension_type = detect_dimension_type(dimension)
    lower_query = query.lower()

    # -----------------------------
    # Correlation
    # -----------------------------
    if "correlation" in operations and len(metrics) >= 2:
        return "scatter"

    # -----------------------------
    # Explicit comparison
    # -----------------------------
    if "compare" in operations:

        if len(metrics) >= 2:

            if dimension_type == "time":
                return "multi_line"

            return "grouped_bar"

    # -----------------------------
    # Trend / Growth
    # -----------------------------
    if (
        "trend" in operations
        or "growth" in operations
        or dimension_type == "time"
    ):
        return "line"

    # -----------------------------
    # Distribution
    # -----------------------------
    if (
        "distribution" in lower_query
        or "share" in lower_query
        or "breakdown" in lower_query
    ):

        if dimension_type == "category":
            return "pie"

    # -----------------------------
    # Multiple metrics
    # -----------------------------
    if len(metrics) > 1:

        if dimension_type == "time":
            return "multi_line"

        return "grouped_bar"

    # -----------------------------
    # Single metric
    # -----------------------------
    return "bar"


# -------------------------------------------------------
# Axis Builder
# -------------------------------------------------------

def choose_axes(
    metrics: List[str],
    dimension: Optional[str],
    metric_field_map: Optional[dict] = None,
):

    y_fields = metric_fields(metrics, metric_field_map)

    if dimension:
        x_axis = dimension
    else:
        x_axis = "index"

    return x_axis, y_fields


# -------------------------------------------------------
# Title Builder
# -------------------------------------------------------

def build_title(
    metrics: List[str],
    chart_type: str,
    dimension: Optional[str],
    metric_display_map: Optional[dict] = None,
):

    if metric_display_map is not None:
        metric_names = " vs ".join(
            metric_display_map.get(metric, metric.capitalize())
            for metric in metrics
        )
    else:
        metric_names = " vs ".join(
            metric.capitalize()
            for metric in metrics
        )

    if dimension:
        return f"{metric_names} by {dimension.capitalize()}"

    return metric_names


# -------------------------------------------------------
# Legend
# -------------------------------------------------------

def needs_legend(metrics: List[str]) -> bool:

    return len(metrics) > 1


# -------------------------------------------------------
# Pie Chart Validation
# -------------------------------------------------------

def should_use_pie(
    dimension: Optional[str],
    query: str,
) -> bool:

    if dimension is None:
        return False

    if detect_dimension_type(dimension) != "category":
        return False

    lower = query.lower()

    return (
        "distribution" in lower
        or "share" in lower
        or "breakdown" in lower
    )


# -------------------------------------------------------
# Scatter Validation
# -------------------------------------------------------

def should_use_scatter(
    metrics: List[str],
    operations: List[str],
):

    if "correlation" not in operations:
        return False

    return len(metrics) >= 2


# -------------------------------------------------------
# Build Chart Configuration
# -------------------------------------------------------

def build_chart_config(
    chart_type: str,
    metrics: List[str],
    dimension: Optional[str],
    metric_field_map: Optional[dict] = None,
    metric_display_map: Optional[dict] = None,
):

    x_axis, y_axis = choose_axes(metrics, dimension, metric_field_map)

    config = {
        "chart_type": chart_type,
        "title": build_title(
            metrics,
            chart_type,
            dimension,
            metric_display_map,
        ),
        "x_key": x_axis,
        "y_key": y_axis[0],
        "y_keys": y_axis,
        "x_label": x_axis.replace("_", " ").title(),
        "legend": needs_legend(metrics),
    }

    if chart_type in (
        "grouped_bar",
        "multi_line",
    ):
        config["comparison"] = True

    if chart_type == "pie":
        config["legend"] = True

    if chart_type == "scatter":
        config["x_key"] = y_axis[0]
        config["y_key"] = y_axis[1]

    return config

# -------------------------------------------------------
# Public API
# -------------------------------------------------------


def prepare_chart_data(
    data: List[Dict],
    request: Dict,
    metric_field_map: Optional[dict] = None,
) -> List[Dict]:
    metrics = request.get("metrics", [])
    dimension = request.get("dimension")

    metric_pairs = []

    if metric_field_map is not None:
        for metric in metrics:
            if metric in metric_field_map:
                metric_pairs.append((metric, metric_field_map[metric]))
    else:
        for metric in metrics:
            if metric in METRIC_TO_FIELD:
                metric_pairs.append(
                    (metric, METRIC_TO_FIELD[metric])
                )

    chart_data = []

    for index, record in enumerate(data):
        chart_record = {}

        if dimension:
            chart_record[dimension] = record.get(dimension)
        else:
            chart_record["index"] = index + 1

        for metric, field_name in metric_pairs:
            chart_record[field_name] = record.get(field_name)

        chart_data.append(chart_record)

    return chart_data

def decide_chart(request: Dict, metric_field_map: Optional[dict] = None, metric_display_map: Optional[dict] = None) -> Dict:
    """
    Main function called by app.py.

    Input:
    {
        "query": "...",
        "metrics": [...],
        "operations": [...],
        ...
    }

    Returns:
    {
        "use_llm": False,
        "chart_spec": {...}
    }

    OR

    {
        "use_llm": True
    }

    `metric_field_map`/`metric_display_map` are optional (metric_key -> field
    name / display name) dicts used by the V2 dynamic-metric callers to
    bypass the fixed SUPPORTED_METRICS/METRIC_TO_FIELD whitelist -- when
    omitted (every existing V1 call site), behavior is unchanged.
    """

    query = request.get("query", "")

    metrics = request.get("metrics", [])

    operations = request.get("operations", [])

    dimension = request.get("dimension")

    # ----------------------------
    # Basic validation
    # ----------------------------

    if not supports_rule_engine(metrics, metric_field_map):
        return {
        "chart_spec": None,
        "error": "Unsupported or missing metrics.",
    }




    # ----------------------------
    # Scatter override
    # ----------------------------

    if should_use_scatter(
        metrics,
        operations,
    ):
        chart = "scatter"

    # ----------------------------
    # Pie override
    # ----------------------------

    elif should_use_pie(
        dimension,
        query,
    ):
        chart = "pie"

    # ----------------------------
    # Normal rule engine
    # ----------------------------

    else:

        chart = choose_chart_type(
            metrics,
            operations,
            dimension,
            query,
        )

    chart_spec = build_chart_config(
        chart,
        metrics,
        dimension,
        metric_field_map,
        metric_display_map,
    )

    return {
    "chart_spec": chart_spec
}


# -------------------------------------------------------
# Optional helper
# -------------------------------------------------------

def explain_chart(chart_spec: Dict) -> str:
    """
    Short explanation for debugging.
    """

    chart = chart_spec["chart_type"]

    if chart == "line":
        return "Time-based trend detected."

    if chart == "bar":
        return "Single metric comparison."

    if chart == "grouped_bar":
        return "Multiple metrics comparison."

    if chart == "multi_line":
        return "Multiple metric trend."

    if chart == "pie":
        return "Part-to-whole distribution."

    if chart == "scatter":
        return "Correlation between metrics."

    return "Rule engine selected chart."


# -------------------------------------------------------
# Example
# -------------------------------------------------------

if __name__ == "__main__":

    request = {
        "query": "Compare revenue and profit by month",
        "metrics": [
            "revenue",
            "profit",
        ],
        "operations": [
            "compare",
        ],
    }

    result = decide_chart(request)

    from pprint import pprint

    print(result)