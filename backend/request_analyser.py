from __future__ import annotations
import re

from registries import Metrics, Operations
import pandas as pd

# Operations are checked in this order when multiple operations are detected.
OPERATION_PRIORITY = [
    "forecast",
    "correlation",
    "trend",
    "growth",
    "top_users",
    "bottom_users",
    "unique_users",
    "average",
    "total",
    "maximum",
    "minimum",
]


EXPLICIT_CHART_TERMS = [
    "chart",
    "graph",
    "plot",
    "visualize",
    "visualization",
]


TIME_SERIES_PHRASES = [
    "by month",
    "monthly",
    "over time",
    "by quarter",
    "quarterly",
    "by year",
    "yearly",
]


EXPLANATION_TERMS = [
    "why",
    "explain",
    "reason",
    "reasons",
    "cause",
    "caused",
    "interpret",
]


RECOMMENDATION_TERMS = [
    "suggest",
    "recommend",
    "recommendation",
    "strategy",
    "what should",
    "how can",
    "improve",
    "optimize",
]


INSIGHT_TERMS = [
    "insight",
    "insights",
    "summarize",
    "summary",
    "highlight",
    "key changes",
    "what stands out",
]


MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]


DIMENSIONS = {
    "month": [
        "by month",
        "per month",
        "monthly",
        "month over month",
    ],
    "quarter": [
        "by quarter",
        "per quarter",
        "quarterly",
        "quarter over quarter",
    ],
    "year": [
        "by year",
        "per year",
        "yearly",
        "year over year",
        "annually",
        "this year",
        "last year",
    ],
    "region": [
        "by region",
        "per region",
        "region-wise",
        "region wise",
    ],
}


def contains_alias(query: str, alias: str) -> bool:
    """
    Return True when an alias appears as a complete phrase.

    Example:
        alias='profit'
        matches 'show profit'
        does not accidentally match part of an unrelated word
    """
    pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
    return re.search(pattern, query) is not None


def extract_metrics(query: str) -> list[str]:
    """Detect all metrics mentioned in the user query."""
    normalized_query = query.lower()
    detected_metrics: list[str] = []

    for metric, aliases in Metrics.items():
        for alias in aliases:
            if contains_alias(normalized_query, alias.lower()):
                detected_metrics.append(metric)
                break

    return detected_metrics


def extract_operations(query: str) -> list[str]:
    """Detect all analytical operations mentioned in the query."""
    normalized_query = query.lower()
    detected_operations: list[str] = []

    for operation, aliases in Operations.items():
        for alias in aliases:
            if contains_alias(normalized_query, alias.lower()):
                detected_operations.append(operation)
                break

    return detected_operations


def extract_dimension(query: str) -> str | None:
    """
    Detect the requested grouping dimension.

    Examples:
        'revenue by month' -> 'month'
        'profit by region' -> 'region'
    """
    normalized_query = query.lower()

    for dimension, aliases in DIMENSIONS.items():
        for alias in aliases:
            if alias in normalized_query:
                return dimension

    return None




CITY_MAPPING = {
    "delhi": "new delhi",
    "bombay": "mumbai",
    "bangalore": "bengaluru",
}

def extract_location(
    query: str,
    data: list[dict],
) -> dict | None:
    """
    Dynamically detect a location-like filter from the current dataset.

    No hardcoded city/state/geo_zone columns are required.
    We inspect categorical/text columns in the current dataset and
    look for an exact value mentioned after phrases such as "in".
    """

    if not data:
        return None

    q = query.lower().strip()
    df = pd.DataFrame(data)

    if df.empty:
        return None

    # Look for phrases such as:
    # "in Rajasthan"
    # "in North"
    # "in Delhi"
    match = re.search(
        r"\bin\s+([a-zA-Z][a-zA-Z0-9 .&'_-]*?)(?=\s+by|\s+between|\s+where|\s+from|\s*$)",
        q,
    )

    if not match:
        return None

    value = match.group(1).strip()

    if not value:
        return None

    normalized_value = re.sub(r"\s+", " ", value).strip().lower()

    # Examine columns dynamically.
    # We intentionally skip obvious numeric/date columns.
    for column in df.columns:
        series = df[column]

        if pd.api.types.is_numeric_dtype(series):
            continue

        # Convert values to strings for semantic comparison.
        values = (
            series.dropna()
            .astype(str)
            .map(lambda x: re.sub(r"\s+", " ", x.strip()).lower())
        )

        if values.empty:
            continue

        # Exact value match.
        matches = values == normalized_value

        if matches.any():
            return {
                "field": column,
                "value": value,
            }

    return None

def choose_primary_operation(operations: list[str]) -> str | None:
    """
    Select the main operation when the query contains multiple operations.
    """
    if not operations:
        return None

    if "compare" in operations:
        return "compare"

    # Examples:
    # "highest growth" should primarily find the maximum.
    if "maximum" in operations and "growth" in operations:
        return "maximum"

    if "minimum" in operations and "growth" in operations:
        return "minimum"

    # Examples:
    # "maximum forecasted revenue" asks for the maximum forecast value.
    if "forecast" in operations and "maximum" in operations:
        return "maximum"

    if "forecast" in operations and "minimum" in operations:
        return "minimum"

    for operation in OPERATION_PRIORITY:
        if operation in operations:
            return operation

    return operations[0]


def determine_needs_chart(
    query: str,
    metrics: list[str],
    operations: list[str],
    dimension: str | None = None,
) -> bool:
    """Determine whether the result should include a chart."""
    normalized_query = query.lower()

    if any(term in normalized_query for term in EXPLICIT_CHART_TERMS):
        return True

    if "compare" in operations:
        return True

    if "trend" in operations:
        return True

    if any(
        phrase in normalized_query
        for phrase in TIME_SERIES_PHRASES
    ):
        return True

    if dimension is not None:
        return True

    if "correlation" in operations:
        return True

    return False


def determine_needs_prediction(operations: list[str]) -> bool:
    """Prediction is required only for forecast requests."""
    return "forecast" in operations


def determine_needs_llm(query: str) -> bool:
    """
    Use the LLM for explanations, recommendations, summaries, anomaly
    identification, and other open-ended analytical requests.
    """
    normalized_query = query.lower()

    llm_terms = (
        EXPLANATION_TERMS
        + RECOMMENDATION_TERMS
        + INSIGHT_TERMS
        + [
            "what stands out",
            "stand out",
            "unusual pattern",
            "unusual patterns",
            "anomaly",
            "anomalies",
            "interesting pattern",
            "important finding",
            "important findings",
            "key finding",
            "key findings",
            "business finding",
            "business findings",
        ]
    )

    return any(
        term in normalized_query
        for term in llm_terms
    )


def determine_needs_insights(
    query: str,
    operations: list[str],
    dimension: str | None = None,
) -> bool:
    """Determine whether generated analytical insights are needed."""
    normalized_query = query.lower()

    if any(term in normalized_query for term in INSIGHT_TERMS):
        return True

    if "compare" in operations:
        return True

    if "trend" in operations:
        return True

    if "growth" in operations:
        return True

    if any(
        phrase in normalized_query
        for phrase in TIME_SERIES_PHRASES
    ):
        return True

    if "forecast" in operations:
        return True

    if "margin" in normalized_query:
        return True

    if (
        "churn rate" in normalized_query
        or (
            "churn" in normalized_query
            and "rate" in normalized_query
        )
    ):
        return True

    if "correlation" in operations:
        return True

    if dimension is not None:
        return True

    return False

def extract_top_n(query: str) -> int:
    """
    Extract the requested ranking count from natural-language queries.

    Supports:
        "top 5 users"
        "bottom 3 states"
        "3 states with the lowest invoice value"
        "5 products have the most scans"
        "3 states have the highest quantity"

    Defaults to 5.
    """
    q = query.lower()

    # Explicit top/bottom N
    match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", q)
    if match:
        return int(match.group(1))

    # N ... highest/lowest/most/least
    match = re.search(
        r"\b(\d+)\s+\w+(?:\s+\w+){0,4}\s+"
        r"(?:with\s+the\s+|have\s+the\s+)"
        r"(?:highest|lowest|most|least)\b",
        q,
    )
    if match:
        return int(match.group(1))

    return 5


def detect_special_operation(query: str) -> str | None:
    """
    Detect operations that aren't present in the registry.
    """
    q = query.lower()

    # Explicit top-N phrasing:
    # "top 5 products", "top 3 states", etc.
    if re.search(r"\btop\s+\d+\b", q):
        return "top_users"

    # "N ... most" phrasing:
    # "5 products have the most scans"
    # "3 states have the highest quantity"
    # "10 products with the most scans"
    if re.search(
        r"\b\d+\s+\w+(?:\s+\w+){0,4}\s+"
        r"(?:have\s+the\s+most|with\s+the\s+most|"
        r"have\s+the\s+highest|with\s+the\s+highest)\b",
        q,
    ):
        return "top_users"

    # "most N" / "N highest" style ranking
    if re.search(r"\b\d+\s+(?:highest|most)\b", q):
        return "top_users"

    if re.search(r"\bbottom\s+\d+\b", q):
        return "bottom_users"

    if re.search(r"\b\d+\s+\w+(?:\s+\w+){0,4}\s+"
                 r"(?:have\s+the\s+lowest|with\s+the\s+lowest|"
                 r"have\s+the\s+least|with\s+the\s+least)\b", q):
        return "bottom_users"

    # Existing special cases
    if "top users" in q:
        return "top_users"

    if "top user" in q:
        return "top_users"

    if "best users" in q:
        return "top_users"

    if "unique users" in q:
        return "unique_users"

    if "distinct users" in q:
        return "unique_users"

    if "number of users" in q:
        return "unique_users"

    if "how many users" in q:
        return "unique_users"

    if "user count" in q:
        return "unique_users"

    if "bottom users" in q:
        return "bottom_users"

    if "bottom user" in q:
        return "bottom_users"

    if "worst users" in q:
        return "bottom_users"

    if "lowest users" in q:
        return "bottom_users"

    if "least users" in q:
        return "bottom_users"

    if "fewest" in q:
        return "bottom_users"

    if "least" in q:
        return "bottom_users"

    return None
    

def extract_filters(
    query: str,
    data: list[dict],
) -> list[dict]:
    """
    Extract simple natural-language filters.

    Returns the normalized condition-list format:

        [
            {
                "field": "...",
                "operator": "...",
                "value": "..."
            }
        ]

    Location resolution remains the responsibility of
    extract_location().
    """

    if not isinstance(query, str):
        return []

    normalized_query = query.casefold()
    filters: list[dict] = []

    # ---------------------------------------------------------
    # Negative location filters
    #
    # Examples:
    #   "not in Delhi"
    #   "is not in Delhi"
    # ---------------------------------------------------------

    negative_location_match = re.search(
        r"\b(?:is\s+)?not\s+in\s+([a-zA-Z][a-zA-Z\s-]*?)(?=\?|$|\s+(?:and|where)\b)",
        normalized_query,
    )

    if negative_location_match:
        location_text = (
            negative_location_match.group(1)
            .strip()
        )

        # extract_location() expects the phrase "in <location>"
        location_filter = extract_location(
            f"in {location_text}",
            data,
        )

        if location_filter:
            filters.append(
                {
                    "field": location_filter["field"],
                    "operator": "not_equals",
                    "value": location_filter["value"],
                }
            )

    # ---------------------------------------------------------
    # Existing month detection
    # ---------------------------------------------------------

    detected_months = [
        month
        for month in MONTHS
        if contains_alias(
            normalized_query,
            month,
        )
    ]

    if detected_months:
        filters.append(
            {
                "field": "month",
                "operator": "in",
                "value": detected_months,
            }
        )

    return filters

def analyze_request(
    query: str,
    data: list[dict],
):
    """
    Convert a natural-language request into a structured execution plan.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    if not query.strip():
        raise ValueError("query cannot be empty")

    metrics = extract_metrics(query)
    operations = extract_operations(query)

    special_operation = detect_special_operation(query)

    if special_operation and special_operation not in operations:
       operations.append(special_operation)



    dimension = extract_dimension(query)
    filters = extract_filters(
        query,
        data,
    )

    print("DEBUG EXTRACTED FILTERS:", filters)

    location = extract_location(
        query,
        data,
    )
    primary_operation = choose_primary_operation(operations)

    requires_chart = determine_needs_chart(
        query=query,
        metrics=metrics,
        operations=operations,
        dimension=dimension,
    )

    requires_insights = determine_needs_insights(
        query=query,
        operations=operations,
        dimension=dimension,
    )

    requires_llm = determine_needs_llm(query)

    requires_prediction = determine_needs_prediction(operations)

    return {
        "query": query,
        "metrics": metrics,
        "operations": operations,
        "dimension": dimension,
        "filters": filters,
        "location": location,
        "primary_operation": primary_operation,
        "requires_chart": requires_chart,
        "requires_insights": requires_insights,
        "requires_prediction": requires_prediction,
        "requires_llm": requires_llm,
        "top_n": extract_top_n(query),
    }


if __name__ == "__main__":
    sample_queries = [
        "What is the total revenue?",
        "Show monthly revenue growth as a chart",
        "Compare revenue and profit by quarter",
        "Forecast revenue for the next 3 months",
        "Explain why customer churn increased",
        "Show revenue by region",
        "What was the highest profit in January?",
    ]

    for sample_query in sample_queries:
        print("\nQuery:", sample_query)
        print(analyze_request(sample_query))