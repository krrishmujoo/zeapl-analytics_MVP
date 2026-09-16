import logging
from logging import config
import pandas as pd
import re

from aggregation_dispatcher import dispatch_aggregation
from request_analyser import (
    extract_metrics,
    extract_operations,
    extract_dimension,
    choose_primary_operation,
    detect_special_operation,
    extract_location,
    extract_top_n,
    determine_needs_llm,
    determine_needs_insights,
)

from registries import Operations
from aggregation import (
    total_metric,
    average_metric,
    highest_metric,
    lowest_metric,
    group_totals,
    correlation_coefficient,
    rollup_scan_activity_to_monthly,
)
from metric_resolver import resolve_metric, resolve_all_mentioned_metrics
from dataset_runtime_config import build_runtime_config
from dynamic_router import run_dynamic_forecast, apply_generic_filters
from semantic_planner import (
    build_semantic_plan,
    build_recovery_plan,
)
from chart import decide_chart, prepare_chart_data
from response_builder import build_response
from LLM import analyze_data


logger = logging.getLogger(__name__)

def _query_requests_distinct_count(query: str) -> bool:
    """
    Return True only when the user is actually asking for a count of
    entities/identifiers.
    """
    normalized_query = _normalize(query)

    count_patterns = (
        r"\bhow many\b",
        r"\bnumber of\b",
        r"\bcount of\b",
        r"\bcount\b",
        r"\bunique\b",
        r"\bdistinct\b",
        r"\btotal scans?\b",
        r"\btotal users?\b",
        r"\btotal products?\b",
        r"\btotal schemes?\b",
        r"\btotal devices?\b",
    )

    return any(
        re.search(pattern, normalized_query)
        for pattern in count_patterns
    )


def _is_identifier_count_metric(
    metric_key: str,
    config: dict,
) -> bool:
    metric_spec = config.get("metrics", {}).get(
        metric_key,
        {},
    )

    return (
        metric_spec.get("default_aggregation")
        in {"nunique", "count"}
        or metric_key.endswith("_distinct_count")
    )


def _remove_incidental_count_metrics(
    query: str,
    candidates: list[dict],
    config: dict,
) -> list[dict]:
    """
    Words such as 'scans' may describe filtered records rather than the
    requested metric. Identifier-count metrics are allowed only when the
    query explicitly asks for a count.
    """
    if _query_requests_distinct_count(query):
        return candidates

    return [
        candidate
        for candidate in candidates
        if not _is_identifier_count_metric(
            candidate.get("metric_key", ""),
            config,
        )
    ]
def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

# "trend"/"growth" both mean "show me the series over time" for this
# endpoint's purposes -- both map to the same execution path.
_TREND_OPERATIONS = {"trend", "growth"}

def _pluralize_dimension_name(name: str) -> str:
    """Return a useful English plural for schema-derived aliases."""
    if name.endswith("y") and len(name) > 1:
        return f"{name[:-1]}ies"

    if name.endswith(("s", "x", "z", "ch", "sh")):
        return f"{name}es"

    return f"{name}s"


def _resolve_dimension_phrase(
    phrase: str,
    data,
    config: dict,
) -> str | None:
    """
    Resolve natural-language entity wording to a real dataset dimension.

    The aliases are derived from the current dataset rather than tied to
    schemes, users, products, or any other fixed business schema.
    """
    df = (
        data
        if isinstance(data, pd.DataFrame)
        else pd.DataFrame(data)
    )

    available_dimensions = set(
        config.get("dimensions", [])
    )

    available_dimensions.update(
        column
        for column in df.columns
        if isinstance(column, str)
        and column.endswith("_id")
    )

    normalized_phrase = _normalize(phrase)

    # Remove harmless entity wording that people commonly include.
    normalized_phrase = re.sub(
        r"\b(?:the|all|each|every)\b",
        " ",
        normalized_phrase,
    )

    normalized_phrase = _normalize(normalized_phrase)

    matches = []

    for dimension in available_dimensions:
        if dimension not in df.columns:
            continue

        natural_name = _normalize(
            dimension.replace("_id", "")
        )

        plural_name = _pluralize_dimension_name(
            natural_name
        )

        aliases = {
            _normalize(dimension),
            natural_name,
            plural_name,
            f"{natural_name} id",
            f"{natural_name} ids",
            f"{plural_name} id",
            f"{plural_name} ids",
        }

        if normalized_phrase in aliases:
            matches.append(dimension)

    # Ambiguity must not be guessed.
    if len(matches) == 1:
        return matches[0]

    return None


_TOP_RANKING_WORDS = {
    "top",
    "highest",
    "best",
    "largest",
    "greatest",
    "most",
}

_BOTTOM_RANKING_WORDS = {
    "bottom",
    "lowest",
    "worst",
    "smallest",
    "least",
}


def _ranking_operation(word: str) -> str | None:
    normalized_word = _normalize(word)

    if normalized_word in _TOP_RANKING_WORDS:
        return "top_users"

    if normalized_word in _BOTTOM_RANKING_WORDS:
        return "bottom_users"

    return None

def _looks_like_nested_ranking_query(
    query: str,
) -> bool:
    """
    Return True when the wording clearly contains two ranking clauses.

    This allows unsupported or ambiguous n_extract_nested_ranking_requestested requests to fail safely
    instead of being silently flattened into a one-level ranking.
    """
    normalized_query = _normalize(query)

    ranking_word_pattern = (
        r"\b(?:top|highest|best|largest|greatest|most|"
        r"bottom|lowest|worst|smallest|least)\b"
    )

    ranking_words = re.findall(
        ranking_word_pattern,
        normalized_query,
    )

    has_nested_separator = bool(
        re.search(
            r"\b(?:in|inside|within|among)\b",
            normalized_query,
        )
    )

    return (
        len(ranking_words) >= 2
        and has_nested_separator
    )

def _extract_nested_ranking_request(
    query: str,
    data,
    config: dict,
) -> dict | None:
    """
    Detect a safe two-level ranking request.

    Supported examples:
    - Top 5 users in the top 3 schemes by invoice value
    - Highest 4 products within the highest 2 states by quantity
    - 5 best users inside the 3 best schemes by points earned
    - Bottom 3 products among the bottom 2 cities by quantity

    Returns outer level first, followed by inner level.
    """
    normalized_query = _normalize(query)

    ranking_words = (
        "top|highest|best|largest|greatest|most|"
        "bottom|lowest|worst|smallest|least"
    )

    separators = (
        "in|inside|within|among"
    )

    patterns = [
        re.compile(
            rf"\b(?P<inner_word>{ranking_words})\s+"
            r"(?P<inner_n>\d+)\s+"
            r"(?P<inner_phrase>.+?)\s+"
            rf"(?:{separators})\s+(?:the\s+)?"
            rf"(?P<outer_word>{ranking_words})\s+"
            r"(?P<outer_n>\d+)\s+"
            r"(?P<outer_phrase>.+?)(?:\s+by\s+|$)"
        ),
        re.compile(
            r"\b(?P<inner_n>\d+)\s+"
            rf"(?P<inner_word>{ranking_words})\s+"
            r"(?P<inner_phrase>.+?)\s+"
            rf"(?:{separators})\s+(?:the\s+)?"
            r"(?P<outer_n>\d+)\s+"
            rf"(?P<outer_word>{ranking_words})\s+"
            r"(?P<outer_phrase>.+?)(?:\s+by\s+|$)"
        ),
    ]

    match = None

    for pattern in patterns:
        match = pattern.search(normalized_query)

        if match:
            break

    if not match:
        return None

    inner_operation = _ranking_operation(
        match.group("inner_word")
    )

    outer_operation = _ranking_operation(
        match.group("outer_word")
    )

    # Mixed direction such as "top users in bottom schemes" requires a
    # richer plan than the current one-direction execution contract.
    if (
        inner_operation is None
        or outer_operation is None
        or inner_operation != outer_operation
    ):
        return None

    inner_dimension = _resolve_dimension_phrase(
        phrase=match.group("inner_phrase"),
        data=data,
        config=config,
    )

    outer_dimension = _resolve_dimension_phrase(
        phrase=match.group("outer_phrase"),
        data=data,
        config=config,
    )

    if (
        not inner_dimension
        or not outer_dimension
        or inner_dimension == outer_dimension
    ):
        return None

    inner_top_n = int(match.group("inner_n"))
    outer_top_n = int(match.group("outer_n"))

    if inner_top_n <= 0 or outer_top_n <= 0:
        return None

    return {
        "operation": inner_operation,
        "ranking_levels": [
            {
                "dimension": outer_dimension,
                "top_n": outer_top_n,
            },
            {
                "dimension": inner_dimension,
                "top_n": inner_top_n,
            },
        ],
    }


def _extract_nested_ranking_levels(
    query: str,
    data,
    config: dict,
) -> list[dict]:
    """
    Backward-compatible convenience wrapper.
    """
    request = _extract_nested_ranking_request(
        query=query,
        data=data,
        config=config,
    )

    if request is None:
        return []

    return request["ranking_levels"]

def resolve_semantic_request(
    query: str,
    data: list[dict],
    config: dict,
    plan_override: dict | None = None,
) -> dict | None:
    """
    Ask Claude to understand the user's intent and map it to the
    dynamically discovered dataset schema.

    Returns None if semantic planning fails so the existing deterministic
    resolver can remain the safe fallback.
    """
    if plan_override is not None:
        plan = plan_override
        logger.warning(
            "RECOVERY PLAN DEBUG: %s",
            plan,
        )

    else:
        try:
            plan = build_semantic_plan(
                query=query,
                data=data,
                config=config,
            )
            logger.warning(
                "SEMANTIC PLAN DEBUG: %s",
                plan,
            )

        except Exception as error:
            logger.exception(
                "Semantic planner failed; falling back to "
                "deterministic resolver: %s",
                error,
            )
            return None

    if not plan:
        return None

    nested_ranking = _extract_nested_ranking_request(
        query=query,
        data=data,
        config=config,
    )

    if nested_ranking is not None:
        ranking_levels = nested_ranking["ranking_levels"]

        plan["operation"] = nested_ranking["operation"]
        plan["ranking_levels"] = ranking_levels
        plan["dimension"] = ranking_levels[1]["dimension"]
        plan["top_n"] = ranking_levels[1]["top_n"]
        plan["group_by"] = [
            ranking_levels[1]["dimension"]
        ]

    metric_keys = plan.get("metric_keys", [])
    operation = plan.get("operation", "llm_analysis")

    # ------------------------------------------------------------------
    # Ranking recovery
    #
    # The semantic planner can identify a ranking operation while
    # occasionally returning dimension=None.
    #
    # Recover the ranking dimension deterministically before allowing
    # validation/execution to fail.
    # ------------------------------------------------------------------
    if operation in {"top_users", "bottom_users"} and not plan.get("dimension"):
        ranking_dimension = _resolve_ranking_dimension(
            data=data,
            config=config,
            resolved={
                "dimension": None,
                "query": query,
            },
            query=query,
        )

        if ranking_dimension:
            logger.warning(
                "RANKING DIMENSION RECOVERED: %s -> %s",
                query,
                ranking_dimension,
            )

            plan["dimension"] = ranking_dimension

            if not plan.get("group_by"):
                plan["group_by"] = [ranking_dimension]

            semantic_error = str(plan.get("error") or "")

            if "requires a real ranking dimension" in semantic_error.lower():
                plan["error"] = None

    operations_requiring_one_metric = {
        "total",
        "average",
        "maximum",
        "minimum",
        "trend",
        "forecast",
        "top_users",
        "bottom_users",
    }

    operations_requiring_two_metrics = {
        "compare",
        "correlation",
    }

    if (
        operation in operations_requiring_one_metric
        and len(metric_keys) < 1
    ):
        logger.warning(
            "Semantic planner returned operation %r without a metric; "
            "falling back to deterministic resolution.",
            operation,
        )
        return None

    if (
        operation in operations_requiring_two_metrics
        and len(metric_keys) < 2
    ):
        logger.warning(
            "Semantic planner returned operation %r with fewer than "
            "two metrics; falling back to deterministic resolution.",
            operation,
        )
        return None
    requires_forecast = operation == "forecast"

    requires_llm = (
        plan.get("requires_llm", False)
        or operation == "llm_analysis"
    )

    # Existing LLM execution path understands "explain".
    if operation == "llm_analysis":
        execution_operation = "explain"
    else:
        execution_operation = operation

    dimension = plan.get("dimension")

    # For time-based requests, preserve the granularity Claude understood.
    if (
        plan.get("granularity")
        and execution_operation in {
            "trend",
            "compare",
            "total",
            "average",
            "maximum",
            "minimum",
        }
    ):
        dimension = plan["granularity"]

    metric_fields = [
        config["metrics"][key]["field"]
        for key in metric_keys
        if key in config.get("metrics", {})
    ]

    requires_chart = (
        execution_operation in {
            "trend",
            "compare",
            "correlation",
        }
        or dimension is not None
        or bool(plan.get("group_by", []))
        or execution_operation in {
            "top_users",
            "bottom_users",
        }
    )

    return {
        "query": query,
        "metric_keys": metric_keys,
        "metric_fields": metric_fields,
        "field": plan.get("field"),
        "operation": execution_operation,

        "location": None,
        "filters": plan.get("filters", {}),        "top_n": plan.get("top_n"),
        "ranking_levels": plan.get("ranking_levels", []),
        "dimension": dimension,
        "group_by": plan.get("group_by", []),
                "filter_logic": plan.get(
            "filter_logic",
            "and",
        ),
        "requires_chart": requires_chart,
        "requires_forecast": requires_forecast,
        "requires_llm": requires_llm,
        "requires_insights": requires_llm,
        "granularity": plan.get("granularity"),
        "horizon": plan.get("horizon"),
        "resolution_method": plan.get(
            "resolution_method",
            "llm_semantic_plan",
        ),
        "resolution_confidence": plan.get("confidence", 0.0),
        "semantic_reason": plan.get("semantic_reason", ""),
        "error": plan.get("error"),
    }



def resolve_request(
    query: str,
    data: list[dict],
    config: dict,
    metric_override: str | None = None,
    granularity_override: str | None = None,
    horizon_override: int | None = None,
) -> dict:
    """
    Resolve a question into an executable request shape.

    Returns an ``error`` value of ``None`` on success. When ``error`` is set,
    callers should return a clear client-facing error rather than silently
    guessing an unsupported metric.
    """
    operations = extract_operations(query)
    special_operation = detect_special_operation(query)
    requires_llm = determine_needs_llm(query)
    location = extract_location(query, data)

    dynamic_filters = extract_dynamic_filters(
        query=query,
        data=data,
        config=config,
    )
    print("DEBUG DYNAMIC FILTERS:", dynamic_filters)
    if special_operation and special_operation not in operations:
        operations.append(special_operation)

    requires_forecast = "forecast" in operations
    operation = (
        "forecast"
        if requires_forecast
        else choose_primary_operation(operations)
    )

    special_user_operations = {
        "top_users",
        "bottom_users",
        "unique_users",
    }

    if operation in _TREND_OPERATIONS:
        operation = "trend"
    elif operation is None:
        operation = "explain" if requires_llm else "total"

    dimension = extract_dimension(query)
    print("DEBUG GROUPING CHECK:", {
        "query": query,
        "initial_dimension": dimension,
        "available_dimensions": config.get("dimensions", []),
    })


    # Resolve natural-language grouping requests such as:
    # "each user", "per user", "by user", etc.
    #
    # Examples:
    #   "How many scans did each user make?" -> user_id
    #   "How much invoice value per product?" -> product_id
    #   "Quantity by scheme" -> scheme_id

    if dimension is None:
        query_lower = query.lower()

        grouping_aliases = {
            "user_id": [
                "each user",
                "per user",
                "by user",
                "for each user",
                "user",
                "users",
                "user id",
                "user ids",
            ],

            "product_id": [
                "each product",
                "per product",
                "by product",
                "for each product",
                "products",
                "product ids",
            ],

            "scheme_id": [
                "each scheme",
                "per scheme",
                "by scheme",
                "for each scheme",
                "scheme",
                "schemes",
                "scheme id",
                "scheme ids",
            ],
            "state": [
                "each state",
                "per state",
                "by state",
                "for each state",
                "states",
            ],
            "city": [
                "each city",
                "per city",
                "by city",
                "for each city",
                "cities",
            ],
        }

        for group_dimension, aliases in grouping_aliases.items():
            available_group_dimensions = set(
                config.get("dimensions", [])
            )

            # Identifier fields can also be valid grouping dimensions.
            for metric_spec in config.get("metrics", {}).values():
                if not isinstance(metric_spec, dict):
                    continue

                metric_field = metric_spec.get("field")

                if metric_field:
                    available_group_dimensions.add(
                        metric_field
                    )

            if (
                group_dimension in available_group_dimensions
                and any(alias in query_lower for alias in aliases)
            ):
                dimension = group_dimension
                break

        print("DEBUG GROUPING RESULT:", dimension)
    print("DEBUG RANKING DIMENSION BEFORE:", {
        "operation": operation,
        "dimension": dimension,
        "query": query,
    })

    if operation in {"top_users", "bottom_users"} and dimension is None:
        query_lower = query.lower()

        ranking_aliases = {
            "product_id": (
                "product",
                "products",
                "product id",
                "product ids",
            ),
            "user_id": (
                "user",
                "users",
                "user id",
                "user ids",
            ),
            "scheme_id": (
                "scheme",
                "schemes",
                "scheme id",
                "scheme ids",
            ),
            "state": (
                "state",
                "states",
            ),
            "city": (
                "city",
                "cities",
            ),
        }

        data_columns = set(
            data.columns
            if isinstance(data, pd.DataFrame)
            else pd.DataFrame(data).columns
        )

        for ranking_dimension, aliases in ranking_aliases.items():
            if (
                ranking_dimension in data_columns
                and any(alias in query_lower for alias in aliases)
            ):
                dimension = ranking_dimension
                break

    if operation in {"top_users", "bottom_users"} and dimension is None:
        dimension = _resolve_ranking_dimension(
            data=data,
            config=config,
            resolved={"dimension": None},
            query=query,

        )

    print("DEBUG RANKING DIMENSION AFTER:", dimension)
    if dimension is None:
        query_lower = query.lower()
        for dim in config.get("dimensions", []):
            dim_lower = dim.lower()
            if (
                dim_lower in query_lower
                or dim_lower.replace("_", " ") in query_lower
            ):
                dimension = dim
                break

    # Resolve ranking dimensions dynamically from the dataset.
    # This handles queries such as:
    # "top 3 cities by invoice value"
    # "bottom 5 products by scans"
    if operation in {"top_users", "bottom_users"} and dimension is None:
        dimension = _resolve_ranking_dimension(
            data=data,
            config=config,
            resolved={"dimension": None},
        )

    requires_insights = determine_needs_insights(
        query=query,
        operations=operations,
        dimension=dimension,
    )

    if operation == "unique_users":
        user_metric_key = None

        for key, spec in config.get("metrics", {}).items():
            if (
                isinstance(spec, dict)
                and spec.get("field") == "user_id"
                and spec.get("default_aggregation") == "nunique"
            ):
                user_metric_key = key
                break

        if user_metric_key:
            return {
                "query": query,
                "metric_keys": [user_metric_key],
                "metric_fields": ["user_id"],
                "operation": "total",
                "top_n": extract_top_n(query),
                "dimension": dimension,
                "filters": {},
                "location": location,
                "requires_chart": False,
                "requires_forecast": False,
                "requires_llm": False,
                "requires_insights": requires_insights,
                "granularity": granularity_override,
                "horizon": horizon_override,
                "resolution_method": "dynamic_identifier_metric",
                "resolution_confidence": 1.0,
                "error": None,
            }

        # Preserve the old special-operation behavior if no
        # dynamically discovered user metric exists.
        return {
            "query": query,
            "metric_keys": [],
            "metric_fields": [],
            "operation": operation,
            "top_n": extract_top_n(query),
            "dimension": dimension,
            "location": location,
            "filters": dynamic_filters,
            "requires_chart": False,
            "requires_forecast": False,
            "requires_llm": requires_llm,
            "requires_insights": requires_insights,
            "granularity": granularity_override,
            "horizon": horizon_override,
            "resolution_method": "special_operation",
            "resolution_confidence": 1.0,
            "error": None,
        }

    if metric_override:
        if metric_override not in config["metrics"]:
            return {
                "error": (
                    f"unknown metric {metric_override!r} for this dataset. "
                    f"Available metrics: {sorted(config['metrics'].keys())}"
                ),
                "operation": operation,
                "dimension": dimension,
            }

        metric_keys = [metric_override]
        resolution_method = "explicit_override"
        resolution_confidence = 1.0

    elif operation == "compare" or requires_llm:
        resolution_query = query

        if (
            "revenue" in query.lower()
            and "invoice_value" in config["metrics"]
        ):
            resolution_query = re.sub(
                r"\brevenue\b",
                "invoice value",
                query,
                flags=re.IGNORECASE,
            )

        candidates = resolve_all_mentioned_metrics(
            resolution_query,
            config,
        )

        candidates = _remove_incidental_count_metrics(
            query=query,
            candidates=candidates,
            config=config,
        )

        if operation == "compare" and len(candidates) < 2:
            return {
                "error": (
                    "a comparison needs at least two recognizable metrics "
                    "in the question. "
                    f"Available metrics: {sorted(config['metrics'].keys())}"
                ),
                "operation": operation,
                "dimension": dimension,
            }

        if not candidates:
            # General LLM questions such as dataset summaries may not name
            # a specific metric. Allow them through to the LLM execution
            # path instead of treating that as a resolution failure.
            metric_keys = []
            resolution_method = "llm_general"
            resolution_confidence = 1.0
        else:
            metric_keys = [
                candidate["metric_key"]
                for candidate in candidates
            ]
            resolution_method = "multi_metric_match"
            resolution_confidence = 0.8
    else:
        normalized_query = query.lower()

        if (
            "revenue" in normalized_query
            and "invoice_value" in config["metrics"]
        ):
            metric_keys = ["invoice_value"]
            resolution_method = "business_alias"
            resolution_confidence = 1.0

        else:
            # ---------------------------------------------------------
            # IMPORTANT:
            # Resolve the requested metric separately from metrics
            # that only appear inside filter conditions.
            #
            # Example:
            #   "total invoice value where quantity > 100"
            #
            # invoice_value = requested metric
            # quantity      = filter field
            #
            # We must not let "quantity" win metric resolution simply
            # because it appears in the question.
            # ---------------------------------------------------------

           # ---------------------------------------------------------
            # Resolve the requested metric from the main question,
            # separately from metric fields used inside filters.
            #
            # Example:
            #
            # "total invoice value for scans where
            #  quantity > 3 and invoice value > 5000"
            #
            # Requested metric:
            #     invoice_value
            #
            # Filter metrics:
            #     quantity
            #     invoice_value
            # ---------------------------------------------------------

            metric_resolution_query = query

            # A filter clause normally begins after "where".
            # Resolve the requested metric from the part before it.
            where_match = re.search(
                r"\bwhere\b",
                metric_resolution_query,
                flags=re.IGNORECASE,
            )

            if where_match:
                metric_resolution_query = (
                    metric_resolution_query[
                        :where_match.start()
                    ].strip()
                )

            print(
                "DEBUG METRIC RESOLUTION QUERY:",
                metric_resolution_query,
            )

            resolution = resolve_metric(
                metric_resolution_query,
                config,
            )

            if (
                resolution.get("resolved")
                and _is_identifier_count_metric(
                    resolution.get("metric_key", ""),
                    config,
                )
                and not _query_requests_distinct_count(query)
            ):
                resolution = {
                    "resolved": False,
                    "resolution_method": "incidental_identifier_match",
                    "resolution_confidence": 0.0,
                    "candidates": [],
                    "message": (
                        "an identifier/count term appeared in the query, "
                        "but the user did not request a distinct count"
                    ),
                }

            if not resolution.get("resolved"):
                available = sorted(
                    config["metrics"].keys()
                )

                candidates = resolution.get(
                    "candidates",
                    []
                )

                message = resolution.get(
                    "message",
                    "could not resolve a metric for this question",
                )

                if candidates:
                    message += (
                        "; possible matches: "
                        f"{[candidate['metric_key'] for candidate in candidates]}"
                    )

                return {
                    "error": (
                        f"{message}. "
                        f"Available metrics: {available}"
                    ),
                    "operation": operation,
                    "dimension": dimension,
                }

            metric_keys = [
                resolution["metric_key"]
            ]

            resolution_method = (
                resolution["resolution_method"]
            )

            resolution_confidence = (
                resolution["resolution_confidence"]
            )

    requires_chart = (
        operation in ("trend", "compare")
        or dimension is not None
        or (
            "correlation" in operations
            and len(metric_keys) >= 2
        )
    )

    print(
    "DEBUG RESOLVED:",
    {
        "query": query,
        "operation": operation,
        "metric_keys": metric_keys,
        "dimension": dimension,
        "requires_llm": requires_llm,
        "requires_insights": requires_insights,
    },
)



    metric_fields = [
            config["metrics"][key]["field"]
            for key in metric_keys
            if key in config.get("metrics", {})
        ]

    nested_ranking = _extract_nested_ranking_request(
        query=query,
        data=data,
        config=config,
    )

    ranking_levels = (
        nested_ranking["ranking_levels"]
        if nested_ranking is not None
        else []
    )

    if nested_ranking is not None:
        operation = nested_ranking["operation"]
        dimension = ranking_levels[1]["dimension"]

    return {
        "query": query,
        "metric_keys": metric_keys,
        "metric_fields": metric_fields,
        "operation": operation,
        "top_n": (
            ranking_levels[1]["top_n"]
            if len(ranking_levels) == 2
            else extract_top_n(query)
        ),
        "ranking_levels": ranking_levels,
        "dimension": dimension,
        "group_by": [dimension] if dimension else [],
        "location": location,
        "filters": dynamic_filters,
        "requires_chart": requires_chart,
        "requires_forecast": requires_forecast,
        "requires_llm": requires_llm,
        "requires_insights": requires_insights,
        "granularity": granularity_override,
        "horizon": horizon_override,
        "resolution_method": resolution_method,
        "resolution_confidence": resolution_confidence,
        "error": None,
    }


def _is_negated_filter_value(
    normalized_query: str,
    value_text: str,
) -> bool:
    """
    Return True when a categorical value is explicitly excluded.
    """

    escaped_value = re.escape(value_text)

    patterns = (
        rf"\boutside\s+(?:of\s+)?{escaped_value}\b",
        rf"\bexcluding\s+{escaped_value}\b",
        rf"\bexcept\s+{escaped_value}\b",
        rf"\bother\s+than\s+{escaped_value}\b",
        rf"\bnot\s+in\s+{escaped_value}\b",
        rf"\bnot\s+{escaped_value}\b",
    )

    matched = any(
        re.search(pattern, normalized_query)
        for pattern in patterns
    )

    print(
        "DEBUG NEGATION:",
        {
            "query": normalized_query,
            "value": value_text,
            "matched": matched,
        },
    )

    return matched


def extract_dynamic_filters(
    query: str,
    data: list[dict],
    config: dict,
) -> list[dict]:
    """
    Dynamically discover filters from the current dataset schema and values.

    Does not assume specific fields such as state, city, or is_valid.
    """

    if not data:
        return []

    df = pd.DataFrame(data)
    normalized_query = _normalize(query)
    filters = []

    # ---------------------------------------------------------
    # 1. Boolean-like fields
    # ---------------------------------------------------------
    for field, profile in config.get("column_profiles", {}).items():
        if field not in df.columns:
            continue

        semantic_type = profile.get("semantic_type")

        if semantic_type == "boolean":
            print(
                    "DEBUG BOOLEAN FIELD:",
                    field,
                    profile,
                    "QUERY:",
                    normalized_query,
                )
            if re.search(r"\bvalid\b", normalized_query):
                filters.append(
                    {
                        "field": field,
                        "operator": "equals",
                        "value": 1,
                    }
                )

            elif re.search(r"\binvalid\b", normalized_query):
                filters.append(
                    {
                        "field": field,
                        "operator": "equals",
                        "value": 0,
                    }
                )

    # ---------------------------------------------------------
    # 2. Match actual categorical values in the dataset
    # ---------------------------------------------------------

    boolean_filter_detected = any(
        condition.get("field") in df.columns
        and condition.get("field") in config.get("column_profiles", {})
        and config["column_profiles"][condition["field"]].get("semantic_type") == "boolean"
        for condition in filters
    )
    for field, profile in config.get("column_profiles", {}).items():
        if field not in df.columns:
            continue

        print(
            "DEBUG FIELD PROFILE:",
            {
                "field": field,
                "semantic_type": profile.get("semantic_type"),
            },
        )

        if profile.get("semantic_type") not in {
            "categorical",
            "geographic",
        }:
            continue

        values = df[field].dropna().unique()

        print(
            "DEBUG CATEGORICAL FIELD:",
            {
                "field": field,
                "sample_values": [
                    str(v)
                    for v in values[:20]
                ],
                "query": normalized_query,
            },
        )

        matches = []

        for value in values:
            value_text = _normalize(str(value))

            if not value_text:
                continue

            if boolean_filter_detected and value_text in {"valid", "invalid"}:
                continue

            if re.search(
                rf"(?<!\w){re.escape(value_text)}(?!\w)",
                normalized_query,
            ):
                matches.append(value)

        if len(matches) == 1:

            print(
                "DEBUG CATEGORICAL MATCH:",
                {
                    "field": field,
                    "matches": matches,
                },
            )

            matched_value = matches[0]
            matched_value_text = _normalize(
                str(matched_value)
            )

            operator = (
                "not_equals"
                if _is_negated_filter_value(
                    normalized_query,
                    matched_value_text,
                )
                else "equals"
            )

            filters.append(
                {
                    "field": field,
                    "operator": operator,
                    "value": matched_value,
                }
            )

        elif len(matches) > 1:

            is_negated = bool(
                re.search(
                    r"\bnot\s+in\b",
                    normalized_query,
                )
                or re.search(
                    r"\boutside\s+(?:of\s+)?",
                    normalized_query,
                )
                or re.search(
                    r"\bexcluding\b",
                    normalized_query,
                )
                or re.search(
                    r"\bexcept\b",
                    normalized_query,
                )
            )

            filters.append(
                {
                    "field": field,
                    "operator": "not_in" if is_negated else "in",
                    "value": matches,
                }
            )




        # ---------------------------------------------------------
    # Date range filters
    # ---------------------------------------------------------

    date_column = config.get("date_column")

    if date_column and date_column in df.columns:

        # -----------------------------------------------------
        # Explicit date range
        #
        # Examples:
        #   between January 1, 2026 and March 31, 2026
        #   from January 1, 2026 to March 31, 2026
        # -----------------------------------------------------

        month_names = (
            "january|february|march|april|may|june|"
            "july|august|september|october|november|december"
        )

        date_pattern = (
            rf"(?:between|from)\s+"
            rf"({month_names})\s+"
            rf"(\d{{1,2}}),?\s+"
            rf"(\d{{4}})"
            rf"\s+(?:and|to)\s+"
            rf"({month_names})\s+"
            rf"(\d{{1,2}}),?\s+"
            rf"(\d{{4}})"
        )

        date_match = re.search(
            date_pattern,
            normalized_query,
            re.IGNORECASE,
        )

        if date_match:

            start_month = date_match.group(1)
            start_day = date_match.group(2)
            start_year = date_match.group(3)

            end_month = date_match.group(4)
            end_day = date_match.group(5)
            end_year = date_match.group(6)

            try:
                start_date = pd.to_datetime(
                    f"{start_month} {start_day}, {start_year}"
                )

                end_date = pd.to_datetime(
                    f"{end_month} {end_day}, {end_year}"
                )

                filters.append(
                    {
                        "field": date_column,
                        "operator": "date_from",
                        "value": start_date.strftime("%Y-%m-%d"),
                    }
                )

                filters.append(
                    {
                        "field": date_column,
                        "operator": "date_to",
                        "value": end_date.strftime("%Y-%m-%d"),
                    }
                )

            except (ValueError, TypeError):
                pass





        # ---------------------------------------------------------
        # 3. Numeric comparison filters
        # ---------------------------------------------------------

        numeric_fields = set()

        # Discover numeric columns from the actual dataframe.
        for field in df.columns:
            if pd.api.types.is_numeric_dtype(df[field]):
                # Boolean columns such as is_valid are handled above.
                profile = config.get(
                    "column_profiles",
                    {},
                ).get(field, {})

                if profile.get("semantic_type") != "boolean":
                    numeric_fields.add(field)

        # Also include metric fields discovered by the runtime config.
        for metric_spec in config.get(
            "metrics",
            {},
        ).values():

            if not isinstance(metric_spec, dict):
                continue

            metric_field = metric_spec.get("field")

            if (
                metric_field
                and metric_field in df.columns
                and pd.api.types.is_numeric_dtype(
                    df[metric_field]
                )
            ):
                numeric_fields.add(metric_field)

        # Build natural-language aliases for each numeric field.
        field_aliases = {}

        for field in numeric_fields:

            aliases = {
                field,
                field.replace("_", " "),
            }

            for metric_key, metric_spec in config.get(
                "metrics",
                {},
            ).items():

                if not isinstance(metric_spec, dict):
                    continue

                if metric_spec.get("field") != field:
                    continue

                aliases.add(metric_key)

                display_name = metric_spec.get(
                    "display_name"
                )

                if display_name:
                    aliases.add(
                        str(display_name)
                    )

                for alias in metric_spec.get(
                    "aliases",
                    [],
                ):
                    aliases.add(
                        str(alias)
                    )

            field_aliases[field] = sorted(
                {
                    _normalize(alias)
                    for alias in aliases
                    if alias
                },
                key=len,
                reverse=True,
            )

        # ---------------------------------------------------------
        # Comparison operators
        # ---------------------------------------------------------

        comparison_patterns = [
            (
                "greater_than_or_equal",
                r"(?:greater\s+than\s+or\s+equal\s+to|at\s+least|>=)",
            ),
            (
                "less_than_or_equal",
                r"(?:less\s+than\s+or\s+equal\s+to|at\s+most|<=)",
            ),
            (
                "greater_than",
                r"(?:greater\s+than|more\s+than|above|over|>)",
            ),
            (
                "less_than",
                r"(?:less\s+than|fewer\s+than|below|under|<)",
            ),
        ]

        for field, aliases in field_aliases.items():

            found_comparison = False

            for alias in aliases:

                if not alias:
                    continue

                escaped_alias = re.escape(alias)

                # -------------------------------------------------
                # between
                # Example:
                # quantity between 50 and 100
                # -------------------------------------------------

                between_pattern = (
                    rf"\b{escaped_alias}\b"
                    rf"\s+between\s+"
                    rf"(-?\d+(?:\.\d+)?)"
                    rf"\s+and\s+"
                    rf"(-?\d+(?:\.\d+)?)"
                )

                between_match = re.search(
                    between_pattern,
                    normalized_query,
                )

                if between_match:

                    lower = float(
                        between_match.group(1)
                    )

                    upper = float(
                        between_match.group(2)
                    )

                    if lower.is_integer():
                        lower = int(lower)

                    if upper.is_integer():
                        upper = int(upper)

                    filters.append(
                        {
                            "field": field,
                            "operator": "between",
                            "value": [
                                lower,
                                upper,
                            ],
                        }
                    )

                    found_comparison = True
                    break

                # -------------------------------------------------
                # greater / less comparisons
                # -------------------------------------------------

                for operator, operator_pattern in comparison_patterns:

                    comparison_pattern = (
                        rf"\b{escaped_alias}\b"
                        rf"\s*(?:is\s+)?"
                        rf"{operator_pattern}"
                        rf"\s*(-?\d+(?:\.\d+)?)"
                    )

                    match = re.search(
                        comparison_pattern,
                        normalized_query,
                    )

                    if not match:
                        continue

                    numeric_value = float(
                        match.group(1)
                    )

                    if numeric_value.is_integer():
                        numeric_value = int(
                            numeric_value
                        )

                    filters.append(
                        {
                            "field": field,
                            "operator": operator,
                            "value": numeric_value,
                        }
                    )

                    found_comparison = True
                    break

                if found_comparison:
                    break

    return filters

# ---------------------------------------------------------------------------
# Deterministic aggregation execution (non-forecast operations)
# ---------------------------------------------------------------------------

def _period_key(record: dict, date_column: str, granularity: str) -> str | None:
    import pandas as pd
    raw = record.get(date_column)
    if raw is None:
        return None
    ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        return None
    if granularity == "year":
        return str(ts.year)
    if granularity == "quarter":
        return f"{ts.year}-Q{ts.quarter}"
    if granularity == "week":
        return f"{ts.isocalendar().year}-W{ts.isocalendar().week:02d}"
    return ts.strftime("%Y-%m")  # month (default time grouping for "by month")


def _group_by_time(
    data,
    metric_spec: dict,
    date_column: str,
    granularity: str,
    operation: str = "total",
) -> dict:
    """Group a metric over time while respecting its discovered aggregation."""
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    field = metric_spec["field"]
    if date_column not in df.columns or field not in df.columns:
        return {}

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])
    if df.empty:
        return {}

    if granularity == "year":
        df["_period"] = df[date_column].dt.strftime("%Y")
    elif granularity == "quarter":
        df["_period"] = df[date_column].dt.to_period("Q").astype(str)
    elif granularity == "week":
        iso = df[date_column].dt.isocalendar()
        df["_period"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    elif granularity == "day":
        df["_period"] = df[date_column].dt.strftime("%Y-%m-%d")
    else:
        df["_period"] = df[date_column].dt.strftime("%Y-%m")

    default_agg = metric_spec.get("default_aggregation", "sum")
    grouped = df.groupby("_period", sort=True)[field]

    if default_agg == "nunique":
        result = grouped.nunique(dropna=True)
    elif default_agg == "count":
        result = grouped.count()
    else:
        df[field] = pd.to_numeric(df[field], errors="coerce")
        grouped = df.groupby("_period", sort=True)[field]

        if operation == "average":
            result = grouped.mean()
        elif operation == "maximum":
            result = grouped.max()
        elif operation == "minimum":
            result = grouped.min()
        elif default_agg == "mean":
            result = grouped.mean()
        else:
            result = grouped.sum(min_count=1)

    result = result.dropna()
    return {str(period): float(value) for period, value in result.items()}


def _display_name(config: dict, metric_key: str) -> str:
    return config["metrics"][metric_key].get("display_name", metric_key)


def _aggregate_metric(
    data,
    metric_spec: dict,
    operation: str | None = None,
):
    """Aggregate one discovered metric using its MetricSpec."""
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    field = metric_spec["field"]
    if field not in df.columns:
        return None

    series = df[field]
    default_agg = metric_spec.get("default_aggregation", "sum")

    # Identifier-derived metrics represent counts of unique entities.
    if default_agg == "nunique":
        return int(series.dropna().nunique())
    if default_agg == "count":
        return int(series.notna().sum())

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None

    if operation == "average":
        return float(numeric.mean())
    if operation == "maximum":
        return float(numeric.max())
    if operation == "minimum":
        return float(numeric.min())
    if default_agg == "mean":
        return float(numeric.mean())
    return float(numeric.sum())


def _group_metric(
    data,
    metric_spec: dict,
    dimension: str,
    operation: str,
) -> dict:
    """Group one metric by one dimension using its MetricSpec."""
    df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    field = metric_spec["field"]
    if field not in df.columns or dimension not in df.columns:
        return {}

    df = df.dropna(subset=[dimension]).copy()
    default_agg = metric_spec.get("default_aggregation", "sum")
    grouped = df.groupby(dimension, dropna=False)[field]

    if default_agg == "nunique":
        result = grouped.nunique(dropna=True)
    elif default_agg == "count":
        result = grouped.count()
    else:
        df[field] = pd.to_numeric(df[field], errors="coerce")
        grouped = df.groupby(dimension, dropna=False)[field]

        if operation == "average":
            result = grouped.mean()
        elif operation == "maximum":
            result = grouped.max()
        elif operation == "minimum":
            result = grouped.min()
        elif default_agg == "mean":
            result = grouped.mean()
        else:
            result = grouped.sum(min_count=1)

    result = result.dropna()
    return {str(key): float(value) for key, value in result.items()}

def _resolve_ranking_dimension(
    data,
    config: dict,
    resolved: dict,
    query: str | None = None,
) -> str | None:
    """
    Find the real dataset dimension representing the entity being ranked.
    Prefer the dimension explicitly resolved from the user's query.
    """
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

    requested_dimension = resolved.get("dimension")
    if query:
        normalized_query = query.lower()

        dimension_aliases = {
            "city": ("city", "cities"),
            "state": ("state", "states"),
            "region": ("region", "regions"),
            "product_id": ("product", "products"),
            "user_id": ("user", "users"),
            "scheme_id": ("scheme", "schemes"),
        }

        for dimension, aliases in dimension_aliases.items():
            if dimension not in df.columns:
                continue

            if any(
                re.search(
                    rf"\b{re.escape(alias)}\b",
                    normalized_query,
                )
                for alias in aliases
            ):
                return dimension

    # 1. Use the dimension already resolved from the query.
    if (
        requested_dimension
        and requested_dimension in df.columns
    ):
        return requested_dimension

    # 2. Check dynamically discovered dimensions against the query.
    query = str(resolved.get("query", "")).lower()

    available_dimensions = [
        dimension
        for dimension in config.get("dimensions", [])
        if dimension in df.columns
    ]

    # Prefer an exact field-name / natural-language match.
    for dimension in available_dimensions:
        normalized_dimension = dimension.lower().replace("_", " ")

        if normalized_dimension in query:
            return dimension

    # 3. Check common entity aliases dynamically.
    dimension_aliases = {
        "state": ["state", "states"],
        "city": ["city", "cities"],
        "product_id": ["product", "products", "product id"],
        "scheme_id": ["scheme", "schemes", "scheme id"],
        "device_id": ["device", "devices", "device id"],
        "user_id": ["user", "users", "user id"],
        "customer_id": ["customer", "customers", "customer id"],
        "vendor_id": ["vendor", "vendors", "vendor id"],
    }

    for dimension, aliases in dimension_aliases.items():
        if dimension not in df.columns:
            continue

        if any(alias in query for alias in aliases):
            return dimension

    return None


def execute_request(data: list[dict], config: dict, resolved: dict) -> dict:

    """
    Executes a resolved, non-forecast request deterministically. Returns
    {"result", "insights", "chart_spec", "chart_data", "warnings"}.
    """
    warnings = []
    operation = resolved["operation"]
    metric_keys = resolved["metric_keys"]
    metric_fields_map = {k: config["metrics"][k]["field"] for k in metric_keys}
    metric_display_map = {k: _display_name(config, k) for k in metric_keys}
    date_column = config.get("date_column")
    dimension = resolved.get("dimension")
    group_by = resolved.get("group_by", [])

    if not isinstance(group_by, list):
        group_by = []

    time_group_fields = {
        "day",
        "week",
        "month",
        "quarter",
        "year",
    }

    data_columns = set(
        data.columns
        if isinstance(data, pd.DataFrame)
        else pd.DataFrame(data).columns
    )

    group_by = [
        field
        for field in group_by
        if (
            field in data_columns
            or field in time_group_fields
        )
    ]

    result = None
    insights = None
    chart_spec = None
    chart_data = None

    is_time_dimension = dimension in ("day", "week", "month", "quarter", "year")

    if operation == "correlation":
        if len(metric_keys) != 2:
            return {
                "result": None,
                "insights": None,
                "chart_spec": None,
                "chart_data": None,
                "warnings": [
                    "Correlation requires exactly two metrics."
                ],
            }

        df = (
            data.copy()
            if isinstance(data, pd.DataFrame)
            else pd.DataFrame(data)
        )

        first_key = metric_keys[0]
        second_key = metric_keys[1]

        first_spec = config["metrics"][first_key]
        second_spec = config["metrics"][second_key]

        first_field = first_spec["field"]
        second_field = second_spec["field"]

        first_label = metric_display_map[first_key]
        second_label = metric_display_map[second_key]

        correlation_frame = None
        working_df = df.copy()
        grouping_columns = []
        output_group_names = {}

        requested_groups = list(group_by)

        if not requested_groups and dimension:
            requested_groups = [dimension]

        granularity = resolved.get("granularity")

        time_fields = {
            "day",
            "week",
            "month",
            "quarter",
            "year",
        }

        requested_time_group = next(
            (
                field
                for field in requested_groups
                if field in time_fields
            ),
            None,
        )

        if not requested_time_group and granularity in time_fields:
            requested_time_group = granularity

        if requested_time_group:
            if not date_column or date_column not in working_df.columns:
                return {
                    "result": None,
                    "insights": None,
                    "chart_spec": None,
                    "chart_data": None,
                    "warnings": [
                        "A date column is required for "
                        "time-grouped correlation."
                    ],
                }

            parsed_dates = pd.to_datetime(
                working_df[date_column],
                errors="coerce",
            )

            if requested_time_group == "day":
                working_df["__correlation_time__"] = (
                    parsed_dates.dt.strftime("%Y-%m-%d")
                )
            elif requested_time_group == "week":
                working_df["__correlation_time__"] = (
                    parsed_dates.dt.to_period("W").astype(str)
                )
            elif requested_time_group == "month":
                working_df["__correlation_time__"] = (
                    parsed_dates.dt.to_period("M").astype(str)
                )
            elif requested_time_group == "quarter":
                working_df["__correlation_time__"] = (
                    parsed_dates.dt.to_period("Q").astype(str)
                )
            else:
                working_df["__correlation_time__"] = (
                    parsed_dates.dt.year
                )

            grouping_columns.append("__correlation_time__")
            output_group_names[
                "__correlation_time__"
            ] = requested_time_group

        for group_field in requested_groups:
            if group_field in time_fields:
                continue

            if group_field not in working_df.columns:
                return {
                    "result": None,
                    "insights": None,
                    "chart_spec": None,
                    "chart_data": None,
                    "warnings": [
                        f"Correlation grouping field "
                        f"{group_field!r} does not exist."
                    ],
                }

            grouping_columns.append(group_field)
            output_group_names[group_field] = group_field

        if grouping_columns:
            def grouped_metric_series(metric_spec):
                metric_field = metric_spec["field"]

                if metric_field not in working_df.columns:
                    raise ValueError(
                        f"Metric field {metric_field!r} "
                        "does not exist in the dataset."
                    )

                default_aggregation = metric_spec.get(
                    "default_aggregation",
                    "sum",
                )

                grouped = working_df.groupby(
                    grouping_columns,
                    dropna=False,
                    sort=True,
                )[metric_field]

                if default_aggregation == "nunique":
                    return grouped.nunique(dropna=True)

                if default_aggregation == "count":
                    return grouped.count()

                numeric_values = pd.to_numeric(
                    working_df[metric_field],
                    errors="coerce",
                )

                numeric_frame = working_df.copy()
                numeric_frame[metric_field] = numeric_values

                numeric_grouped = numeric_frame.groupby(
                    grouping_columns,
                    dropna=False,
                    sort=True,
                )[metric_field]

                if default_aggregation == "mean":
                    return numeric_grouped.mean()

                return numeric_grouped.sum(
                    min_count=1
                )

            try:
                first_values = grouped_metric_series(
                    first_spec
                )
                second_values = grouped_metric_series(
                    second_spec
                )
            except ValueError as error:
                return {
                    "result": None,
                    "insights": None,
                    "chart_spec": None,
                    "chart_data": None,
                    "warnings": [str(error)],
                }


            correlation_frame = pd.concat(
                {
                    first_key: first_values,
                    second_key: second_values,
                },
                axis=1,
            ).dropna().reset_index()

            correlation_frame = correlation_frame.rename(
                columns=output_group_names
            )

        else:
            if (
                first_field not in working_df.columns
                or second_field not in working_df.columns
            ):
                return {
                    "result": None,
                    "insights": None,
                    "chart_spec": None,
                    "chart_data": None,
                    "warnings": [
                        "One or both correlation fields are missing."
                    ],
                }

            correlation_frame = working_df[
                [first_field, second_field]
            ].copy()

            correlation_frame[first_key] = pd.to_numeric(
                correlation_frame[first_field],
                errors="coerce",
            )

            correlation_frame[second_key] = pd.to_numeric(
                correlation_frame[second_field],
                errors="coerce",
            )

            correlation_frame = correlation_frame[
                [first_key, second_key]
            ].dropna()

        if len(correlation_frame) < 2:
            return {
                "result": None,
                "insights": None,
                "chart_spec": None,
                "chart_data": None,
                "warnings": [
                    "At least two aligned observations are "
                    "required for correlation."
                ],
            }

        correlation_value = correlation_frame[
            first_key
        ].corr(
            correlation_frame[second_key]
        )

        if pd.isna(correlation_value):
            return {
                "result": None,
                "insights": None,
                "chart_spec": None,
                "chart_data": None,
                "warnings": [
                    "Correlation could not be calculated because "
                    "one selected series has no variation."
                ],
            }

        absolute_value = abs(float(correlation_value))

        if absolute_value >= 0.7:
            strength = "strong"
        elif absolute_value >= 0.3:
            strength = "moderate"
        else:
            strength = "weak"

        if correlation_value > 0:
            direction = "positive"
        elif correlation_value < 0:
            direction = "negative"
        else:
            direction = "neutral"

        max_chart_points = 1000
        chart_frame = correlation_frame

        if len(correlation_frame) > max_chart_points:
            chart_frame = correlation_frame.sample(
                n=max_chart_points,
                random_state=42,
            )

            warnings.append(
                f"Scatter chart sampled to "
                f"{max_chart_points:,} points from "
                f"{len(correlation_frame):,} observations."
            )

        return {
            "result": {
                "operation": "correlation",
                "group_by": requested_groups,
                "results": {
                    "correlation": round(
                        float(correlation_value),
                        4,
                    ),
                    "first_metric": first_key,
                    "second_metric": second_key,
                    "observations": len(correlation_frame),
                },
            },
            "insights": {
                "summary": (
                    f"{first_label} and {second_label} have a "
                    f"{strength} {direction} linear relationship "
                    f"(correlation {correlation_value:.3f}) across "
                    f"{len(correlation_frame):,} observations. "
                    "Correlation does not prove causation."
                )
            },
            "chart_spec": {
                "chart_type": "scatter",
                "title": f"{first_label} vs {second_label}",
                "x_key": first_key,
                "y_key": second_key,
                "legend": False,
            },
            "chart_data": chart_frame.to_dict("records"),
            "warnings": warnings,
        }
    #
    # Supports semantic plans such as:
    #   group_by = ["user_id"]
    #   group_by = ["user_id"], granularity = "month"
    #   group_by = ["state"]
    #
    # The semantic planner decides WHAT to group by.
    # The executor only performs the requested aggregation.
    # ------------------------------------------------------------------

    if operation == "distinct_count" and not group_by:
        df = (
            data.copy()
            if isinstance(data, pd.DataFrame)
            else pd.DataFrame(data)
        )

        distinct_field = resolved.get("field")

        if not distinct_field and metric_keys:
            distinct_field = config["metrics"][
                metric_keys[0]
            ]["field"]

        if not distinct_field:
            return {
                "result": None,
                "insights": None,
                "chart_spec": None,
                "chart_data": None,
                "warnings": [
                    "A field is required for distinct counting."
                ],
            }

        if distinct_field not in df.columns:
            return {
                "result": None,
                "insights": None,
                "chart_spec": None,
                "chart_data": None,
                "warnings": [
                    f"Distinct-count field {distinct_field!r} "
                    "does not exist in the dataset."
                ],
            }

        distinct_value = int(
            df[distinct_field].dropna().nunique()
        )

        result_key = (
            metric_keys[0]
            if metric_keys
            else f"{distinct_field}_distinct_count"
        )

        return {
            "result": {
                "operation": "distinct_count",
                "field": distinct_field,
                "results": {
                    result_key: distinct_value,
                },
            },
            "insights": {
                "summary": (
                    f"There were {distinct_value:,} unique "
                    f"{distinct_field.replace('_', ' ')} values."
                )
            },
            "chart_spec": None,
            "chart_data": None,
            "warnings": warnings,
        }

    if group_by and operation not in {"top_users", "bottom_users"}:
        if not metric_keys:
            return {
                "result": None,
                "insights": None,
                "chart_spec": None,
                "chart_data": None,
                "warnings": [
                    "A metric is required for grouped analysis."
                ],
            }

        df = (
            data.copy()
            if isinstance(data, pd.DataFrame)
            else pd.DataFrame(data)
        )

        if df.empty:
            return {
                "result": {
                    "operation": operation,
                    "group_by": group_by,
                    "results": [],
                },
                "insights": {
                    "summary": "No rows were available for grouped analysis."
                },
                "chart_spec": None,
                "chart_data": [],
                "warnings": warnings,
            }

        # --------------------------------------------------------------
        # Validate grouping columns
        # --------------------------------------------------------------

        physical_group_fields = [
            field
            for field in group_by
            if field not in time_group_fields
        ]

        missing_group_fields = [
            field
            for field in physical_group_fields
            if field not in df.columns
        ]

        if missing_group_fields:
            return {
                "result": None,
                "insights": None,
                "chart_spec": None,
                "chart_data": None,
                "warnings": [
                    "Unknown grouping field(s): "
                    + ", ".join(missing_group_fields)
                ],
            }

        # --------------------------------------------------------------
        # Add time grouping when semantic planner supplied granularity.
        #
        # Example:
        #   group_by = ["user_id"]
        #   granularity = "month"
        #
        # becomes:
        #   user_id + __time_group__
        # --------------------------------------------------------------

        requested_time_groups = [
            field
            for field in group_by
            if field in time_group_fields
        ]

        granularity = resolved.get("granularity")

        if not granularity and requested_time_groups:
            granularity = requested_time_groups[0]

        working_df = df.copy()
        grouping_fields = list(physical_group_fields)

        if granularity:
            if not date_column or date_column not in working_df.columns:
                return {
                    "result": None,
                    "insights": None,
                    "chart_spec": None,
                    "chart_data": None,
                    "warnings": [
                        "A valid date column is required for time grouping."
                    ],
                }

            parsed_dates = pd.to_datetime(
                working_df[date_column],
                errors="coerce",
            )

            if granularity == "day":
                working_df["__time_group__"] = (
                    parsed_dates.dt.strftime("%Y-%m-%d")
                )

            elif granularity == "week":
                working_df["__time_group__"] = (
                    parsed_dates.dt.to_period("W")
                    .astype(str)
                )

            elif granularity == "month":
                working_df["__time_group__"] = (
                    parsed_dates.dt.to_period("M")
                    .astype(str)
                )

            elif granularity == "quarter":
                working_df["__time_group__"] = (
                    parsed_dates.dt.to_period("Q")
                    .astype(str)
                )

            elif granularity == "year":
                working_df["__time_group__"] = (
                    parsed_dates.dt.year
                )

            grouping_fields.append("__time_group__")

        # --------------------------------------------------------------
        # Execute every requested metric.
        #
        # This intentionally works from metric specs rather than assuming
        # that every metric is a simple sum.
        # --------------------------------------------------------------

        grouped_results = []

        grouped = working_df.groupby(
            grouping_fields,
            dropna=False,
            sort=True,
        )

        for group_values, group_df in grouped:

            if not isinstance(group_values, tuple):
                group_values = (group_values,)

            row = {}

            for field_name, field_value in zip(
                grouping_fields,
                group_values,
            ):
                if field_name == "__time_group__":
                    output_field = granularity
                else:
                    output_field = field_name

                row[output_field] = (
                    None
                    if pd.isna(field_value)
                    else field_value
                )

            for metric_key in metric_keys:
                metric_spec = config["metrics"][metric_key]
                metric_field = metric_spec["field"]

                if metric_field not in group_df.columns:
                    row[metric_key] = None
                    continue

                default_aggregation = metric_spec.get(
                    "default_aggregation",
                    "sum",
                )

                if (
                    operation == "distinct_count"
                    or default_aggregation == "nunique"
                ):
                    value = int(
                        group_df[metric_field]
                        .dropna()
                        .nunique()
                    )

                elif default_aggregation == "count":
                    value = int(
                        group_df[metric_field]
                        .notna()
                        .sum()
                    )

                else:
                    values = pd.to_numeric(
                        group_df[metric_field],
                        errors="coerce",
                    ).dropna()

                    if operation == "average":
                        value = (
                            float(values.mean())
                            if not values.empty
                            else None
                        )

                    elif operation == "maximum":
                        value = (
                            float(values.max())
                            if not values.empty
                            else None
                        )

                    elif operation == "minimum":
                        value = (
                            float(values.min())
                            if not values.empty
                            else None
                        )

                    else:
                        value = (
                            float(values.sum())
                            if not values.empty
                            else None
                        )
                if (
                    value is not None
                    and isinstance(value, float)
                    and value.is_integer()
                ):
                    value = int(value)

                row[metric_key] = value

            grouped_results.append(row)

        # --------------------------------------------------------------
        # Frontend-friendly chart data
        # --------------------------------------------------------------

        chart_data = grouped_results

        result = {
            "operation": operation,
            "group_by": group_by,
            "granularity": granularity,
            "results": grouped_results,
        }

        # --------------------------------------------------------------
        # Basic insight
        # --------------------------------------------------------------

        insights = {
            "summary": (
                f"Grouped {', '.join(group_by)}"
                + (
                    f" by {granularity}"
                    if granularity
                    else ""
                )
                + "."
            )
        }

        # --------------------------------------------------------------
        # Choose a sensible chart.
        #
        # Single dimension → bar
        # Time grouping → line
        # Multi-dimensional → table-compatible data
        # --------------------------------------------------------------

        if granularity:
            is_multi_metric = len(metric_keys) > 1

            chart_spec = {
                "chart_type": (
                    "multi_line"
                    if is_multi_metric
                    else "line"
                ),
                "title": (
                    " vs ".join(
                        metric_display_map[metric_key]
                        for metric_key in metric_keys
                    )
                    + f" by {granularity}"
                ),
                "x_key": granularity,
                "y_key": metric_keys[0],
                "y_keys": metric_keys,
                "legend": is_multi_metric,
            }

        elif len(group_by) == 1:
            chart_spec = {
                "chart_type": "bar",
                "title": (
                    f"{metric_display_map[metric_keys[0]]} "
                    f"by {group_by[0].replace('_', ' ').title()}"
                ),
                "x_key": group_by[0],
                "y_key": metric_keys[0],
                "legend": False,
            }

        else:
            chart_spec = None

        return {
            "result": result,
            "insights": insights,
            "chart_spec": chart_spec,
            "chart_data": chart_data,
            "warnings": warnings,
        }


    if operation in ("top_users", "bottom_users"):
        if not metric_keys:
            return {
                "result": None,
                "insights": None,
                "chart_spec": None,
                "chart_data": None,
                "warnings": [
                    "A metric is required for a top or bottom ranking."
                ],
            }

        metric_key = metric_keys[0]
        metric_spec = config["metrics"][metric_key]
        metric_field = metric_spec["field"]
        metric_label = metric_display_map[metric_key]

        ranking_levels = resolved.get("ranking_levels", [])

        # ---------------------------------------------------------
        # Nested ranking
        #
        # Example:
        # Top 5 users inside the top 5 schemes by invoice value.
        #
        # ranking_levels[0] is the outer ranking.
        # ranking_levels[1] is the inner ranking.
        # ---------------------------------------------------------

        if len(ranking_levels) == 2:
            outer_level = ranking_levels[0]
            inner_level = ranking_levels[1]

            outer_dimension = outer_level["dimension"]
            inner_dimension = inner_level["dimension"]
            outer_top_n = int(outer_level["top_n"])
            inner_top_n = int(inner_level["top_n"])

            df = (
                data.copy()
                if isinstance(data, pd.DataFrame)
                else pd.DataFrame(data)
            )

            required_columns = {
                outer_dimension,
                inner_dimension,
                metric_field,
            }

            missing_columns = required_columns.difference(df.columns)

            if missing_columns:
                return {
                    "result": None,
                    "insights": None,
                    "chart_spec": None,
                    "chart_data": None,
                    "warnings": [
                        "Nested ranking could not be executed because "
                        f"required columns were missing: "
                        f"{sorted(missing_columns)}"
                    ],
                }

            reverse_sort = operation == "top_users"

            outer_values = _group_metric(
                data=df,
                metric_spec=metric_spec,
                dimension=outer_dimension,
                operation="total",
            )

            ordered_outer_values = sorted(
                outer_values.items(),
                key=lambda item: item[1],
                reverse=reverse_sort,
            )[:outer_top_n]

            nested_rows = []

            for outer_rank, (outer_entity, outer_value) in enumerate(
                ordered_outer_values,
                start=1,
            ):
                outer_frame = df[
                    df[outer_dimension].astype(str)
                    == str(outer_entity)
                ].copy()

                inner_values = _group_metric(
                    data=outer_frame,
                    metric_spec=metric_spec,
                    dimension=inner_dimension,
                    operation="total",
                )

                ordered_inner_values = sorted(
                    inner_values.items(),
                    key=lambda item: item[1],
                    reverse=reverse_sort,
                )[:inner_top_n]

                children = [
                    {
                        "rank": inner_rank,
                        "dimension": inner_dimension,
                        "entity": inner_entity,
                        "value": inner_value,
                        inner_dimension: inner_entity,
                        metric_field: inner_value,
                    }
                    for inner_rank, (
                        inner_entity,
                        inner_value,
                    ) in enumerate(
                        ordered_inner_values,
                        start=1,
                    )
                ]

                nested_rows.append(
                    {
                        "rank": outer_rank,
                        "dimension": outer_dimension,
                        "entity": outer_entity,
                        "value": outer_value,
                        outer_dimension: outer_entity,
                        metric_field: outer_value,
                        "children": children,
                    }
                )

            direction_label = (
                "Top"
                if operation == "top_users"
                else "Bottom"
            )

            result = {
                "operation": "nested_ranking",
                "direction": (
                    "top"
                    if operation == "top_users"
                    else "bottom"
                ),
                "metric": metric_key,
                "ranking_levels": ranking_levels,
                "results": nested_rows,
            }

            if nested_rows:
                leader = nested_rows[0]

                insights = {
                    "summary": (
                        f"{leader['entity']} ranks first among the "
                        f"{direction_label.lower()} {outer_top_n} "
                        f"{outer_dimension.replace('_', ' ')} values by "
                        f"{metric_label}, with "
                        f"{leader['value']:,.2f}. Each selected "
                        f"{outer_dimension.replace('_', ' ')} includes its "
                        f"{direction_label.lower()} {inner_top_n} "
                        f"{inner_dimension.replace('_', ' ')} values."
                    )
                }
            else:
                insights = {
                    "summary": (
                        "No rows were available for the requested "
                        "nested ranking."
                    )
                }

            # Nested visualization will be implemented in the next
            # frontend phase. Do not flatten this into a misleading chart.
            return {
                "result": result,
                "insights": insights,
                "chart_spec": None,
                "chart_data": None,
                "warnings": warnings,
            }
        ranking_dimension = _resolve_ranking_dimension(
            data=data,
            config=config,
            resolved=resolved,
        )

        if ranking_dimension is None:
            return {
                "result": None,
                "insights": None,
                "chart_spec": None,
                "chart_data": None,
                "warnings": [
                    "Could not identify which dataset dimension should be ranked."
                ],
            }

        top_n = int(resolved.get("top_n") or 5)

        grouped_values = _group_metric(
            data=data,
            metric_spec=metric_spec,
            dimension=ranking_dimension,
            operation="total",
        )

        reverse_sort = operation == "top_users"

        ordered_values = sorted(
            grouped_values.items(),
            key=lambda item: item[1],
            reverse=reverse_sort,
        )[:top_n]

        ranking_rows = []

        for entity, value in ordered_values:
            row = {
                "dimension": entity,
                "value": value,
                ranking_dimension: entity,
                metric_field: value,
            }

            # Preserve compatibility with the current frontend while also
            # returning the real dynamic field names.
            if ranking_dimension == "user_id":
                row["user_id"] = entity

            if metric_field == "invoice_value":
                row["invoice_value"] = value

            ranking_rows.append(row)

        result = {
            "operation": operation,
            "dimension": ranking_dimension,
            "metric": metric_key,
            "results": {
                metric_key: ranking_rows,
            },
        }

        chart_data = [
            {
                "dimension": row["dimension"],
                "value": row["value"],
            }
            for row in ranking_rows
        ]

        direction_label = (
            "Top"
            if operation == "top_users"
            else "Bottom"
        )

        chart_spec = {
            "chart_type": "bar",
            "title": (
                f"{direction_label} {top_n} "
                f"{ranking_dimension.replace('_', ' ').title()} "
                f"by {metric_label}"
            ),
            "x_key": "dimension",
            "y_key": "value",
            "legend": False,
        }

        if ranking_rows:
            leader = ranking_rows[0]
            insights = {
                "summary": (
                    f"{leader['dimension']} ranks first in this "
                    f"{direction_label.lower()} {top_n} ranking by "
                    f"{metric_label}, with {leader['value']:,.2f}."
                )
            }
        else:
            insights = {
                "summary": (
                    "No rows were available for the requested ranking."
                )
            }

        return {
            "result": result,
            "insights": insights,
            "chart_spec": chart_spec,
            "chart_data": chart_data,
            "warnings": warnings,
        }

    if operation == "unique_users":
        entity_dimension = _resolve_ranking_dimension(
            data=data,
            config=config,
            resolved=resolved,
        )

        if entity_dimension is None:
            return {
                "result": None,
                "insights": None,
                "chart_spec": None,
                "chart_data": None,
                "warnings": [
                    "Could not identify the entity dimension to count."
                ],
            }

        df = (
            data.copy()
            if isinstance(data, pd.DataFrame)
            else pd.DataFrame(data)
        )

        unique_count = int(
            df[entity_dimension].dropna().nunique()
        )

        result = {
            "operation": "unique_users",
            "dimension": entity_dimension,
            "results": {
                entity_dimension: unique_count,
            },
        }

        insights = {
            "summary": (
                f"The dataset contains {unique_count:,} unique "
                f"{entity_dimension.replace('_', ' ')} values."
            )
        }

        return {
            "result": result,
            "insights": insights,
            "chart_spec": None,
            "chart_data": None,
            "warnings": warnings,
        }

    if operation == "total" and len(metric_keys) == 1 and not dimension:
        metric_key = metric_keys[0]
        metric_spec = config["metrics"][metric_key]
        value = _aggregate_metric(data, metric_spec, operation="total")
        result = {"operation": "total", "results": {metric_key: value}}
        label = metric_display_map[metric_key]
        insights = {"summary": f"Total {label} was {value:,.2f}." if value is not None
                                else f"No usable {label} data was found."}

    elif operation == "average" and len(metric_keys) == 1 and not dimension:
        metric_key = metric_keys[0]
        metric_spec = config["metrics"][metric_key]
        value = _aggregate_metric(data, metric_spec, operation="average")
        result = {"operation": "average", "results": {metric_key: value}}
        label = metric_display_map[metric_key]
        insights = {"summary": f"Average {label} was {value:,.2f}." if value is not None
                                else f"No usable {label} data was found."}

    elif operation == "maximum" and len(metric_keys) == 1 and not dimension:
        metric_key = metric_keys[0]
        metric_spec = config["metrics"][metric_key]
        field = metric_spec["field"]
        record = highest_metric(data, field)
        value = _aggregate_metric(data, metric_spec, operation="maximum")
        period = record.get(date_column) if record and date_column else None
        result = {"operation": "maximum", "results": {metric_key: value}, "record": record}
        label = metric_display_map[metric_key]
        insights = {"summary": (f"The highest {label} was {value:,.2f}" + (f", on {period}." if period else ".")
                                 if value is not None else f"No usable {label} data was found.")}

    elif operation == "minimum" and len(metric_keys) == 1 and not dimension:
        metric_key = metric_keys[0]
        metric_spec = config["metrics"][metric_key]
        field = metric_spec["field"]
        record = lowest_metric(data, field)
        value = _aggregate_metric(data, metric_spec, operation="minimum")
        period = record.get(date_column) if record and date_column else None
        result = {"operation": "minimum", "results": {metric_key: value}, "record": record}
        label = metric_display_map[metric_key]
        insights = {"summary": (f"The lowest {label} was {value:,.2f}" + (f", on {period}." if period else ".")
                                 if value is not None else f"No usable {label} data was found.")}

    elif operation in ("total", "average", "maximum", "minimum") and dimension and not is_time_dimension:
        # breakdown by a discovered categorical dimension, e.g. "by region"
        metric_key = metric_keys[0]
        metric_spec = config["metrics"][metric_key]
        groups = _group_metric(data, metric_spec, dimension, operation)
        result = {"operation": operation, "dimension": dimension, "results": groups}
        label = metric_display_map[metric_keys[0]]
        if groups:
            leader = max(groups.items(), key=lambda kv: kv[1])
            insights = {"summary": f"{leader[0]} leads in {label} by {dimension} with {leader[1]:,.2f}."}
        chart_data = [{"dimension": k, "value": v} for k, v in groups.items()]
        chart_spec = {"chart_type": "bar", "title": f"{label} by {dimension.capitalize()}",
                      "x_key": "dimension", "y_key": "value", "legend": False}

    elif operation in ("total", "average", "maximum", "minimum") and is_time_dimension:
        # "total invoice value by month" / "by week" / "by quarter" / "by year"
        if not date_column:
            warnings.append("no reliable date column detected -- cannot group by time")
        else:
            metric_key = metric_keys[0]
            metric_spec = config["metrics"][metric_key]
            series = _group_by_time(data, metric_spec, date_column, dimension, operation)
            result = {"operation": operation, "dimension": dimension, "results": series}
            label = metric_display_map[metric_key]
            if series:
                leader = max(series.items(), key=lambda kv: kv[1])
                insights = {"summary": f"{label} peaked in {leader[0]} at {leader[1]:,.2f}."}
            chart_data = [{"period": k, "value": v} for k, v in series.items()]
            chart_spec = {"chart_type": "line", "title": f"{label} by {dimension.capitalize()}",
                          "x_key": "period", "y_key": "value", "legend": False}

    elif operation == "trend":
        gran = dimension if dimension in ("day", "week", "month", "quarter", "year") else "month"
        if not date_column:
            warnings.append("no reliable date column detected -- cannot show a trend over time")
        else:
            metric_key = metric_keys[0]
            metric_spec = config["metrics"][metric_key]
            series = _group_by_time(data, metric_spec, date_column, gran, "total")
            result = {"operation": "trend", "granularity": gran, "results": series}
            label = metric_display_map[metric_key]
            if len(series) >= 2:
                values = list(series.values())
                change_pct = ((values[-1] - values[0]) / abs(values[0]) * 100) if values[0] else None
                direction = "increased" if (change_pct or 0) > 0 else "decreased" if (change_pct or 0) < 0 else "stayed flat"
                insights = {"summary": f"{label} {direction} over the observed period"
                                        + (f" ({change_pct:+.1f}%)." if change_pct is not None else ".")}
            chart_data = [{"period": k, "value": v} for k, v in series.items()]
            chart_spec = {"chart_type": "line", "title": f"{label} Over Time",
                          "x_key": "period", "y_key": "value", "legend": False}

    elif operation == "compare":
        gran = dimension if dimension in ("day", "week", "month", "quarter", "year") else "month"
        if not date_column:
            warnings.append("no reliable date column detected -- comparing without a time axis")
            result = {
                "operation": "compare",
                "results": {
                    k: _aggregate_metric(data, config["metrics"][k], operation="total")
                    for k in metric_keys
                },
            }
        else:
            per_metric_series = {
                k: _group_by_time(data, config["metrics"][k], date_column, gran, "total")
                for k in metric_keys
            }
            result = {"operation": "compare", "granularity": gran, "results": per_metric_series}
            all_periods = sorted({p for s in per_metric_series.values() for p in s})
            chart_data = [
                {"period": p, **{metric_display_map[k]: per_metric_series[k].get(p) for k in metric_keys}}
                for p in all_periods
            ]
            labels = " vs ".join(metric_display_map[k] for k in metric_keys)
            insights = {"summary": f"Comparing {labels} over time by {gran}."}
            chart_spec = decide_chart(
                {"query": resolved["query"], "metrics": metric_keys,
                 "operations": ["compare"], "dimension": gran},
                metric_field_map=metric_fields_map, metric_display_map=metric_display_map,
            ).get("chart_spec")

    else:
        warnings.append(f"operation {operation!r} with {len(metric_keys)} metric(s) and "
                        f"dimension={dimension!r} isn't a recognized combination for this endpoint")

    return {"result": result, "insights": insights, "chart_spec": chart_spec,
            "chart_data": chart_data, "warnings": warnings}


# ---------------------------------------------------------------------------
# Public entry point (called by app.py)
# ---------------------------------------------------------------------------

def analyze_dynamic(
    data,
    query: str,
    dataset_id: str | None = None,
    source_path: str | None = None,
    metric_override: str | None = None,
    granularity_override: str | None = None,
    horizon_override: int | None = None,
    use_llm_fallback: bool = False,
    _plan_override: dict | None = None,
    _recovery_attempted: bool = False,
) -> dict:
    unfiltered_data = data
    config = build_runtime_config(data, dataset_id=dataset_id, source_path=source_path)

    dataset_meta = {
        "dataset_id": dataset_id,
        "date_column": config.get("date_column"),
        "fingerprint": config["dataset_fingerprint"],
    }

    detected_nested_ranking = (
        _extract_nested_ranking_request(
            query=query,
            data=data,
            config=config,
        )
    )

    

    # Do NOT fail here when the deterministic nested-ranking parser
    # cannot understand the wording.
    #
    # The semantic planner is allowed to resolve natural-language
    # nested ranking using the runtime schema.

    resolved = None
    
    if _plan_override is not None:
        resolved = resolve_semantic_request(
            query=query,
            data=data,
            config=config,
            plan_override=_plan_override,
        )

    elif metric_override is None:
        resolved = resolve_semantic_request(
            query=query,
            data=data,
            config=config,
        )

    if resolved is None:
        resolved = resolve_request(
            query,
            data,
            config,
            metric_override,
            granularity_override,
            horizon_override,
        )
        print("DEBUG FINAL RESOLVED FILTERS:", resolved.get("filters"))

    if resolved.get("error"):
        original_error = resolved["error"]

        recovered = None

        if use_llm_fallback and not _recovery_attempted:
            recovered = build_recovery_plan(
                query=query,
                data=data,
                config=config,
                failed_plan=resolved,
                error_message=original_error,
            )

        if recovered is not None:
            canonical_recovery = resolve_semantic_request(
                query=query,
                data=data,
                config=config,
                plan_override=recovered,
            )

            if canonical_recovery is not None:
                logger.info(
                    "Recovered failed semantic plan for query %r.",
                    query,
                )
                resolved = canonical_recovery

            else:
                recovered = None

        if recovered is None:
            return build_response(
                request={
                    "query": query,
                    "operation": resolved.get("operation"),
                    "dimension": resolved.get("dimension"),
                },
                errors=[original_error],
                status="error",
                dataset=dataset_meta,
                warnings=[],
            )

        # Apply filters discovered by the semantic planner before any
    # aggregation, ranking, trend, relationship, or LLM analysis.
    resolved_filters = resolved.get("filters", {})

    filter_logic = resolved.get(
        "filter_logic",
        "and",
    )

    if resolved_filters:
        data = apply_generic_filters(
                data,
                resolved_filters,
                config,
                filter_logic=filter_logic,
            )
        print("DEBUG FILTERED ROW COUNT:", len(data))

        if len(data) == 0:
            return build_response(
                request=resolved,
                errors=[
                    "The requested filters matched no rows in the dataset."
                ],
                status="error",
                dataset=dataset_meta,
                warnings=[],
            )

    if resolved["requires_forecast"]:
        prediction = run_dynamic_forecast(
            data, query=query, dataset_id=dataset_id, source_path=source_path,
            metric_key_override=metric_override or (resolved["metric_keys"][0] if resolved["metric_keys"] else None),
            granularity_override=granularity_override, horizon_override=horizon_override,
            enable_llm_fallback=use_llm_fallback,
        )
        status = "success" if prediction.get("status") == "success" else prediction.get("status", "error")
        return build_response(
            request=resolved,
            prediction=prediction,
            errors=[] if status == "success" else [prediction.get("message", "forecast failed")],
            status=status,
            dataset=dataset_meta,
            warnings=[],
        )
    if resolved.get("requires_llm"):
        try:
            df = (
                data.copy()
                if isinstance(data, pd.DataFrame)
                else pd.DataFrame(data)
            )

            metric_keys = resolved.get("metric_keys", [])

                        # General LLM questions may not mention a specific metric.
            # Send the filtered dataset directly to Claude instead of
            # incorrectly forcing the request into a metric trend.
            if not metric_keys:
                llm_result = analyze_data(
                    data,
                    query,
                    reason="general dataset analysis",
                )

                return build_response(
                    request=resolved,
                    explanation=llm_result,
                    status="success",
                    dataset=dataset_meta,
                    warnings=[],
                )

            query_lower = query.lower()

            is_relationship_request = (
                len(metric_keys) >= 2
                and any(
                    phrase in query_lower
                    for phrase in (
                        "relationship",
                        "correlation",
                        "correlate",
                    )
                )
            )

            # Not an explicit relationship/correlation question:
            # send it through the existing Claude analysis path.
                        # General LLM analysis.
            # If the same request also needs deterministic output such as
            # a chart, run both paths and merge them into one response.
            if not is_relationship_request:
                analysis_resolved = dict(resolved)

                supported_operations = {
                    "total",
                    "average",
                    "maximum",
                    "minimum",
                    "trend",
                    "compare",
                    "top_users",
                    "bottom_users",
                }

                # If the request is primarily an explanation or
                # recommendation, choose the most useful deterministic
                # calculation to ground Claude's response.
                if analysis_resolved["operation"] not in supported_operations:
                    if analysis_resolved.get("dimension"):
                        # Group the selected metric by the requested entity,
                        # region, product, or other dimension.
                        analysis_resolved["operation"] = "total"
                    elif len(metric_keys) >= 2:
                        analysis_resolved["operation"] = "compare"
                    else:
                        analysis_resolved["operation"] = "trend"

                analysis_execution = execute_request(
                    data,
                    config,
                    analysis_resolved,
                )

                if analysis_execution["result"] is None:
                    return build_response(
                        request=resolved,
                        errors=[
                            "Could not prepare deterministic data for "
                            "the requested LLM analysis."
                        ] + analysis_execution["warnings"],
                        status="error",
                        dataset=dataset_meta,
                        warnings=analysis_execution["warnings"],
                    )

                llm_input = analysis_execution.get("chart_data")

                # Some deterministic operations return a result without
                # chart data. Claude can still reason over that calculated
                # result.
                if not llm_input:
                    llm_input = [
                        {
                            "calculated_result": analysis_execution["result"]
                        }
                    ]

                llm_result = analyze_data(
                    llm_input,
                    query,
                    reason=(
                        "Explain and interpret the deterministic "
                        "calculation prepared by the backend."
                    ),
                )

                return build_response(
                    request=resolved,
                    result=analysis_execution["result"],
                    insights=analysis_execution["insights"],
                    chart_spec=(
                        analysis_execution["chart_spec"]
                        if resolved.get("requires_chart")
                        else None
                    ),
                    chart_data=(
                        analysis_execution["chart_data"]
                        if resolved.get("requires_chart")
                        else None
                    ),
                    explanation=llm_result,
                    status="success",
                    dataset=dataset_meta,
                    warnings=analysis_execution["warnings"],
                )

            if len(metric_keys) < 2:
                return build_response(
                    request=resolved,
                    errors=[
                        "At least two metrics are required to explain a relationship."
                    ],
                    status="error",
                    dataset=dataset_meta,
                    warnings=[],
                )

            first_metric_key = metric_keys[0]
            second_metric_key = metric_keys[1]

            first_field = config["metrics"][first_metric_key]["field"]
            second_field = config["metrics"][second_metric_key]["field"]

            relationship_data = df[
                [first_field, second_field]
            ].copy()

            relationship_data[first_field] = pd.to_numeric(
                relationship_data[first_field],
                errors="coerce",
            )

            relationship_data[second_field] = pd.to_numeric(
                relationship_data[second_field],
                errors="coerce",
            )

            relationship_data = relationship_data.dropna()

            if relationship_data.empty:
                return build_response(
                    request=resolved,
                    errors=[
                        "There is not enough numeric data to analyse this relationship."
                    ],
                    status="error",
                    dataset=dataset_meta,
                    warnings=[],
                )

            correlation = relationship_data[first_field].corr(
                relationship_data[second_field]
            )

            if pd.isna(correlation):
                relationship = "could not be determined"
            elif correlation >= 0.7:
                relationship = "a strong positive relationship"
            elif correlation >= 0.3:
                relationship = "a moderate positive relationship"
            elif correlation > 0:
                relationship = "a weak positive relationship"
            elif correlation <= -0.7:
                relationship = "a strong negative relationship"
            elif correlation <= -0.3:
                relationship = "a moderate negative relationship"
            elif correlation < 0:
                relationship = "a weak negative relationship"
            else:
                relationship = "little or no linear relationship"

            first_name = config["metrics"][first_metric_key].get(
                "display_name",
                first_metric_key,
            )

            second_name = config["metrics"][second_metric_key].get(
                "display_name",
                second_metric_key,
            )

            summary = (
                f"{first_name} and {second_name} have "
                f"{relationship}. "
                f"The correlation coefficient is {correlation:.3f}. "
            )

            if correlation > 0:
                summary += (
                    f"As {second_name} increases, {first_name} generally "
                    "tends to increase as well. However, correlation alone "
                    "does not prove that one metric causes the other."
                )
            elif correlation < 0:
                summary += (
                    f"As {second_name} increases, {first_name} generally "
                    "tends to decrease. However, correlation alone does "
                    "not prove causation."
                )
            else:
                summary += (
                    "Changes in one metric do not appear to consistently "
                    "correspond with changes in the other."
                )

            return build_response(
                request=resolved,
                result={
                    "operation": "relationship",
                    "results": {
                        "correlation": round(float(correlation), 4),
                        "first_metric": first_metric_key,
                        "second_metric": second_metric_key,
                        "rows_analysed": len(relationship_data),
                    },
                },
                insights={
                    "summary": summary,
                },
                status="success",
                dataset=dataset_meta,
                warnings=[],
            )

        except Exception as error:
            logger.exception("LLM analysis failed")

            return build_response(
                request=resolved,
                errors=[f"LLM analysis failed: {error}"],
                status="error",
                dataset=dataset_meta,
                warnings=[],
            )
    execution = execute_request(data, config, resolved)

    if execution["result"] is None:
        original_errors = [
            (
                "could not execute operation "
                f"{resolved['operation']!r} for this request"
            )
        ] + execution["warnings"]

        if use_llm_fallback and not _recovery_attempted:
            recovered = build_recovery_plan(
                query=query,
                data=unfiltered_data,
                config=config,
                failed_plan=resolved,
                error_message="; ".join(original_errors),
            )

            if recovered is not None:
                recovery_response = analyze_dynamic(
                    data=unfiltered_data,
                    query=query,
                    dataset_id=dataset_id,
                    source_path=source_path,
                    metric_override=None,
                    granularity_override=granularity_override,
                    horizon_override=horizon_override,
                    use_llm_fallback=False,
                    _plan_override=recovered,
                    _recovery_attempted=True,
                )

                if recovery_response.get("status") == "success":
                    return recovery_response

        return build_response(
            request=resolved,
            errors=original_errors,
            status="error",
            dataset=dataset_meta,
            warnings=execution["warnings"],
        )

    return build_response(
        request=resolved,
        result=execution["result"],
        insights=execution["insights"],
        chart_spec=execution["chart_spec"] if resolved["requires_chart"] else None,
        chart_data=execution["chart_data"] if resolved["requires_chart"] else None,
        status="success",
        dataset=dataset_meta,
        warnings=execution["warnings"],
    )
