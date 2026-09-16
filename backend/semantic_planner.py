import json
from logging import config
import re
import logging

import pandas as pd

logger = logging.getLogger(__name__)

ALLOWED_OPERATIONS = {
    "total",
    "average",
    "maximum",
    "minimum",
    "trend",
    "compare",
    "forecast",
    "top_users",
    "distinct_count",
    "bottom_users",
    "unique_users",
    "correlation",
    "llm_analysis",
}
ALLOWED_FILTER_OPERATORS = {
    "equals",
    "not_equals",
    "in",
    "not_in",
    "contains",
    "greater_than",
    "greater_than_or_equal",
    "less_than",
    "less_than_or_equal",
    "between",
    "date_from",
    "date_to",
}

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _metric_context(config: dict) -> list[dict]:
    """
    Give Claude only the useful parts of each dynamically discovered metric.
    """
    result = []

    for metric_key, spec in config.get("metrics", {}).items():
        result.append(
            {
                "metric_key": metric_key,
                "field": spec.get("field"),
                "display_name": spec.get("display_name"),
                "aliases": spec.get("aliases", []),
                "default_aggregation": spec.get(
                    "default_aggregation",
                    "sum",
                ),
                "forecastable": spec.get("forecastable", False),
            }
        )

    return result


def _dimension_context(
    data,
    config: dict,
    query: str,
) -> list[dict]:
    """
    Build compact categorical context dynamically from the current dataset.

    Exact values appearing in the question are always included. A small
    sample is also supplied so Claude can understand what each dimension
    represents without sending the entire dataset.
    """
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    query_norm = _normalize(query)

    result = []

    dimension_fields = list(config.get("dimensions", []))

    # Dynamically include categorical, boolean, and identifier fields
    # from the current dataset as valid analytical dimensions.
    column_profiles = config.get("column_profiles", {})

    for field in df.columns:
        if field in dimension_fields:
            continue

        profile = column_profiles.get(field, {})
        semantic_type = (
            profile.get("semantic_type")
            if isinstance(profile, dict)
            else None
        )

        if semantic_type in {
            "categorical",
            "boolean",
            "identifier",
        }:
            dimension_fields.append(field)
            continue

        # Dynamically recognize identifier-style fields regardless of
        # naming convention: user_id, agentID, customerId, campaignID, etc.
        if isinstance(field, str):
            normalized_field = re.sub(
                r"[^a-z0-9]",
                "",
                field.casefold(),
            )

            if normalized_field.endswith("id"):
                dimension_fields.append(field)

    for dimension in dimension_fields:
        if dimension not in df.columns:
            continue

        values = [
            str(value).strip()
            for value in df[dimension].dropna().unique()
            if str(value).strip()
        ]

        mentioned_values = []

        for value in values:
            value_norm = _normalize(value)

            if value_norm and re.search(
                rf"\b{re.escape(value_norm)}\b",
                query_norm,
            ):
                mentioned_values.append(value)

        examples = mentioned_values.copy()

        for value in values[:20]:
            if value not in examples:
                examples.append(value)

        result.append(
            {
                "field": dimension,
                "unique_count": len(values),
                "mentioned_values": mentioned_values,
                "example_values": examples[:25],
            }
        )

    return result


def _extract_json(raw_text: str) -> dict:
    text = raw_text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    start = text.find("{")

    if start == -1:
        raise ValueError("Claude did not return a JSON object.")

    result, _ = json.JSONDecoder().raw_decode(text[start:])

    if not isinstance(result, dict):
        raise ValueError("Semantic plan must be a JSON object.")

    return result

def _enforce_validity_filter(
    plan: dict,
    query: str,
    config: dict,
) -> dict:
    """
    Make 'valid scans' and 'invalid scans' use the dataset's
    discovered validity boolean field.

    A categorical status filter remains allowed when the user
    explicitly mentions the status field.
    """
    if not isinstance(plan, dict):
        return plan

    normalized_query = _normalize(query)

    invalid_request = re.search(
        r"\binvalid\s+scans?\b",
        normalized_query,
    )

    valid_request = re.search(
        r"\bvalid\s+scans?\b",
        normalized_query,
    )

    explicitly_requests_status = re.search(
        r"\bstatus\b",
        normalized_query,
    )

    if (
        explicitly_requests_status
        or not (invalid_request or valid_request)
    ):
        return plan

    requested_value = 0 if invalid_request else 1

    validity_fields = []

    for field, profile in config.get(
        "column_profiles",
        {},
    ).items():
        if profile.get("semantic_type") != "boolean":
            continue

        aliases = profile.get("aliases", [])

        if not isinstance(aliases, list):
            aliases = []

        field_context = _normalize(
            " ".join(
                [
                    field,
                    str(profile.get("display_name", "")),
                    *[str(alias) for alias in aliases],
                ]
            )
        )

        if re.search(
            r"\bvalid\b|\bvalidity\b",
            field_context,
        ):
            validity_fields.append(field)

    normalized_plan = dict(plan)

    if len(validity_fields) != 1:
        unresolved = normalized_plan.get(
            "unresolved_conditions",
            [],
        )

        if isinstance(unresolved, str):
            unresolved = [unresolved]

        if not isinstance(unresolved, list):
            unresolved = []

        unresolved.append(
            "Could not uniquely identify the dataset validity field."
        )

        normalized_plan["unresolved_conditions"] = unresolved
        return normalized_plan

    validity_field = validity_fields[0]
    raw_filters = normalized_plan.get("filters", [])

    if not isinstance(raw_filters, list):
        return normalized_plan

    cleaned_filters = []

    for condition in raw_filters:
        if not isinstance(condition, dict):
            continue

        condition_field = condition.get("field")
        condition_value = _normalize(
            str(condition.get("value", ""))
        )

        if condition_field == validity_field:
            continue

        if condition_value in {"valid", "invalid"}:
            continue

        cleaned_filters.append(condition)

    cleaned_filters.append(
        {
            "field": validity_field,
            "operator": "equals",
            "value": requested_value,
        }
    )

    normalized_plan["filters"] = cleaned_filters
    return normalized_plan

def _validate_plan(plan: dict, config: dict) -> dict:
    """
    Validate and normalize the semantic planner output against the
    current dataset schema.

    The semantic planner may understand natural language, but it must
    never invent metrics, fields, dimensions, filters, or grouping fields.
    """

    available_metrics = set(config.get("metrics", {}).keys())

    available_dimensions = set(
        config.get("dimensions", [])
    )

    # Dynamically allow suitable categorical fields to be used
    # as dimensions/grouping fields.
    for field, profile in config.get("column_profiles", {}).items():
        if not isinstance(profile, dict):
            continue

        semantic_type = profile.get("semantic_type")

        if semantic_type in {
            "categorical",
            "boolean",
            "identifier",
        }:
            available_dimensions.add(field)

    date_column = config.get("date_column")
    validation_errors = []

    # Build the complete set of real fields available in the dataset.
    available_fields = set(
        config.get("column_profiles", {}).keys()
    )

    available_fields.update(
        config.get("dimensions", [])
    )

    for metric_spec in config.get("metrics", {}).values():
        if not isinstance(metric_spec, dict):
            continue

        metric_field = metric_spec.get("field")
        if metric_field:
            available_fields.add(metric_field)

    if date_column:
        available_fields.add(date_column)

    # Real identifier fields can also be used as ranking/grouping
    # dimensions, even when they are not listed in config["dimensions"].

    identifier_dimensions = set()

    for field in available_fields:
        if not isinstance(field, str):
            continue

        normalized_field = re.sub(
            r"[^a-z0-9]",
            "",
            field.casefold(),
        )

        if normalized_field.endswith("id"):
            identifier_dimensions.add(field)

    available_dimensions.update(identifier_dimensions)
    # ---------------------------------------------------------
    # Operation
    # ---------------------------------------------------------
    filter_logic = str(
        plan.get("filter_logic", "and")
    ).lower()

    if filter_logic not in {"and", "or"}:
        validation_errors.append(
            f"Unsupported filter logic {filter_logic!r}."
        )
        filter_logic = "and"

    operation = plan.get(
        "operation",
        "llm_analysis",
    )

    if operation not in ALLOWED_OPERATIONS:
        validation_errors.append(
            f"Unsupported operation {operation!r}."
        )
        operation = "llm_analysis"

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    metric_keys = plan.get(
        "metric_keys",
        [],
    )

    if not isinstance(metric_keys, list):
        validation_errors.append(
            "metric_keys must be a list."
        )
        metric_keys = []

    metric_keys = [
        metric
        for metric in metric_keys
        if metric in available_metrics
    ]

    # ---------------------------------------------------------
    # Distinct-count field
    # ---------------------------------------------------------

    distinct_field = plan.get("field")

    if distinct_field not in available_fields:
        distinct_field = None

    # ---------------------------------------------------------
    # Dimension
    # ---------------------------------------------------------

    dimension = plan.get("dimension")

    if dimension not in available_dimensions:
        dimension = None

    # ---------------------------------------------------------
    # Nested ranking levels
    #
    # Levels are ordered from outer ranking to inner ranking.
    # Existing single-level rankings continue using dimension/top_n.
    # ---------------------------------------------------------

    raw_ranking_levels = plan.get("ranking_levels", [])

    if raw_ranking_levels is None:
        raw_ranking_levels = []

    if not isinstance(raw_ranking_levels, list):
        validation_errors.append(
            "ranking_levels must be a list."
        )
        raw_ranking_levels = []

    ranking_levels = []

    for index, level in enumerate(raw_ranking_levels):
        if not isinstance(level, dict):
            validation_errors.append(
                f"ranking_levels[{index}] must be an object."
            )
            continue

        level_dimension = level.get("dimension")
        level_top_n = level.get("top_n")

        if level_dimension not in available_dimensions:
            validation_errors.append(
                f"ranking_levels[{index}] contains an unsupported "
                f"dimension {level_dimension!r}."
            )
            continue

        if (
            not isinstance(level_top_n, int)
            or isinstance(level_top_n, bool)
            or level_top_n <= 0
        ):
            validation_errors.append(
                f"ranking_levels[{index}] requires a positive integer top_n."
            )
            continue

        ranking_levels.append(
            {
                "dimension": level_dimension,
                "top_n": level_top_n,
            }
        )

    if len(ranking_levels) > 2:
        validation_errors.append(
            "Only two nested ranking levels are currently supported."
        )
        ranking_levels = ranking_levels[:2]

    if (
        len(ranking_levels) == 2
        and ranking_levels[0]["dimension"]
        == ranking_levels[1]["dimension"]
    ):
        validation_errors.append(
            "Nested ranking levels must use different dimensions."
        )

    # ---------------------------------------------------------
    # GROUP BY
    # ---------------------------------------------------------

    raw_group_by = plan.get(
        "group_by",
        [],
    )

    # Allow the planner to return one field as a string.
    if isinstance(raw_group_by, str):
        raw_group_by = [raw_group_by]

    if not isinstance(raw_group_by, list):
        validation_errors.append(
            "group_by must be a list of dataset fields."
        )
        raw_group_by = []

    allowed_time_groups = {
        "day",
        "week",
        "month",
        "quarter",
        "year",
    }

    group_by = []

    for group_field in raw_group_by:
        if group_field in available_fields:
            if group_field not in group_by:
                group_by.append(group_field)

        elif (
            group_field in allowed_time_groups
            and date_column
        ):
            if group_field not in group_by:
                group_by.append(group_field)

        else:
            validation_errors.append(
                f"Group-by field {group_field!r} is not available "
                "in the current dataset."
            )

    # ---------------------------------------------------------
    # Filters
    # ---------------------------------------------------------

    raw_filters = plan.get(
        "filters",
        [],
    )

    valid_filter_fields = set(
        available_fields
    )

    # Metric fields can also be filter fields.
    for metric_spec in config.get("metrics", {}).values():
        if not isinstance(metric_spec, dict):
            continue

        metric_field = metric_spec.get("field")

        if metric_field:
            valid_filter_fields.add(metric_field)

    if date_column:
        valid_filter_fields.add(date_column)

    # Backwards compatibility:
    # Convert old dictionary-style filters into condition objects.
    if isinstance(raw_filters, dict):
        converted_filters = []

        for filter_field, value in raw_filters.items():

            if filter_field in (
                "date_from",
                "date_to",
            ):
                converted_filters.append(
                    {
                        "field": date_column,
                        "operator": filter_field,
                        "value": value,
                    }
                )

            elif isinstance(value, list):
                converted_filters.append(
                    {
                        "field": filter_field,
                        "operator": "in",
                        "value": value,
                    }
                )

            else:
                converted_filters.append(
                    {
                        "field": filter_field,
                        "operator": "equals",
                        "value": value,
                    }
                )

        raw_filters = converted_filters

    if not isinstance(raw_filters, list):
        validation_errors.append(
            "filters must be a list."
        )
        raw_filters = []

    validated_filters = []

    for condition in raw_filters:

        if not isinstance(condition, dict):
            validation_errors.append(
                "Invalid filter condition."
            )
            continue

        filter_field = condition.get(
            "field"
        )

        operator = condition.get(
            "operator",
            "equals",
        )

        value = condition.get(
            "value"
        )

        # Field must physically exist.
        if filter_field not in valid_filter_fields:
            validation_errors.append(
                f"Filter field {filter_field!r} is not present "
                "in the current dataset."
            )
            continue

        # Operator must be supported.
        if operator not in ALLOWED_FILTER_OPERATORS:
            validation_errors.append(
                f"Filter operator {operator!r} is unsupported."
            )
            continue

        # Date filters must use the actual date column.
        # Date filters may use any real datetime field in the
        # current dataset, not only the default date column.
        if operator in (
            "date_from",
            "date_to",
        ):
            date_fields = {
                field
                for field, profile in config.get(
                    "column_profiles",
                    {},
                ).items()
                if (
                    isinstance(profile, dict)
                    and profile.get("semantic_type") == "datetime"
                )
            }

            if filter_field not in date_fields:
                validation_errors.append(
                    f"Date filter field {filter_field!r} is not "
                    "a detected datetime field. Available date "
                    f"fields: {sorted(date_fields)}."
                )
                continue

        # BETWEEN requires exactly two values.
        if operator == "between":
            if (
                not isinstance(value, list)
                or len(value) != 2
            ):
                validation_errors.append(
                    "The between operator requires exactly "
                    "two values."
                )
                continue

        # IN requires a list.
        # Multiple values excluded with not_equals must use not_in.
        if (
            operator == "not_equals"
            and isinstance(value, list)
        ):
            operator = "not_in"

        # IN and NOT IN always require a list.
        if operator in {"in", "not_in"}:
            if not isinstance(value, list):
                value = [value]

        validated_filters.append(

            {
                "field": filter_field,
                "operator": operator,
                "value": value,
            }
        )

    # ---------------------------------------------------------
    # Unresolved semantic conditions
    # ---------------------------------------------------------

    raw_unresolved = plan.get(
        "unresolved_conditions",
        [],
    )

    if isinstance(raw_unresolved, str):
        raw_unresolved = [
            raw_unresolved
        ]

    if not isinstance(raw_unresolved, list):
        raw_unresolved = [
            "The semantic planner returned invalid "
            "unresolved conditions."
        ]

    unresolved_conditions = [
        str(condition).strip()
        for condition in raw_unresolved
        if str(condition).strip()
    ]

    # ---------------------------------------------------------
    # Top N
    # ---------------------------------------------------------

    top_n = plan.get(
        "top_n"
    )

    if (
        not isinstance(top_n, int)
        or top_n <= 0
    ):
        top_n = None

    # ---------------------------------------------------------
    # Forecast horizon
    # ---------------------------------------------------------

    horizon = plan.get(
        "horizon"
    )

    if (
        not isinstance(horizon, int)
        or horizon <= 0
    ):
        horizon = None

    # ---------------------------------------------------------
    # Granularity
    # ---------------------------------------------------------

    granularity = plan.get(
        "granularity"
    )

    time_group_fields = {
        "day",
        "week",
        "month",
        "quarter",
        "year",
    }

    if (
        granularity is not None
        and granularity not in time_group_fields
    ):
        granularity = None

    has_time_group = any(
        field in time_group_fields
        for field in group_by
    )

    if (
        granularity
        and operation not in {
            "trend",
            "compare",
            "forecast",
        }
        and not has_time_group
    ):
        granularity = None

    # ---------------------------------------------------------
    # Final validated plan
    # ---------------------------------------------------------

    return {
        "operation": operation,
        "metric_keys": metric_keys,
        "field": distinct_field,
        "dimension": dimension,
        "group_by": group_by,
        "filters": validated_filters,
        "filter_logic": filter_logic,
        "unresolved_conditions": unresolved_conditions,
        "validation_errors": validation_errors,
        "top_n": top_n,
        "ranking_levels": ranking_levels,
        "granularity": granularity,
        "horizon": horizon,
        "requires_llm": bool(
            plan.get(
                "requires_llm",
                False,
            )
        ),
        "confidence": plan.get(
            "confidence",
            0.0,
        ),
        "semantic_reason": plan.get(
            "semantic_reason",
            "",
        ),
        "resolution_method": "llm_semantic_plan",
    }

def _semantic_plan_errors(plan: dict) -> list[str]:
    """
    Check whether a validated semantic plan contains everything required
    for its chosen operation.
    """
    errors = list(plan.get("validation_errors", []))

    for condition in plan.get("unresolved_conditions", []):
        errors.append(
            f"Unresolved user condition: {condition}"
        )

    operation = plan.get("operation")
    metric_keys = plan.get("metric_keys", [])
    ranking_levels = plan.get("ranking_levels", [])
    dimension = plan.get("dimension")

    one_metric_operations = {
        "total",
        "average",
        "maximum",
        "minimum",
        "trend",
        "forecast",
        "top_users",
        "bottom_users",
    }

    two_metric_operations = {
        "compare",
        "correlation",
    }

    if (
        operation in one_metric_operations
        and len(metric_keys) < 1
    ):
        errors.append(
            f"operation {operation!r} requires one real metric_key"
        )

    if (
        operation in two_metric_operations
        and len(metric_keys) < 2
    ):
        errors.append(
            f"operation {operation!r} requires two real metric_keys"
        )

    if (
        operation in {"top_users", "bottom_users"}
        and not dimension
        and len(ranking_levels) < 2
    ):
        errors.append(
            f"operation {operation!r} requires a real ranking dimension"
        )

    if ranking_levels and len(ranking_levels) != 2:
        errors.append(
            "nested ranking requires exactly two ranking levels"
        )

    if (
        ranking_levels
        and operation not in {"top_users", "bottom_users"}
    ):
        errors.append(
            "ranking_levels can only be used with a ranking operation"
        )

    if (
        operation == "distinct_count"
        and not plan.get("field")
    ):
        errors.append(
            "operation 'distinct_count' requires a real dataset field"
        )

    return errors

def _enforce_explicit_time_granularity(
    plan: dict,
    query: str,
    config: dict,
) -> dict:
    """
    Preserve explicit time-grouping language from the user's query.

    This prevents the LLM from mentioning monthly/quarterly grouping
    in semantic_reason while omitting granularity and group_by.
    """
    if not isinstance(plan, dict):
        return plan

    if not config.get("date_column"):
        return plan

    normalized_query = query.casefold()

    granularity_patterns = {
        "day": (
            r"\bdaily\b",
            r"\beach day\b",
            r"\bper day\b",
            r"\bby day\b",
        ),
        "week": (
            r"\bweekly\b",
            r"\beach week\b",
            r"\bper week\b",
            r"\bby week\b",
        ),
        "month": (
            r"\bmonthly\b",
            r"\beach month\b",
            r"\bper month\b",
            r"\bby month\b",
        ),
        "quarter": (
            r"\bquarterly\b",
            r"\beach quarter\b",
            r"\bper quarter\b",
            r"\bby quarter\b",
        ),
        "year": (
            r"\byearly\b",
            r"\bannually\b",
            r"\beach year\b",
            r"\bper year\b",
            r"\bby year\b",
        ),
    }

    requested_granularity = None

    for granularity, patterns in granularity_patterns.items():
        if any(
            re.search(pattern, normalized_query)
            for pattern in patterns
        ):
            requested_granularity = granularity
            break

    if requested_granularity is None:
        return plan

    normalized_plan = dict(plan)
    normalized_plan["granularity"] = requested_granularity

    operation = str(
        normalized_plan.get("operation", "")
    ).casefold()

    # Forecasting needs a time interval but should not group historical
    # rows merely because the horizon is expressed in months or years.
    if operation != "forecast":
        raw_group_by = normalized_plan.get("group_by", [])

        if isinstance(raw_group_by, str):
            group_by = [raw_group_by]
        elif isinstance(raw_group_by, list):
            group_by = list(raw_group_by)
        else:
            group_by = []

        time_group_fields = {
            "day",
            "week",
            "month",
            "quarter",
            "year",
        }

        group_by = [
            field
            for field in group_by
            if field not in time_group_fields
        ]

        group_by.append(requested_granularity)
        normalized_plan["group_by"] = group_by

    return normalized_plan




def _enforce_natural_ranking(
    plan: dict,
    query: str,
) -> dict:
    """
    Convert natural-language entity-ranking questions into top/bottom
    ranking operations without hardcoded metric or dimension mappings.

    Examples:
        "Which users have the highest invoice value?"
            -> top_users

        "Which cities have the lowest points?"
            -> bottom_users

        "What is the maximum invoice value?"
            -> remains maximum
    """
    if not isinstance(plan, dict):
        return plan

    normalized_query = query.casefold().strip()
    operation = str(
        plan.get("operation", "")
    ).casefold()

    dimension = plan.get("dimension")

    # A scalar maximum/minimum should remain unchanged.
    if not dimension:
        return plan

    if operation not in {"maximum", "minimum"}:
        return plan

    # "Which users/cities/schemes have..." is an entity-ranking request.
    entity_question = bool(
        re.search(
            r"^\s*(which|what)\b",
            normalized_query,
        )
    )

    if not entity_question:
        return plan

    top_language = bool(
        re.search(
            r"\b(highest|most|best|top|greatest)\b",
            normalized_query,
        )
    )

    bottom_language = bool(
        re.search(
            r"\b(lowest|least|worst|bottom|smallest)\b",
            normalized_query,
        )
    )

    if not top_language and not bottom_language:
        return plan

    normalized_plan = dict(plan)

    if top_language:
        normalized_plan["operation"] = "top_users"
    else:
        normalized_plan["operation"] = "bottom_users"

    top_n = plan.get("top_n")

    if not isinstance(top_n, int) or top_n <= 0:
        normalized_plan["top_n"] = 5

    if not normalized_plan.get("group_by"):
        normalized_plan["group_by"] = [dimension]

    return normalized_plan



def _enforce_monetary_metric(
    plan: dict,
    query: str,
    config: dict,
) -> dict:
    """
    Map general monetary business language to the dataset's monetary
    metric when exactly one suitable monetary metric exists.

    Explicit point/reward questions are left unchanged.
    """
    if not isinstance(plan, dict):
        return plan

    normalized_query = query.casefold()

    # "Points earned" explicitly refers to the reward metric.
    if re.search(r"\bpoints?\b", normalized_query):
        return plan

    monetary_patterns = (
        r"\bearnings?\b",
        r"\brevenue\b",
        r"\bmoney\b",
        r"\bamount billed\b",
        r"\bbilled amount\b",
        r"\bbilling value\b",
        r"\bbusiness value\b",
        r"\bsales value\b",
        r"\bprofit\b",
    )

    if not any(
        re.search(pattern, normalized_query)
        for pattern in monetary_patterns
    ):
        return plan

    operation = str(
        plan.get("operation", "")
    ).casefold()

    # Do not overwrite multi-metric analytical requests.
    if operation in {"compare", "correlation"}:
        return plan

    monetary_metrics = [
        metric_key
        for metric_key, metric_spec
        in config.get("metrics", {}).items()
        if metric_spec.get("semantic_type") == "monetary"
    ]

    # Avoid guessing when the dataset has multiple monetary metrics.
    if len(monetary_metrics) != 1:
        return plan

    metric_key = monetary_metrics[0]
    metric_spec = config["metrics"][metric_key]

    normalized_plan = dict(plan)
    normalized_plan["metric_keys"] = [metric_key]
    normalized_plan["field"] = metric_spec.get("field")

    display_name = metric_spec.get(
        "display_name",
        metric_key,
    )

    normalized_plan["semantic_reason"] = (
        f"The query uses monetary business language, which maps "
        f"to the dataset's monetary metric {display_name!r}."
    )

    normalized_plan["confidence"] = max(
        float(normalized_plan.get("confidence", 0.0)),
        0.95,
    )

    return normalized_plan

def build_semantic_plan(
    query: str,
    data,
    config: dict,
) -> dict:
    """
    Convert arbitrary natural language into a validated execution plan.

    Anthropic is imported lazily so importing this module does not make
    backend startup depend on the Anthropic SDK.
    """
    try:
        from LLM import get_anthropic_client, MODEL
        client = get_anthropic_client()
    except Exception as error:
        logger.warning(
            "Anthropic client unavailable; using deterministic resolver: %s",
            error,
        )
        return None

    metrics = _metric_context(config)

    dimensions = _dimension_context(
        data=data,
        config=config,
        query=query,
    )

    # Dynamically expose every datetime field in the current dataset.
    date_fields = []

    for field, profile in config.get("column_profiles", {}).items():
        if not isinstance(profile, dict):
            continue

        if profile.get("semantic_type") == "datetime":
            date_fields.append(field)

    schema_context = {
        "metrics": metrics,
        "dimensions": dimensions,
        "date_column": config.get("date_column"),
        "date_fields": sorted(set(date_fields)),
        "time_range": config.get("time_range"),
    }

    prompt = f"""
You are the semantic query planner for a business analytics application.

Your job is to understand what the user MEANS, not merely match exact words.

USER QUESTION:
{query}

CURRENT DATASET SCHEMA:
{json.dumps(schema_context, default=str)}

Map the user's business language to the most semantically appropriate fields
in this CURRENT dataset.

Examples of semantic interpretation:
- "earnings", "money generated", "sales value", or similar language may refer
  to a monetary metric if that interpretation is supported by the schema.
- A place, category, user, state, city, product, scheme, device, status,
  boolean state, or other value mentioned or implied in the question should
  become a filter when the current dataset schema supports it.
- Resolve filters using the actual fields and values available in the CURRENT
  dataset. Do not assume field names from previous datasets.
- Natural-language status words should be mapped to boolean/categorical
  fields when the schema supports that meaning. For example, "valid scans"
  should map to an is_valid/status-like field when such a field exists, and
  "invalid scans" should map to its corresponding false/invalid value.
- A phrase such as "in New Delhi" should become a filter on the appropriate
  location field whose available values match New Delhi, such as city, rather
  than being stored only as a generic location string.
- When the user specifies a filter, do not silently drop it just because the
  metric or operation is otherwise easy to resolve.
- "over time" normally means a trend using the available date column.
- "top N", "highest N", "most", "best N", or equivalent ranking language
  means operation "top_users".
- "bottom N", "lowest N", "least", "worst N", or equivalent ranking language
  means operation "bottom_users".
- If a ranking request does not specify N, use top_n = 5.
- For a ranking request, the metric_keys must contain the metric being
  ranked, and dimension must contain the real dataset field representing
  the entities being ranked.
- Interpret ordinary business words semantically using the CURRENT DATASET
  SCHEMA. For example, if "points" clearly refers to a schema metric such
  as "points_earned", use that actual metric key. Do not require the user
  to use the physical column name.
- Never invent a metric. If no schema-supported metric can safely represent
  the user's wording, leave it unresolved.
- When the user asks for results "for each", "per", or "by" an entity
  or category, use "group_by" to represent the real dataset field that
  should define each group.
- Any real categorical, boolean, or identifier field exposed in the
  CURRENT DATASET SCHEMA may be used as a grouping dimension, including
  fields that are not pre-listed in config["dimensions"].
- For example, if the schema contains a categorical field named
  "frequency", then "How many campaigns are there for each frequency?"
  must use group_by = ["frequency"].

- Examples:
  - "How many scans did each user make?" means group_by = ["user_id"]
  - "How many scans did each product have?" means group_by = ["product_id"]
  - "How much quantity did each scheme have?" means group_by = ["scheme_id"]
  - "What is the invoice value by state?" means group_by = ["state"]

- When the user asks for an entity broken down by time, use both group_by
  and granularity.
  Example:
  "How many scans did each user make each month?"
  means group_by = ["user_id"] and granularity = "month".

- "each month" or "monthly" means granularity = "month".
- "each week" or "weekly" means granularity = "week".
- "each quarter" or "quarterly" means granularity = "quarter".
- "each year" or "yearly" means granularity = "year".

- group_by must contain only real physical fields from the current dataset
  schema. Never invent grouping fields.

- Do not use group_by for a normal single-value total.
  means operation "bottom_users".
- When the user requests a ranking inside another ranking, return exactly two
  ordered objects in "ranking_levels".
- The first ranking_levels object is the outer ranking and the second is the
  inner ranking.
- Each ranking level must contain a real dataset dimension and the exact
  positive top_n requested for that level.
- Example: "Top 5 users in the top 5 schemes by invoice value" means:
  operation = "top_users",
  metric_keys = ["invoice_value"],
  dimension = "user_id",
  top_n = 5,
  ranking_levels = [
    {{"dimension": "scheme_id", "top_n": 5}},
    {{"dimension": "user_id", "top_n": 5}}
].
- For ordinary one-level rankings, ranking_levels must be empty.
- For ranking requests, preserve the exact number N requested in "top_n".
- For top_users and bottom_users, "metric_keys" must contain the metric
  whose value determines the ranking.
- For top_users and bottom_users, "dimension" must contain the real dataset
  field representing the entity/category being ranked.
- The ranking dimension can be ANY real categorical/entity field discovered
  in the current dataset. Do not assume it is user_id, customer_id, product,
  city, state, or any other specific field.
- The ranking metric can be ANY real metric supported by the current dataset.
- Never choose a metric merely because it is commonly used for ranking or
  because it was used for a previous dataset.
- Never leave dimension null for top_users or bottom_users when an appropriate
  entity/category dimension exists in the supplied schema.
- If the user asks to rank one field/category by another field/metric,
  use the first as the dimension and the second as the metric.
  For example, "top 3 states by points earned" means:
  operation = "top_users",
  dimension = "state",
  metric = "points earned",
  top_n = 3.
- Do NOT invent columns, metrics, or filter fields.
- Distinguish scalar extrema from entity rankings:
- "What is the maximum invoice value?" means operation "maximum".
- "Which users have the highest invoice value?" means operation "top_users".
- "Which cities have the lowest points?" means operation "bottom_users".
- When the user asks WHICH entities have the highest/lowest/most/least,
    return a ranking operation and a positive top_n. Default top_n to 5
    when no number is specified.
- For questions asking "how many unique", "how many distinct", "number of unique", or equivalent wording, use operation "distinct_count" and select the real dataset field representing the requested entity.
- For distinct_count, the selected field may be any discovered identifier-like field in the current dataset; do not require a pre-created metric for that field.
- Do NOT calculate the final numerical answer.
- The ranking dimension is the entity/category being ranked.
- The ranking metric is the metric whose values determine the ranking.
- Preserve the exact requested N in top_n.
- If ordinary deterministic operations can answer the question, set
  requires_llm to false.
- If the question requires interpretation/recommendations/reasoning beyond
  ordinary aggregation, set requires_llm to true.
- If the requested analysis does not map cleanly to a supported deterministic
  operation, use operation "llm_analysis".
- Every filter field must be a real physical field from the supplied schema.
- Use equals for one categorical value, such as state equals Karnataka.
- Words such as outside, excluding, except, other than, and not in express
  negation. For one excluded categorical value, use operator not_equals.
- Never convert phrases such as "outside Delhi" or "excluding Karnataka"
  into an equals filter because that reverses the user's intent.
- Use in when multiple values are requested.
- Use numeric comparison operators for phrases such as above, below, at least, at most, or between.
- Use date_from and date_to with the real physical date/datetime field
  that matches the meaning expressed by the user.
- The dataset may contain multiple datetime fields. Do NOT automatically
  use the dataset's default date_column when another datetime field is
  semantically more appropriate.
- Interpret date-related business language using the CURRENT DATASET SCHEMA.
- Words such as "created", "created on", or "created during" should refer
  to a datetime field whose meaning represents record creation.
- Words such as "updated", "last updated", or "modified" should refer to
  a datetime field whose meaning represents record modification.
- Words such as "scheduled to start", "starts", "starting", or "campaign
  start" should refer to a datetime field whose meaning represents the
  scheduled start.
- Words such as "scheduled to end", "ends", "ending", or "campaign end"
  should refer to a datetime field whose meaning represents the scheduled
  end.
- If multiple datetime fields could reasonably represent the requested
  meaning and the schema does not provide enough evidence to choose safely,
  leave the date condition unresolved rather than guessing.
- The date_column is only the default date field. It must NOT override a
  more semantically appropriate datetime field.
- Return an empty filters list when the question contains no filter.
- Before returning the plan, check every meaningful condition in the user's
  question and represent it in the "filters" array whenever a corresponding
  field/value exists in the supplied schema.
- For forecast requests, "horizon" must contain the exact number of future
  periods requested by the user.
  - Use group_by to represent every entity or time level described by words
  such as "each", "per", or "by".
- group_by must always be a JSON list, even when it contains one field.
- "How many scans did each user make?" means:
  operation = "distinct_count",
  field = "scan_id",
  group_by = ["user_id"].
- "How many unique users scanned each month?" means:
  operation = "distinct_count",
  field = "user_id",
  group_by = ["month"],
  granularity = "month".
- "How many scans did each user make each month?" means:
  operation = "distinct_count",
  field = "scan_id",
  group_by = ["user_id", "month"],
  granularity = "month".
- Do not represent "each user" or "each month" as a global total.
- Preserve every requested grouping level in group_by.

- "next 3 months" means:
  operation = "forecast"
  horizon = 3
  granularity = "month".

- "next 6 months" means:
  operation = "forecast"
  horizon = 6
  granularity = "month".

- "next 2 quarters" means:
  operation = "forecast"
  horizon = 2
  granularity = "quarter".

- "next year" means:
  operation = "forecast"
  horizon = 1
  granularity = "year".

- Do not require the literal word "forecast" to identify forecast intent.
  Future-looking phrases such as "what will", "what should we expect",
  "expected", "project", "likely to be", "predict", "estimate", and
  equivalent wording should be interpreted semantically as forecast
  requests when the question asks about future values.

- For forecast requests, select the metric whose future value the user is
  asking about and do not calculate the forecast yourself.
- filter_logic describes how multiple filter conditions are combined.
- Use filter_logic "and" when every filter must match.
- Use filter_logic "or" when any filter may match.
- OR conditions on different fields must remain separate filters with
  filter_logic "or".
- Example: "quantity above 3 or points earned above 1000" means:
  filter_logic = "or",
  filters = [
    {{"field": "quantity", "operator": "greater_than", "value": 3}},
    {{"field": "points_earned", "operator": "greater_than", "value": 1000}}
  ].
- Do not route ordinary OR filtering to LLM analysis.
- Do not remove OR filters or calculate an unrestricted total.
- Multiple alternative values for the same field may use one "in" filter.

- If the user specifies a future time period but does not specify the number
  of periods clearly, infer the horizon only when the requested period is
  unambiguous; otherwise set horizon to null.
- Every meaningful condition from the user's question must appear either
  in filters or in unresolved_conditions.
- If a location, identifier, date, category, status, comparison, or other
  condition cannot be mapped safely to the supplied schema, preserve the
  original phrase in unresolved_conditions.
- Never omit an unsupported condition merely to produce a valid plan.
- An unresolved condition is safer than executing a broader unfiltered query.
- For example, if "Atlantis" is not an available dataset value, return:
  "unresolved_conditions": ["location Atlantis"]

- Use "group_by" when the user asks for results for each user,
  product, scheme, state, city, frequency, or any other entity/category.
- "each user" means group_by should contain the real user identifier field.
- "each product" means group_by should contain the real product identifier field.
- "each scheme" means group_by should contain the real scheme identifier field.
- "each month" means granularity = "month".
- If the user asks for an entity broken down by time, use both group_by
  and granularity.
- Do not use group_by for ordinary single-value totals.
- Every group_by field must be a real physical field from the supplied schema.
Allowed operations:
- When the current schema contains a boolean field representing validity,
  "valid scans" and "invalid scans" must use that same boolean field with
  true/1 and false/0 respectively.
- Do not use a categorical status field for "invalid scans" unless the user
  explicitly mentions status, such as "scans where status is invalid".
- Valid scans and invalid scans must be complementary filters on the same
  discovered validity field.
- Set granularity to null unless the user explicitly requests a time
  grouping or forecast period. Never use a placeholder granularity.
{sorted(ALLOWED_OPERATIONS)}

Return ONLY one JSON object using exactly this structure:

{{

  "operation": "total|average|maximum|minimum|trend|compare|forecast|top_users|bottom_users|unique_users|distinct_count|correlation|llm_analysis",
  "metric_keys": ["metric_key_from_schema"],
  "field": "real_dataset_field_or_null",
  "dimension": "dimension_field_or_null",
  "group_by": [],
  "unresolved_conditions": [],
  "filter_logic": "and|or",
    "filters": [
    {{
      "field": "real_dataset_column",
      "operator": "equals|not_equals|in|contains|greater_than|greater_than_or_equal|less_than|less_than_or_equal|between|date_from|date_to",
      "value": "single value, list of values, or two-value range"
    }}
  ],
  "top_n": null,
"ranking_levels": [
    {{
        "dimension": "real_dataset_dimension",
        "top_n": 5
    }}
],
  ],
  "granularity": "day|week|month|quarter|year|null",
  "horizon": null,
  "requires_llm": false,
  "confidence": 0.0,
  "semantic_reason": "short explanation"
}}
"""
    last_validated_plan = None

    last_error = None

    for attempt in range(2):
        attempt_prompt = prompt

        if last_error:
            attempt_prompt += f"""

Your previous semantic plan was invalid for this reason:
{last_error}

Return a corrected plan. Ensure every metric_key, dimension, field, and
filter field exactly matches the supplied current dataset schema.
"""

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1600,
                messages=[
                    {
                        "role": "user",
                        "content": attempt_prompt,
                    }
                ],
            )

            raw_text = "".join(
                block.text
                for block in response.content
                if block.type == "text"
            ).strip()

            raw_plan = _extract_json(raw_text)

            raw_plan = _enforce_explicit_time_granularity(
                plan=raw_plan,
                query=query,
                config=config,
            )

            raw_plan = _enforce_monetary_metric(
                plan=raw_plan,
                query=query,
                config=config,
            )

            raw_plan = _enforce_natural_ranking(
                plan=raw_plan,
                query=query,
            )

            validated_plan = _validate_plan(
                plan=raw_plan,
                config=config,
            )
            last_validated_plan = validated_plan

            validation_errors = _semantic_plan_errors(
                validated_plan
            )

            if not validation_errors:
                return validated_plan

            last_error = "; ".join(validation_errors)

            logger.warning(
                "Semantic plan attempt %s was incomplete: %s",
                attempt + 1,
                last_error,
            )

        except Exception as error:
            last_error = str(error)

            logger.warning(
                "Semantic plan attempt %s failed: %s",
                attempt + 1,
                error,
            )

    logger.error(
        "Semantic planner failed after two attempts: %s",
        last_error,
    )

    if last_validated_plan is not None:
        available_metrics = sorted(
            config.get("metrics", {}).keys()
        )

        last_validated_plan["error"] = (
            "The question contained conditions that could not be "
            f"safely resolved: {last_error}. "
            f"Available metrics: {available_metrics}"
        )
        return last_validated_plan

    return None

def build_recovery_plan(
    query: str,
    data,
    config: dict,
    failed_plan: dict | None,
    error_message: str,
) -> dict | None:
    """
    Ask Claude for one schema-constrained repair after normal planning fails.

    The returned proposal is never executed directly. It must pass the same
    schema validation and semantic completeness checks as an ordinary plan.
    """
    try:
        from LLM import get_anthropic_client, MODEL

        client = get_anthropic_client()
    except Exception as error:
        logger.warning(
            "Recovery planner unavailable: %s",
            error,
        )
        return None

    metrics = _metric_context(config)
    dimensions = _dimension_context(
        data=data,
        config=config,
        query=query,
    )

    date_fields = []

    for field, profile in config.get("column_profiles", {}).items():
        if not isinstance(profile, dict):
            continue

        if profile.get("semantic_type") == "datetime":
            date_fields.append(field)

    schema_context = {
        "metrics": metrics,
        "dimensions": dimensions,
        "date_column": config.get("date_column"),
        "date_fields": sorted(set(date_fields)),
        "time_range": config.get("time_range"),
    }

    prompt = f"""
You repair failed semantic plans for a business analytics application.

USER QUESTION:
{query}

FAILED PLAN:
{json.dumps(failed_plan or {}, default=str)}

BACKEND ERROR:
{error_message}

CURRENT DATASET SCHEMA:
{json.dumps(schema_context, default=str)}

Propose one corrected deterministic plan.

Safety rules:
- Use only metrics, fields, dimensions, and values supported by the schema.
- Preserve every valid filter already present in the failed plan.
- For general dataset-understanding questions such as "what kind of
  data is this", "give me an overview", "summarize this dataset",
  or "what does this dataset contain", use operation "llm_analysis".
- For these general dataset-understanding questions, do NOT put the
  user's question itself into unresolved_conditions. Set
  unresolved_conditions to [] unless there is a specific unsupported
  condition or filter in the question.
- Represent every meaningful user condition in filters.
- If any condition cannot be resolved safely, include it in
  unresolved_conditions.
- Never invent a metric, field, dimension, filter value, or operation.
- Do not return executable code, SQL, prose, markdown, or calculations.
- If the user's question cannot be safely represented by a deterministic
  operation but can be answered by analyzing the current dataset, use
  operation "llm_analysis" and set requires_llm to true.
- Use llm_analysis only for genuine analytical, interpretive, summary,
  recommendation, or dataset-understanding questions.
- Do not use llm_analysis when a supported deterministic operation can
  safely answer the question.
- Return only one JSON object.
- Interpret ordinary business language semantically.
- When the schema contains multiple datetime fields, select the datetime
  field whose semantic meaning matches the user's wording rather than
  automatically using date_column.
- "created" or "created during" refers to the appropriate creation datetime.
- "updated" or "modified" refers to the appropriate update datetime.
- "scheduled to start" or "campaign start" refers to the appropriate
  scheduled-start datetime.
- "scheduled to end" or "campaign end" refers to the appropriate
  scheduled-end datetime.
- If the schema does not provide enough evidence to safely select among
  multiple datetime fields, leave the condition unresolved.
- Do not require the user to use dataset column names.
- If the current schema contains a metric whose semantic meaning clearly
  matches the user's wording, use that schema-supported metric.
- Do not invent aliases, fields, metrics, or dimensions.
- If multiple schema metrics could reasonably match, leave the condition
  unresolved instead of guessing.
- "Which entities have the highest, most, or best values" is a ranking request.
- "Which entities have the lowest, least, or worst values" is a bottom-ranking request.
- If the user does not specify a ranking count, use top_n = 5.

Use exactly this structure:
{{
"operation": "total|average|maximum|minimum|trend|compare|forecast|top_users|bottom_users|unique_users|distinct_count|correlation|llm_analysis",
  "metric_keys": [],
  "field": null,
  "dimension": null,
  "group_by": [],
  "unresolved_conditions": [],
  "filter_logic": "and",
  "filters": [],
  "top_n": null,
  "granularity": null,
  "horizon": null,
  "requires_llm": false,
  "confidence": 0.0,
  "semantic_reason": "short explanation of the repair"
}}
"""

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1600,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        raw_text = "".join(
            block.text
            for block in response.content
            if block.type == "text"
        ).strip()

        print("DEBUG LLM SEMANTIC RESPONSE:", raw_text)
        raw_plan = _extract_json(raw_text)

        raw_plan = _enforce_explicit_time_granularity(
            plan=raw_plan,
            query=query,
            config=config,
        )

        raw_plan = _enforce_monetary_metric(
            plan=raw_plan,
            query=query,
            config=config,
        )

        validated_plan = _validate_plan(
            plan=raw_plan,
            config=config,
        )

        validation_errors = _semantic_plan_errors(
            validated_plan
        )

        if validation_errors:
            logger.warning(
                "Recovery plan rejected: %s",
                "; ".join(validation_errors),
            )
            return None


        # A repair must not remove filters that were already valid.
        failed_filters = (
            failed_plan.get("filters", [])
            if isinstance(failed_plan, dict)
            else []
        )

        repaired_filters = validated_plan.get("filters", [])

        for failed_filter in failed_filters:
            if (
                isinstance(failed_filter, dict)
                and failed_filter not in repaired_filters
            ):
                logger.warning(
                    "Recovery plan rejected because it removed filter %s.",
                    failed_filter,
                )
                return None

        validated_plan["resolution_method"] = "llm_error_recovery"
        validated_plan["error"] = None

        return validated_plan

    except Exception as error:
        logger.warning(
            "Recovery plan generation failed: %s",
            error,
        )
        return None