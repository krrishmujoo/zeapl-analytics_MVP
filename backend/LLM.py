import json
import re
import pandas as pd
from dotenv import load_dotenv
from fastapi import HTTPException

from aggregation import (
    cal_avg,
    highest_rev,
    lowest_rev,
    total_rev,
    month_over_month_growth,
    quarterly_growth,
    total_profit,
    average_profit,
    profit_margin,
    monthly_profit_growth,
)


load_dotenv()

MODEL = "claude-sonnet-5"

_anthropic_client = None


def get_anthropic_client():
    """
    Create the Anthropic client only when an LLM request is made.

    Normal totals, charts, filters, rankings, and forecasts can therefore
    start and run without importing the Anthropic SDK.
    """
    global _anthropic_client

    if _anthropic_client is None:
        from anthropic import Anthropic

        _anthropic_client = Anthropic()

    return _anthropic_client
MAX_RECORDS_FOR_LLM = 500

MAX_PROFILE_VALUES = 10
MAX_CORRELATIONS = 20


def _safe_stat_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def _build_dataset_statistics(
    data: list[dict],
) -> dict:
    """
    Build authoritative statistics from the complete supplied dataset.
    """
    df = pd.DataFrame(data)

    statistics = {
        "record_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "column_profiles": {},
        "numeric_correlations": [],
    }

    numeric_columns = {}

    for column in df.columns:
        series = df[column]
        non_null = series.dropna()

        profile = {
            "non_null_count": int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
            "unique_count": int(
                series.nunique(dropna=True)
            ),
        }

        normalized_column = str(column).casefold()

        is_date_column = any(
            word in normalized_column
            for word in (
                "date",
                "time",
                "timestamp",
            )
        )

        if is_date_column:
            parsed_dates = pd.to_datetime(
                series,
                errors="coerce",
            )
            valid_dates = parsed_dates.dropna()

            if not valid_dates.empty:
                profile.update(
                    {
                        "type": "datetime",
                        "minimum": valid_dates.min().isoformat(),
                        "maximum": valid_dates.max().isoformat(),
                    }
                )

        numeric_series = pd.to_numeric(
            series,
            errors="coerce",
        )

        non_null_count = int(series.notna().sum())
        numeric_count = int(numeric_series.notna().sum())

        numeric_ratio = (
            numeric_count / non_null_count
            if non_null_count
            else 0
        )

        if (
            not is_date_column
            and numeric_count
            and numeric_ratio >= 0.95
        ):
            valid_numeric = numeric_series.dropna()

            profile.update(
                {
                    "type": "numeric",
                    "sum": _safe_stat_value(
                        valid_numeric.sum()
                    ),
                    "mean": _safe_stat_value(
                        valid_numeric.mean()
                    ),
                    "minimum": _safe_stat_value(
                        valid_numeric.min()
                    ),
                    "maximum": _safe_stat_value(
                        valid_numeric.max()
                    ),
                }
            )

            numeric_columns[column] = numeric_series

        elif not is_date_column:
            profile["type"] = "categorical"

        if 0 < profile["unique_count"] <= 100:
            value_counts = (
                non_null.astype(str)
                .value_counts()
                .head(MAX_PROFILE_VALUES)
            )

            profile["top_values"] = [
                {
                    "value": str(value),
                    "count": int(count),
                }
                for value, count in value_counts.items()
            ]

            profile["top_values_are_complete"] = (
                profile["unique_count"]
                <= MAX_PROFILE_VALUES
            )

        statistics["column_profiles"][column] = profile

    numeric_items = list(numeric_columns.items())

    for first_index, (
        first_column,
        first_series,
    ) in enumerate(numeric_items):
        for second_column, second_series in numeric_items[
            first_index + 1:
        ]:
            aligned = pd.concat(
                [first_series, second_series],
                axis=1,
            ).dropna()

            if len(aligned) < 2:
                continue

            if (
                aligned.iloc[:, 0].nunique() < 2
                or aligned.iloc[:, 1].nunique() < 2
            ):
                continue

            correlation = aligned.iloc[:, 0].corr(
                aligned.iloc[:, 1]
            )

            if pd.isna(correlation):
                continue

            statistics["numeric_correlations"].append(
                {
                    "first_column": first_column,
                    "second_column": second_column,
                    "correlation": round(
                        float(correlation),
                        4,
                    ),
                    "observations": int(len(aligned)),
                }
            )

            if (
                len(statistics["numeric_correlations"])
                >= MAX_CORRELATIONS
            ):
                break

        if (
            len(statistics["numeric_correlations"])
            >= MAX_CORRELATIONS
        ):
            break

    return statistics

def _parse_analysis_json(raw_text: str) -> dict:
    """
    Extract and validate the first JSON object in a Claude response.
    """
    text = raw_text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    json_start = text.find("{")

    if json_start == -1:
        raise ValueError(
            "Claude response did not contain a JSON object."
        )

    analysis, _ = json.JSONDecoder().raw_decode(
        text[json_start:]
    )

    if not isinstance(analysis, dict):
        raise ValueError(
            "Claude response must be a JSON object."
        )

    reasoning = analysis.get("reasoning")
    insights = analysis.get("insights", [])

    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError(
            "The reasoning field must be a non-empty string."
        )

    if not isinstance(insights, list):
        raise ValueError(
            "The insights field must be a list."
        )

    clean_insights = [
        insight.strip()
        for insight in insights
        if isinstance(insight, str) and insight.strip()
    ]

    return {
        "reasoning": reasoning.strip(),
        "insights": clean_insights,
    }

def analyze_data(
    data: list[dict],
    question: str | None = None,
    reason: str | None = None,
) -> dict:
    if not data:
        raise HTTPException(
            status_code=400,
            detail="No data to analyze.",
        )

    stats = _build_dataset_statistics(data)

    dataset_for_prompt = data
    dataset_note = ""
    if len(data) > MAX_RECORDS_FOR_LLM:
        dataset_for_prompt = data[:MAX_RECORDS_FOR_LLM]
        dataset_note = (
            f" (showing the first {MAX_RECORDS_FOR_LLM} of "
            f"{len(data)} rows; the summary statistics below "
            "are computed from the complete dataset)"
        )

    question_block = (
        f'The user asked: "{question}"'
        if question
        else "The user did not provide a specific question."
    )

    reason_block = (
        f"Routing reason: {reason}"
        if reason
        else "Routing reason: complex business-analysis request."
    )

    prompt = f"""You are a business data analyst.

Answer the user's question using only the supplied dataset and summary statistics.

{question_block}

{reason_block}

Focus on:
- important patterns and changes,
- comparisons between relevant metrics,
- explanations supported by the available data,
- useful business insights.

Do not choose a chart. Chart selection is handled by another backend module.

Dataset{dataset_note}:
{json.dumps(dataset_for_prompt, default=str)}

Summary statistics:
{json.dumps(stats, default=str)}

Respond with only one raw JSON object.
Do not use markdown fences, a preamble, or additional text.

Use this exact schema:

{{
  "reasoning": "<a clear and concise explanation that directly answers the user's question>",
  "insights": [
    "<important data-supported insight 1>",
    "<important data-supported insight 2>",
    "<important data-supported insight 3>"
  ]
}}

Rules:
- Use only the supplied dataset and statistics.
- Treat the full-dataset summary statistics as authoritative.
- The displayed rows may be only a sample; never describe sample frequencies as full-dataset frequencies.
- Never claim that the dataset contains only the categorical values visible in the row sample.
- top_values may contain only the most frequent values; check top_values_are_complete before treating the list as exhaustive.
- Prefer full-dataset counts, sums, averages, date ranges, distinct counts, value frequencies, and correlations over impressions from sampled rows.
- Do not describe invalid or duplicate records as fraud, data-quality failures, or enforcement outcomes unless the supplied data explicitly proves that cause.
- Do not call a relationship disproportionate unless the supplied statistics directly support that claim.
- When the user requests exactly three findings, return exactly three insights.
- Do not invent facts or causes.
- Clearly state when the data cannot prove why something happened.
- Directly answer the user's question.
- Keep the reasoning concise and business-focused.
- Use the exact metric names from the user's question and supplied dataset.
- Never substitute unrelated metrics such as revenue, profit, or units sold unless those metrics actually exist in the supplied data.
- Keep reasoning under 250 words.
- Return syntactically valid JSON with properly escaped strings.
- Correlation describes association only. Never use words such as causes, proves, confirms, driven by, or because solely from correlation.
- Do not claim that correlation rules out other explanations such as pricing, mix, seasonality, or operational changes.
- Identical aggregate counts or perfect aggregate correlation do not prove a one-to-one relationship between individual entities.
- A constant distinct count means measured breadth stayed constant; it does not prove that the underlying category had no effect on outcomes.
- Use exact dataset metric names. If the dataset contains Invoice Value but not revenue, call it Invoice Value rather than revenue.
- Never add a currency symbol or measurement unit unless the dataset schema or user question explicitly supplies it.
"""

    try:
        client = get_anthropic_client()
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Claude client error: {error}",
        ) from error

    response_prompt = prompt
    last_error = None
    last_raw_text = ""

    for attempt in range(2):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=3200,
                messages=[
                    {
                        "role": "user",
                        "content": response_prompt,
                    }
                ],
            )
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail=f"Claude API error: {error}",
            ) from error

        last_raw_text = "".join(
            block.text
            for block in response.content
            if block.type == "text"
        ).strip()

        try:
            return _parse_analysis_json(
                last_raw_text
            )

        except (json.JSONDecodeError, ValueError) as error:
            last_error = error

            if attempt == 0:
                response_prompt = f"""
Repair the following incomplete or malformed Claude response.

Return only one syntactically valid JSON object using exactly this schema:

{{
  "reasoning": "<clear answer>",
  "insights": [
    "<data-supported insight 1>",
    "<data-supported insight 2>",
    "<data-supported insight 3>"
  ]
}}

Do not use Markdown fences or add any text outside the JSON object.
Preserve only claims already present in the response. Do not invent facts.

INVALID RESPONSE:
{last_raw_text[:12000]}
"""
                continue

    raise HTTPException(
        status_code=502,
        detail=(
            "Claude returned invalid JSON after one repair attempt. "
            f"Validation error: {last_error}. "
            f"Response started with: {last_raw_text[:300]!r}"
        ),
    )