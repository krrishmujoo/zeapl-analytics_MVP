# conversation_orchestrator.py

from __future__ import annotations

import json
import logging
import re
from typing import Any

from dynamic_analysis import analyze_dynamic
from LLM import get_anthropic_client, MODEL
from conversation_memory import (
    conversation_memory,
    build_turn_from_request,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent splitting
# ---------------------------------------------------------------------------

def _clean_intent(text: str) -> str:
    """Normalize whitespace and remove common list prefixes."""
    text = re.sub(r"\s+", " ", text).strip()

    # Remove prefixes such as:
    # "1. What..."
    # "- What..."
    # "a) What..."
    text = re.sub(
        r"^(?:[-*•]\s*|\d+[\.\)]\s*|[a-zA-Z][\.\)]\s*)",
        "",
        text,
    )

    return text.strip()


def _looks_like_independent_intent(text: str) -> bool:
    """
    Conservative check used by the deterministic splitter.

    We only split when the clause looks like an independent analytics
    request rather than part of a filter expression.
    """
    normalized = text.lower().strip()

    if not normalized:
        return False

    analytics_patterns = (
        r"\bwhat\b",
        r"\bwhich\b",
        r"\bhow many\b",
        r"\bshow\b",
        r"\bgive me\b",
        r"\btell me\b",
        r"\bcompare\b",
        r"\banalyze\b",
        r"\bforecast\b",
        r"\bbreak down\b",
        r"\bbreakdown\b",
        r"\bidentify\b",
        r"\bfind\b",
        r"\blist\b",
    )

    return any(re.search(pattern, normalized) for pattern in analytics_patterns)


def deterministic_split(question: str) -> list[str]:
    """
    Conservative multi-intent splitter.

    It intentionally avoids splitting:
      - "Delhi or Rajasthan"
      - "Delhi and Rajasthan"
      - "quantity between 2 and 3"
      - normal single-condition questions

    It primarily handles:
      "What is X, and how many Y?"
      "Show X; show Y"
      "What is X and which Z?"
    """
    question = question.strip()

    if not question:
        return []

    # Semicolons/newlines are strong intent boundaries.
    pieces = re.split(r"[;\n]+", question)

    cleaned = [_clean_intent(piece) for piece in pieces]
    cleaned = [piece for piece in cleaned if piece]

    if len(cleaned) > 1:
        return cleaned

    question = cleaned[0] if cleaned else question

    # Do NOT split "or".
    # Do NOT split geographic "and":
    #   "Delhi and Rajasthan"
    #
    # Only split "and" when the following clause clearly starts another
    # analytics request.
    pattern = re.compile(
        r"(?:,\s*|\s+\band\b\s+)"
        r"(?="
        r"what\b|which\b|how\s+many\b|how\s+much\b|show\b|give\s+me\b|"
        r"tell\s+me\b|compare\b|analyze\b|forecast\b|break\b|"
        r"identify\b|find\b|list\b"
        r")",
        re.IGNORECASE,
    )

    parts = pattern.split(question)

    if len(parts) <= 1:
        return [question]

    parts = [
        _clean_intent(part.rstrip(" ,"))
        for part in parts
    ]

    parts = [
        part
        for part in parts
        if part and _looks_like_independent_intent(part)
    ]

    return parts if len(parts) > 1 else [question]


# ---------------------------------------------------------------------------
# LLM-assisted splitter
# ---------------------------------------------------------------------------

def _llm_split(question: str) -> list[str]:
    """
    Ask Claude to identify independent analytical intents.

    IMPORTANT:
    Claude only splits the question.
    It does NOT calculate anything.
    """
    try:
        client = get_anthropic_client()

        prompt = f"""
You are an intent splitter for an analytics application.

Determine whether the user's message contains one or multiple independent
analytical requests.

Split ONLY independent requests.

Do NOT split:
- "Delhi or Rajasthan"
- "Delhi and Rajasthan"
- "quantity between 2 and 3"
- multiple filters that belong to the same analytical question
- one ranking request with several conditions

Examples:

User:
"What is the invoice value in Delhi and how many valid scans are there?"

Output:
{{
  "intents": [
    "What is the invoice value in Delhi?",
    "How many valid scans are there?"
  ]
}}

User:
"What is the invoice value in Delhi or Rajasthan?"

Output:
{{
  "intents": [
    "What is the invoice value in Delhi or Rajasthan?"
  ]
}}

User:
"What is the invoice value for valid scans in Delhi between January and March?"

Output:
{{
  "intents": [
    "What is the invoice value for valid scans in Delhi between January and March?"
  ]
}}

USER MESSAGE:
{question}

Return ONLY valid JSON:
{{
  "intents": ["..."]
}}
"""

        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
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
            if getattr(block, "type", None) == "text"
        ).strip()

        raw_text = (
            raw_text
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        parsed = json.loads(raw_text)

        intents = parsed.get("intents")

        if not isinstance(intents, list):
            return []

        intents = [
            _clean_intent(str(intent))
            for intent in intents
            if str(intent).strip()
        ]

        # Safety: don't allow Claude to turn one question into hundreds.
        if not intents or len(intents) > 10:
            return []

        return intents

    except Exception as error:
        logger.warning("LLM intent splitting failed: %s", error)
        return []


def split_intents(question: str) -> list[str]:
    """
    Split a user message into independent analytical questions.

    Deterministic splitting is attempted first.
    Claude is used only when the deterministic result is ambiguous.
    """
    deterministic = deterministic_split(question)

    # Strong deterministic result.
    if len(deterministic) > 1:
        return deterministic

    # Otherwise ask Claude whether this is actually multi-intent.
    llm_result = _llm_split(question)

    if len(llm_result) > 1:
        return llm_result

    return [question.strip()]


# ---------------------------------------------------------------------------
# Multi-intent execution
# ---------------------------------------------------------------------------





def _question_needs_context(question: str) -> bool:
    """
    Detect questions that are likely referring to a previous turn.

    This deliberately stays conservative. We don't want to send every
    normal analytics question through Claude just because memory exists.
    """
    q = question.strip().lower()

    if not q:
        return False

    context_phrases = (
        "what about",
        "how about",
        "and what about",
        "what was it",
        "what were they",
        "same for",
        "same in",
        "instead",
        "compared to that",
        "compare that",
        "there",
        "those",
        "these",
        "the same",
        "previous",
        "earlier",
    )

    return (
        len(q.split()) <= 8
        or any(phrase in q for phrase in context_phrases)
    )


def _build_memory_context(
    conversation_id: str,
    dataset_id: str | None,
) -> list[dict]:
    return conversation_memory.get_context(
        conversation_id=conversation_id,
        dataset_id=dataset_id,
    )


def _merge_followup_with_previous_request(
    question: str,
    previous_turn: dict,
) -> str | None:
    """
    Deterministically preserve previous analytical context for simple
    follow-up questions.

    Example:

    Previous:
        invoice_value in Delhi
        Jan 1 2026 -> Mar 31 2026

    Current:
        "What about Rajasthan?"

    Result:
        invoice_value in Rajasthan
        Jan 1 2026 -> Mar 31 2026
    """
    normalized = question.strip().lower()

    if not normalized:
        return None

    previous_requests = previous_turn.get("requests", [])

    if not previous_requests:
        return None

    previous = previous_requests[-1]

    operation = previous.get("operation") or "total"
    metric_keys = previous.get("metric_keys") or []
    filters = previous.get("filters") or []
    granularity = previous.get("granularity")
    top_n = previous.get("top_n")

    # Only handle simple contextual replacements here.
    # More complicated follow-ups will continue to the LLM resolver.
    if not (
        normalized.startswith("what about ")
        or normalized.startswith("how about ")
        or normalized.startswith("and what about ")
    ):
        return None

    new_value = question.strip().rstrip("?.!")
    metric_change = None

    # Detect an explicit metric replacement in follow-ups such as:
    # "What about quantity?"
    # "How about invoice value?"

    prefixes = (
        "what about ",
        "how about ",
        "and what about ",
    )

    for prefix in prefixes:
        if normalized.startswith(prefix):
            new_value = question.strip()[len(prefix):].strip()
            break

    if not new_value:
        return None

    # Remove punctuation from the follow-up value.
    new_value = new_value.rstrip("?.!").strip()

    # Detect an explicit metric replacement such as:
    #   "What about quantity?"
    #   "How about invoice value?"
    metric_change = None

    if "quantity" in normalized:
        metric_change = "quantity"
    elif (
        "invoice value" in normalized
        or "invoice_value" in normalized
    ):
        metric_change = "invoice_value"


    # ---------------------------------------------------------------
    # Explicit trend / month-by-month follow-up.
    #
    # Example:
    #   Previous context:
    #       Rajasthan + Jan-Mar + valid scans
    #
    #   Current:
    #       "show invoice value month by month"
    #
    #   Preserve the existing conversational scope and switch to a
    #   monthly trend on the explicitly requested metric.
    # ---------------------------------------------------------------

    is_monthly_trend = (
        "month by month" in normalized
        or "monthly" in normalized
        or "by month" in normalized
    )

    if is_monthly_trend and metric_change:
        metric_text = metric_change

        sentence = (
            f"Show {metric_text} month by month"
        )

        for item in filters:
            if not isinstance(item, dict):
                continue

            field = item.get("field")
            operator = item.get("operator")
            value = item.get("value")

            if operator == "date_from":
                sentence += f" from {value}"

            elif operator == "date_to":
                sentence += f" to {value}"

            elif operator == "equals":
                if field == "is_valid" and str(value) == "1":
                    sentence += " for valid scans"

                elif field == "is_valid" and str(value) == "0":
                    sentence += " for invalid scans"

                else:
                    sentence += f" where {field} is {value}"

            elif operator == "not_equals":
                sentence += f" where {field} is not {value}"

            elif operator == "in" and isinstance(value, list):
                sentence += (
                    f" where {field} is one of "
                    f"{', '.join(map(str, value))}"
                )

            elif operator == "between" and isinstance(value, list):
                sentence += (
                    f" where {field} is between "
                    f"{value[0]} and {value[1]}"
                )

        return sentence


    # ---------------------------------------------------------------
    # Case 1: The follow-up changes the metric.
    #
    # Example:
    # Previous:
    #   invoice_value + Delhi + Jan-Mar
    #
    # Current:
    #   "What about quantity?"
    #
    # Keep the previous filters and operation, but replace the metric.
    # ---------------------------------------------------------------
    if metric_change:
        metric_text = metric_change

        sentence = (
            f"Give me the {metric_text} using the same analysis as before"
        )

        for item in filters:
            if not isinstance(item, dict):
                continue

            field = item.get("field")
            operator = item.get("operator")
            value = item.get("value")

            if operator == "date_from":
                sentence += f" from {value}"

            elif operator == "date_to":
                sentence += f" to {value}"

            elif operator == "equals":
                sentence += f" where {field} is {value}"

            elif operator == "not_equals":
                sentence += f" where {field} is not {value}"

            elif operator == "in" and isinstance(value, list):
                sentence += (
                    f" where {field} is one of "
                    f"{', '.join(map(str, value))}"
                )

        return sentence


    # ---------------------------------------------------------------
    # Case 2: The follow-up changes the state/location.
    #
    # Example:
    # Previous:
    #   invoice_value + Delhi + Jan-Mar
    #
    # Current:
    #   "What about Rajasthan?"
    #
    # Replace Delhi with Rajasthan while preserving all other filters.
    # ---------------------------------------------------------------
    preserved_filters = []
    replaced = False

    for item in filters:
        if not isinstance(item, dict):
            continue

        field = item.get("field")
        operator = item.get("operator")

        if field == "state" and operator in {
            "equals",
            "not_equals",
        }:
            preserved_filters.append(
                {
                    "field": "state",
                    "operator": "equals",
                    "value": new_value,
                }
            )
            replaced = True
        else:
            preserved_filters.append(item)


    if not replaced:
        return None


    metric_text = (
        metric_keys[0]
        if metric_keys
        else "the same metric"
    )
    sentence = (
        f"Give me the {metric_text} using the same analysis as before"
        f" for {new_value}"
    )

    # Explicitly preserve previous filters by describing them.
    for item in preserved_filters:
        if item["field"] != "state":
            operator = item.get("operator")
            value = item.get("value")

            if operator == "date_from":
                sentence += f" from {value}"
            elif operator == "date_to":
                sentence += f" to {value}"
            elif operator == "equals":
                sentence += f" where {item['field']} is {value}"

    if operation == "top_users":
        sentence = (
            f"Which top {top_n or 5} "
            f"{previous.get('dimension', 'entities')} "
            f"by {metric_text} in {new_value}"
        )

        # Preserve every previous filter except the replaced state.
        for item in preserved_filters:
            if item.get("field") == "state":
                continue

            operator = item.get("operator")
            value = item.get("value")
            field = item.get("field")

            if operator == "date_from":
                sentence += f" from {value}"

            elif operator == "date_to":
                sentence += f" to {value}"

            elif operator == "equals":
                sentence += f" where {field} is {value}"

            elif operator == "not_equals":
                sentence += f" where {field} is not {value}"

            elif operator == "in" and isinstance(value, list):
                sentence += (
                    f" where {field} is one of "
                    f"{', '.join(map(str, value))}"
                )

    elif operation == "bottom_users":
        sentence = (
            f"Which bottom {top_n or 5} "
            f"{previous.get('dimension', 'entities')} "
            f"by {metric_text} in {new_value}"
        )

        for item in preserved_filters:
            if item.get("field") == "state":
                continue

            operator = item.get("operator")
            value = item.get("value")
            field = item.get("field")

            if operator == "date_from":
                sentence += f" from {value}"

            elif operator == "date_to":
                sentence += f" to {value}"

            elif operator == "equals":
                sentence += f" where {field} is {value}"

            elif operator == "not_equals":
                sentence += f" where {field} is not {value}"

            elif operator == "in" and isinstance(value, list):
                sentence += (
                    f" where {field} is one of "
                    f"{', '.join(map(str, value))}"
                )

    return sentence




def _resolve_followup_question(
    question: str,
    conversation_id: str | None,
    dataset_id: str | None,
    context_override: list[dict] | None = None,

) -> str:
    """
    Convert a contextual follow-up into a standalone analytical question.

    Example:

        Previous:
        "What is invoice value in Delhi during Q1 2026?"

        Current:
        "What about Rajasthan?"

        Returned:
        "What is invoice value in Rajasthan during Q1 2026?"
    """
    if not conversation_id:
        return question

    if not _question_needs_context(question):
        return question

    context = (
        context_override
        if context_override is not None
        else _build_memory_context(
            conversation_id,
            dataset_id,
        )
    )
    logger.warning(
        "MEMORY DEBUG - question=%r conversation_id=%r context=%r",
        question,
        conversation_id,
        context,
    )

    if context:
        deterministic_followup = _merge_followup_with_previous_request(
            question=question,
            previous_turn=context[-1],
        )

        logger.warning(
            "MEMORY DEBUG - deterministic_followup=%r",
            deterministic_followup,
        )

        if deterministic_followup:
            logger.info(
                "DETERMINISTIC FOLLOW-UP RESOLVED: %s -> %s",
                question,
                deterministic_followup,
            )
            return deterministic_followup

    if not context:
        return question

    try:
        client = get_anthropic_client()

        context_text = json.dumps(
            context,
            default=str,
            indent=2,
        )

        prompt = f"""
    You are a conversation-context resolver for a business analytics system.

    The user has an existing analytics conversation.

    PREVIOUS CONVERSATION:
    {context_text}

    CURRENT USER QUESTION:
    {question}

    Your task is ONLY to rewrite the current question into a standalone
    analytics question using the previous conversation when necessary.

    Rules:
    - Preserve the user's current intent.
    - Preserve ALL relevant conditions from the immediately preceding
    analytical request unless the user explicitly replaces them.
    - This includes:
    - metric
    - operation
    - filters
    - date ranges
    - specific dates
    - dimensions
    - grouping
    - granularity
    - ranking count
    - If the user changes only one condition, keep all other previous
    conditions unchanged.
    - Do not invent a metric, dimension, filter, date, or value.
    - If the current question is already standalone, return it unchanged.
    - Do not answer the question.
    - Do not calculate anything.

    Example:

    Previous:
    "What is the total invoice value for scans in Delhi between
    January 1, 2026 and March 31, 2026?"

    Current:
    "What about Rajasthan?"

    Output:
    {{
    "standalone_question":
    "What is the total invoice value for scans in Rajasthan between
    January 1, 2026 and March 31, 2026?"
    }}

    Return ONLY valid JSON:

    {{
    "standalone_question": "..."
    }}
    """

        response = client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        raw = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()

        raw = (
            raw
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )

        parsed = json.loads(raw)

        standalone = parsed.get("standalone_question")

        if (
            isinstance(standalone, str)
            and standalone.strip()
        ):
            return standalone.strip()

    except Exception as error:
        logger.warning(
            "Conversation memory resolution failed: %s",
            error,
        )

    return question


def analyze_conversation(
    data,
    question: str,
    dataset_id: str | None = None,
    conversation_id: str | None = None,
    source_path: str | None = None,
    metric_override: str | None = None,
    granularity_override: str | None = None,
    horizon_override: int | None = None,
    use_llm_fallback: bool = False,
) -> dict:
    """
    Conversational orchestration.

    Single intent:
        Resolve the follow-up against persisted conversation memory,
        then execute through the existing analytics engine and return
        the existing single-response shape.

    Multiple intents:
        Split the user's message first, then resolve and execute each
        intent sequentially. The canonical request from each successful
        intent becomes the working context for the next intent.
    """

    original_question = question

    # ---------------------------------------------------------------
    # FIRST split the user's message into independent intents.
    # ---------------------------------------------------------------
    original_intents = split_intents(original_question)

    # ---------------------------------------------------------------
    # Backwards-compatible single-intent path.
    #
    # Memory is resolved against the persisted conversation context.
    # ---------------------------------------------------------------
    if len(original_intents) == 1:
        original_intent = original_intents[0]

        resolved_intent = _resolve_followup_question(
            question=original_intent,
            conversation_id=conversation_id,
            dataset_id=dataset_id,
        )

        if resolved_intent != original_intent:
            logger.info(
                "Resolved intent: %s -> %s",
                original_intent,
                resolved_intent,
            )

        single_response = analyze_dynamic(
            data,
            query=resolved_intent,
            dataset_id=dataset_id,
            source_path=source_path,
            metric_override=metric_override,
            granularity_override=granularity_override,
            horizon_override=horizon_override,
            use_llm_fallback=use_llm_fallback,
        )

        if conversation_id:
            conversation_memory.add_turn(
                conversation_id=conversation_id,
                question=original_question,
                response=single_response,
                dataset_id=dataset_id,
            )

        return single_response

    # ---------------------------------------------------------------
    # Multi-intent execution.
    #
    # Intent 1:
    #   uses persisted conversation memory.
    #
    # Intent 2+:
    #   use the canonical request produced by the previous successful
    #   intent as the working conversational context.
    # ---------------------------------------------------------------
    responses: list[dict[str, Any]] = []

    working_context: list[dict[str, Any]] | None = None

    for index, original_intent in enumerate(
        original_intents,
        start=1,
    ):
        try:
            # -------------------------------------------------------
            # Resolve this intent.
            #
            # For the first intent, working_context is None, so
            # _resolve_followup_question() reads the persisted
            # conversation memory.
            #
            # For later intents, the previous successful canonical
            # request is used as the context.
            # -------------------------------------------------------
            resolved_intent = _resolve_followup_question(
                question=original_intent,
                conversation_id=conversation_id,
                dataset_id=dataset_id,
                context_override=working_context,
            )

            if resolved_intent != original_intent:
                logger.info(
                    "Resolved intent %s: %s -> %s",
                    index,
                    original_intent,
                    resolved_intent,
                )

            # -------------------------------------------------------
            # Execute the resolved intent through the existing engine.
            # -------------------------------------------------------
            result = analyze_dynamic(
                data,
                query=resolved_intent,
                dataset_id=dataset_id,
                source_path=source_path,
                metric_override=metric_override,
                granularity_override=granularity_override,
                horizon_override=horizon_override,
                use_llm_fallback=use_llm_fallback,
            )

            result_status = result.get(
                "status",
                "success",
            )

            responses.append(
                {
                    "intent_id": f"intent_{index}",
                    # Keep the original user wording for the UI.
                    "question": original_intent,
                    "status": result_status,
                    "response": result,
                }
            )

            # -------------------------------------------------------
            # Update working context only from a successful canonical
            # request.
            #
            # Example:
            #
            # Turn 1 memory:
            #   Delhi + Jan-Mar + invoice_value
            #
            # Intent 1:
            #   What about Rajasthan?
            #
            # canonical request:
            #   Rajasthan + Jan-Mar + invoice_value
            #
            # Intent 2:
            #   how many valid scans are there
            #
            # => Rajasthan + Jan-Mar + invoice_value + is_valid=1
            # -------------------------------------------------------
            canonical_request = result.get("request")

            if (
                result_status == "success"
                and isinstance(canonical_request, dict)
            ):
                working_context = [
                    build_turn_from_request(
                        question=resolved_intent,
                        request=canonical_request,
                        dataset_id=dataset_id,
                    )
                ]

        except Exception as error:
            logger.exception(
                "Intent %s failed: %s",
                index,
                error,
            )

            responses.append(
                {
                    "intent_id": f"intent_{index}",
                    "question": original_intent,
                    "status": "error",
                    "response": {
                        "request": {
                            "query": original_intent,
                        },
                        "result": None,
                        "insights": None,
                        "chart_spec": None,
                        "chart_data": None,
                        "prediction": None,
                        "explanation": None,
                        "errors": [
                            f"Intent execution failed: {error}"
                        ],
                        "status": "error",
                    },
                }
            )

            # Do not destroy the previous working context when an
            # individual intent fails. The next intent can still
            # attempt to use the last successful context.

    successful = all(
        item["status"] == "success"
        for item in responses
    )

    combined_response = {
        "status": (
            "success"
            if successful
            else "partial_success"
        ),
        "multi_intent": True,
        "intent_count": len(responses),
        "responses": responses,
    }

    # ---------------------------------------------------------------
    # Store the completed user turn once.
    #
    # This persists the canonical requests from all successful
    # intents for future conversation turns.
    # ---------------------------------------------------------------
    if conversation_id:
        conversation_memory.add_turn(
            conversation_id=conversation_id,
            question=original_question,
            response=combined_response,
            dataset_id=dataset_id,
        )

    return combined_response


if __name__ == "__main__":
    tests = [
        "What is the total invoice value across all scans, how many valid scans are there, and which 5 products have the highest invoice value?",

        "What is the total invoice value for scans in Delhi or Rajasthan?",

        "What is the total invoice value for scans where quantity is between 2 and 3?",

        "What about Rajasthan, how many valid scans are there, and show invoice value month by month?",
    ]

    for test in tests:
        print("\nQUESTION:", test)
        print("INTENTS:", deterministic_split(test))