# conversation_memory.py

from __future__ import annotations

from collections import defaultdict, deque
from threading import Lock
from typing import Any


MAX_TURNS = 6
MAX_CONTEXT_TURNS = 4


class ConversationMemory:
    """
    Lightweight in-process conversation memory.

    This is intentionally session-style memory for the current demo.
    It should later be replaced by Redis/database-backed storage for
    multi-worker production deployment.
    """

    def __init__(self, max_turns: int = MAX_TURNS):
        self.max_turns = max_turns
        self._sessions: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.max_turns)
        )
        self._lock = Lock()

    def add_turn(
        self,
        conversation_id: str,
        question: str,
        response: dict[str, Any],
        dataset_id: str | None = None,
    ) -> None:
        if not conversation_id:
            return

        requests: list[dict[str, Any]] = []

        # Single-intent response
        response_status = response.get(
            "status",
            "success",
        )

        request = response.get("request")

        if (
            response_status == "success"
            and isinstance(request, dict)
        ):
            requests.append(request)

        # Multi-intent response
        for item in response.get("responses", []):
            if item.get("status") != "success":
                continue

            inner = item.get("response", {})

            if not isinstance(inner, dict):
                continue

            if inner.get("status", "success") != "success":
                continue

            inner_request = inner.get("request")

            if isinstance(inner_request, dict):
                requests.append(inner_request)

        # Failed turns must not influence later follow-ups.
        # A partial-success turn stores only its successful intents.
        if not requests:
            return

        # Keep memory compact. We do not store the full result/chart payload.
        compact_requests = []

        for req in requests:
            compact_requests.append(
                {
                    "operation": req.get("operation"),
                    "metric_keys": req.get("metric_keys", []),
                    "metric_fields": req.get("metric_fields", []),
                    "dimension": req.get("dimension"),
                    "group_by": req.get("group_by", []),
                    "filters": req.get("filters", []),
                    "filter_logic": req.get("filter_logic", "and"),
                    "granularity": req.get("granularity"),
                    "horizon": req.get("horizon"),
                    "top_n": req.get("top_n"),
                }
            )

        turn = {
            "question": question,
            "dataset_id": dataset_id,
            "requests": compact_requests,
        }

        with self._lock:
            self._sessions[conversation_id].append(turn)




    def get_context(
        self,
        conversation_id: str,
        dataset_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not conversation_id:
            return []

        with self._lock:
            turns = list(
                self._sessions.get(conversation_id, [])
            )

        if dataset_id is not None:
            turns = [
                turn
                for turn in turns
                if turn.get("dataset_id") in {None, dataset_id}
            ]

        return turns[-MAX_CONTEXT_TURNS:]

    def clear(self, conversation_id: str) -> None:
        if not conversation_id:
            return

        with self._lock:
            self._sessions.pop(conversation_id, None)




def build_turn_from_request(
        question: str,
        request: dict,
        dataset_id: str | None = None,
    ) -> dict:
        return {
            "question": question,
            "dataset_id": dataset_id,
            "requests": [
                {
                    "operation": request.get("operation"),
                    "metric_keys": request.get("metric_keys", []),
                    "metric_fields": request.get("metric_fields", []),
                    "dimension": request.get("dimension"),
                    "group_by": request.get("group_by", []),
                    "filters": request.get("filters", []),
                    "filter_logic": request.get(
                        "filter_logic",
                        "and",
                    ),
                    "granularity": request.get("granularity"),
                    "horizon": request.get("horizon"),
                    "top_n": request.get("top_n"),
                }
            ],
            
        }


conversation_memory = ConversationMemory()