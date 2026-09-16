"""
recommendation_engine.py -- Phase 10

Deterministic recommendation rules keyed by semantic category (inferred from
the metric's field name + semantic type -- not a single hardcoded template
for every "increasing" metric). Every recommendation states its evidence and
limitations, and never claims to guarantee an outcome.
"""
import re

# Category keyword rules -- ordered; first match wins. Each entry maps
# name-hints to a semantic action category with direction-specific phrasing.
_CATEGORY_RULES = [
    ("revenue_like", ("revenue", "sales", "income", "turnover"),
     {
         "up": ("cash_flow_preparation", "Review whether capacity and working capital can support the projected demand increase."),
         "down": ("budget_review", "Review budget assumptions and identify which segments are driving the decline."),
     }),
    ("cost_like", ("cost", "expense", "spend", "budget"),
     {
         "up": ("cost_control", "Review cost drivers and margin exposure before the trend compounds."),
         "down": ("no_action", "Costs are trending down; no immediate action indicated beyond continued monitoring."),
     }),
    ("downtime_like", ("downtime", "outage", "failure"),
     {
         "up": ("preventive_maintenance", "Prioritize maintenance inspection and compare predicted high-risk periods against machine-level history."),
         "down": ("no_action", "Downtime is trending down; continue current maintenance cadence."),
     }),
    ("energy_like", ("energy", "electricity", "power", "usage", "consumption"),
     {
         "up": ("demand_planning", "Check expected operational volume and identify periods with abnormal consumption relative to output."),
         "down": ("no_action", "Consumption is trending down; monitor for correctness of sensors/metering."),
     }),
    ("churn_like", ("churn", "cancellation", "attrition"),
     {
         "up": ("customer_retention", "Review retention cohorts and identify segments with rising churn before it compounds."),
         "down": ("no_action", "Churn is trending down; continue current retention efforts."),
     }),
    ("inventory_like", ("inventory", "stock", "units_sold", "quantity", "sold"),
     {
         "up": ("inventory_planning", "Review inventory/production plans against the projected increase in volume."),
         "down": ("inventory_planning", "Review inventory/production plans against the projected decrease in volume to avoid overstock."),
     }),
    ("staffing_like", ("headcount", "staff", "tickets", "orders", "transactions", "volume"),
     {
         "up": ("staffing", "Review staffing/capacity plans against the projected increase in volume."),
         "down": ("staffing", "Review staffing/capacity plans against the projected decrease in volume."),
     }),
]

_NO_CONTEXT_HINTS = ("temperature", "rating", "score", "latency")


def _match_category(field: str) -> tuple[str, dict] | None:
    lower = field.lower()
    for _, keywords, actions in _CATEGORY_RULES:
        if any(k in lower for k in keywords):
            return _, actions
    return None


def build_recommendations(
    metric_spec: dict,
    insights: dict,
    quality: dict,
) -> list[dict]:
    """
    Returns a list of recommendation dicts (usually 1, sometimes 2: one
    business-facing + one data-quality-facing when quality is low).
    """
    field = metric_spec.get("field", "")
    display_name = metric_spec.get("display_name", field)
    direction = insights.get("direction", "stable")
    change_pct = insights.get("forecast_change_percent")
    quality_level = quality.get("quality_level", "unreliable")
    requires_review = quality_level in ("low", "unreliable")

    recommendations = []

    match = _match_category(field)
    has_context = match is not None and not any(h in field.lower() for h in _NO_CONTEXT_HINTS)

    if direction in ("increasing", "decreasing") and has_context:
        category, actions = match
        key = "up" if direction == "increasing" else "down"
        action_category, action_text = actions[key]
        priority = "high" if quality_level in ("high", "moderate") and change_pct and abs(change_pct) > 10 else "medium"
        recommendations.append({
            "priority": priority,
            "category": action_category,
            "action": action_text,
            "reason": f"{display_name} is forecast to {'increase' if direction == 'increasing' else 'decrease'}"
                      + (f" by {change_pct:+.1f}% over the horizon." if change_pct is not None else "."),
            "evidence": {
                "forecast_change_percent": change_pct,
                "quality_level": quality_level,
                "uncertainty": quality.get("score"),
            },
            "limitations": [
                "This is a statistical forecast, not a guarantee of the outcome.",
                "The model does not identify a cause -- only the projected direction and magnitude.",
            ],
            "requires_human_review": requires_review,
        })
    elif direction in ("increasing", "decreasing") and not has_context:
        recommendations.append({
            "priority": "low",
            "category": "data_collection",
            "action": f"No specific business recommendation is given for {display_name} -- its domain meaning "
                       "isn't inferable from the schema alone. Consider adding business context (e.g. a related "
                       "cost/demand metric) if you want actionable guidance here.",
            "reason": f"{display_name} is forecast to {direction}, but no related operational context was discovered in this dataset.",
            "evidence": {"forecast_change_percent": change_pct, "quality_level": quality_level, "uncertainty": quality.get("score")},
            "limitations": ["Analytical next step only -- not a business action, since the metric's business meaning is unknown."],
            "requires_human_review": True,
        })
    else:
        recommendations.append({
            "priority": "low",
            "category": "no_action_monitoring",
            "action": f"{display_name} is forecast to remain stable; no action indicated beyond continued monitoring.",
            "reason": "Forecasted change over the horizon is within normal noise.",
            "evidence": {"forecast_change_percent": change_pct, "quality_level": quality_level, "uncertainty": quality.get("score")},
            "limitations": ["Stability is inferred from a statistical forecast, not verified operationally."],
            "requires_human_review": False,
        })

    if requires_review:
        recommendations.append({
            "priority": "high" if quality_level == "unreliable" else "medium",
            "category": "data_collection",
            "action": "Collect more historical observations, shorten the decision horizon, and use this forecast's "
                       "direction only (not its exact values) until quality improves.",
            "reason": f"Forecast quality is {quality_level} " + "; ".join(quality.get("reasons", [])[:2]) + ".",
            "evidence": {"quality_level": quality_level, "quality_score": quality.get("score")},
            "limitations": ["Do not allocate significant resources based on a low-quality forecast."],
            "requires_human_review": True,
        })

    return recommendations