Metrics = {
    "revenue": [
        "revenue",
        "income",
        "turnover"
    ],
    "profit": [
        "profit",
        "profits",
        "earnings",
        "net income"
    ],
    "expenses": [
        "expense",
        "expenses",
        "cost",
        "costs",
        "spending"
    ],
    "customers": [
        "customer",
        "customers",
        "client",
        "clients",
        "customer count"
    ],
    "sales": [
        "sale",
        "sales",
        "units sold"
    ],
    "churn": [
        "churn",
        "customer churn",
        "churn rate"
    ]
}

Operations = {
    "average": [
        "average",
        "avg",
        "mean"
    ],
    "total": [
        "total",
        "sum",
        "overall"
    ],
    "maximum": [
        "highest",
        "maximum",
        "max",
        "largest",
        "best"
    ],
    "minimum": [
        "lowest",
        "minimum",
        "min",
        "smallest",
        "worst"
    ],
    "compare": [
        "compare",
        "comparison",
        "versus",
        "vs",
        "against"
    ],
    "trend": [
        "trend",
        "over time",
        "movement",
        "pattern"
    ],
    "growth": [
        "growth",
        "increase",
        "decrease",
        "change",
        "growth rate"
    ],
   "forecast": [
        "forecast",
        "predict",
        "prediction",
        "estimate",
        "future",
        "will be",
        "expected",
        "expect",
        "project",
        "projection",
        "next month",
        "next 3 months",
        "next 6 months",
        "next year",
    ],

    "correlation": [
        "correlation",
        "relationship",
        "related",
        "association"
    ]
}

METRIC_TO_FIELD = {

    "profit": "profit",
    "expenses": "operating_expenses",
    "customers": "active_customers",
    "churn": "churned_customers",
}

PredictionMetrics = {
    "revenue": {
        "keywords": ["revenue", "invoice", "sales value", "income", "turnover"],
        "field": "invoice_value",
        "agg": "sum",
    },
    "quantity": {
        "keywords": ["quantity", "units", "products scanned", "items"],
        "field": "quantity",
        "agg": "sum",
    },
    "points": {
        "keywords": ["point", "points", "reward", "loyalty"],
        "field": "points_earned",
        "agg": "sum",
    },
    "activity": {
        "keywords": ["scan", "scans", "activity", "volume", "transaction", "transactions"],
        "field": "scan_id",
        "agg": "count",
    },
}


METRIC_CATEGORY_TO_PREDICTION_METRIC = {
    "revenue": "revenue",
    "sales": "quantity",
}

PredictionGranularity = {
    "day": ["day", "daily"],
    "week": ["week", "weekly"],
    "month": ["month", "monthly"],
}


# ---------------------------------------------------------------------------
# Dataset config (Stage 3)
#
# The ONLY place the prediction pipeline's raw-data column names live now.
# prediction.py and train_models.py both read this instead of hardcoding
# "scan_date" / "is_valid". To point the prediction pipeline at a new
# dataset:
#   1. Update date_column / valid_column here to match the new file.
#   2. Update PredictionMetrics above so each metric's "field" matches a
#      real numeric column in the new file.
# Nothing else in prediction.py or train_models.py needs to change.
#
# valid_column is optional -- set it to None if the dataset has no
# row-validity flag; every row will be used as-is.
# ---------------------------------------------------------------------------

DatasetConfig = {
    "date_column": "scan_date",
    "valid_column": "is_valid",
}