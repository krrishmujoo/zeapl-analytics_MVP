"""
model_selection.py -- Phase 7 (model-selection part)

A small, deterministic candidate-model framework: always includes a naive
last-value baseline, compares it against LinearRegression/Ridge (and
HistGradientBoostingRegressor once there's enough history) using
chronological cross-validation, and only "promotes" a fancier model if it
beats the baseline by a meaningful margin. random_state is fixed everywhere
for reproducibility.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
MIN_ROWS_FOR_HGB = 40
# A candidate must beat the baseline's MAE by at least this fraction to be
# selected over it -- otherwise the added complexity isn't earning its keep.
MIN_IMPROVEMENT_OVER_BASELINE = 0.03


def _smape(y_true, y_pred) -> float | None:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not mask.any():
        return None
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100)


def _wape(y_true, y_pred) -> float | None:
    denom = np.abs(y_true).sum()
    if denom == 0:
        return None
    return float(np.abs(y_pred - y_true).sum() / denom * 100)


class NaiveLastValueModel:
    """Predicts the last known training value for every row -- the
    mandatory baseline every other candidate must beat."""
    def fit(self, X, y):
        self._last_value = float(y.iloc[-1]) if len(y) else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), self._last_value)


class SeasonalNaiveModel:
    """Predicts using lag_1 directly (i.e. 'same as last period') -- useful
    when lag_1 is available as a feature; falls back to the naive baseline
    if it isn't."""
    def __init__(self, lag_col="lag_1"):
        self.lag_col = lag_col
        self._fallback = 0.0

    def fit(self, X, y):
        self._fallback = float(y.iloc[-1]) if len(y) else 0.0
        return self

    def predict(self, X):
        if self.lag_col in X.columns:
            return X[self.lag_col].fillna(self._fallback).to_numpy()
        return np.full(len(X), self._fallback)


def _candidate_models(n_rows: int) -> dict:
    models = {
        "naive_last_value": NaiveLastValueModel(),
        "seasonal_naive": SeasonalNaiveModel(),
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    }
    if n_rows >= MIN_ROWS_FOR_HGB:
        from sklearn.ensemble import HistGradientBoostingRegressor
        models["hist_gradient_boosting"] = HistGradientBoostingRegressor(random_state=RANDOM_STATE, max_depth=4)
    return models


def _cv_evaluate(model_factory, X: pd.DataFrame, y: pd.Series, n_splits: int) -> dict | None:
    fold_mae, fold_rmse, fold_r2, fold_smape = [], [], [], []

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(X):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if y_train.nunique() <= 1 and not isinstance(model_factory(), (NaiveLastValueModel, SeasonalNaiveModel)):
            continue

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_mae.append(mean_absolute_error(y_test, preds))
        fold_rmse.append(mean_squared_error(y_test, preds) ** 0.5)
        if y_test.nunique() > 1:
            fold_r2.append(r2_score(y_test, preds))
        smape = _smape(y_test.to_numpy(), np.asarray(preds))
        if smape is not None:
            fold_smape.append(smape)

    if not fold_mae:
        return None

    return {
        "mae": round(float(np.mean(fold_mae)), 4),
        "rmse": round(float(np.mean(fold_rmse)), 4),
        "r2": round(float(np.mean(fold_r2)), 4) if fold_r2 else None,
        "smape": round(float(np.mean(fold_smape)), 2) if fold_smape else None,
        "n_folds": len(fold_mae),
    }


def select_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Runs chronological CV for every candidate, then picks the best one by
    MAE -- but only over the baseline's MAE by more than
    MIN_IMPROVEMENT_OVER_BASELINE, otherwise keeps the baseline. Returns a
    dict with the refit winning model plus every candidate's scores, so the
    "why this was selected" reasoning is always available.
    """
    n = len(X)
    n_splits = min(5, max(2, n // 6))
    n_splits = min(n_splits, n - 1) if n > 1 else 1

    candidates = _candidate_models(n)
    scores = {}

    if n_splits >= 2:
        for name, model in candidates.items():
            factory = candidates[name].__class__ if name not in ("linear_regression", "ridge", "hist_gradient_boosting") else (
                (lambda cls=type(model): cls(alpha=1.0, random_state=RANDOM_STATE)) if name == "ridge"
                else (lambda cls=type(model): cls(random_state=RANDOM_STATE, max_depth=4)) if name == "hist_gradient_boosting"
                else (lambda cls=type(model): cls())
            )
            result = _cv_evaluate(factory, X, y, n_splits)
            if result is not None:
                scores[name] = result

    baseline_mae = scores.get("naive_last_value", {}).get("mae")
    winner_name = "naive_last_value" if "naive_last_value" in scores else (next(iter(scores), None))
    winner_mae = scores.get(winner_name, {}).get("mae") if winner_name else None

    if baseline_mae is not None:
        for name, result in scores.items():
            if name in ("naive_last_value",):
                continue
            if result["mae"] < baseline_mae * (1 - MIN_IMPROVEMENT_OVER_BASELINE):
                if winner_mae is None or result["mae"] < winner_mae:
                    winner_name, winner_mae = name, result["mae"]
        if winner_name is None or (winner_name != "naive_last_value" and scores.get(winner_name, {}).get("mae", float("inf")) >= baseline_mae):
            winner_name = "naive_last_value"

    if not scores:
        # too little data even for 2-fold CV -- fall back to a plain refit
        # with no validation, honestly labeled.
        winner_name = "linear_regression"
        model = LinearRegression().fit(X, y)
        return {
            "model": model, "algorithm": winner_name, "candidate_scores": {},
            "selection_reason": "insufficient history for cross-validated model selection; defaulted to linear_regression, unvalidated",
            "validation": {"mae": None, "rmse": None, "r2": None, "smape": None, "n_folds": 0,
                           "baseline_mae": None, "improvement_over_baseline": None},
        }

    winner_model_map = {
        "naive_last_value": NaiveLastValueModel(),
        "seasonal_naive": SeasonalNaiveModel(),
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    }
    if n >= MIN_ROWS_FOR_HGB:
        from sklearn.ensemble import HistGradientBoostingRegressor
        winner_model_map["hist_gradient_boosting"] = HistGradientBoostingRegressor(random_state=RANDOM_STATE, max_depth=4)

    winning_model = winner_model_map[winner_name].fit(X, y)
    winning_validation = dict(scores.get(winner_name, {}))
    improvement = None
    if baseline_mae and winning_validation.get("mae") is not None and baseline_mae != 0:
        improvement = round((baseline_mae - winning_validation["mae"]) / baseline_mae * 100, 2)
    winning_validation["baseline_mae"] = baseline_mae
    winning_validation["improvement_over_baseline"] = improvement

    reason = (
        f"selected {winner_name} (MAE={winning_validation.get('mae')}) -- "
        + (f"beats naive baseline (MAE={baseline_mae}) by {improvement}%" if improvement is not None
           else "only candidate with a valid score")
        if winner_name != "naive_last_value"
        else f"no candidate beat the naive baseline by >{MIN_IMPROVEMENT_OVER_BASELINE:.0%}, kept the baseline"
    )

    return {
        "model": winning_model,
        "algorithm": winner_name,
        "candidate_scores": scores,
        "selection_reason": reason,
        "validation": winning_validation,
    }