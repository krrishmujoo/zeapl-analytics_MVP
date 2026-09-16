import threading
from pathlib import Path

from data_sources import load_data
from model_manager import (
    compare_models,
    compute_dataset_fingerprint,
    compute_newest_timestamp,
    get_model_metrics,
    save_model,
    set_training_status,
    should_retrain,
    update_metadata,
)
from prediction import (
    FEATURE_COLUMNS,
    MIN_PERIODS_NEEDED,
    build_time_series,
    create_features,
    prepare_dataframe,
    train_model,
)
from registries import DatasetConfig, PredictionMetrics


GRANULARITIES = ["day", "week", "month"]

DEFAULT_DATA_PATH = Path(__file__).parent / "fact_scan_activity.csv"


def check_and_train_models(data_path: Path | None = None) -> dict:
    """
    Trains/refreshes every metric x granularity model for one dataset.

    Returns a small summary dict (updated/kept/skipped/error counts) so
    callers -- including the test harness in this task -- can verify what
    happened without re-parsing stdout.
    """
    summary = {"updated": [], "kept": [], "skipped": [], "errors": []}

    csv_path = data_path or DEFAULT_DATA_PATH

    try:
        data = load_data(csv_path)
    except Exception as error:
        set_training_status("error", None)
        summary["errors"].append(f"could not load dataset: {error}")
        return summary

    if not data:
        set_training_status("error", None)
        summary["errors"].append("dataset is empty")
        return summary

    dataset_fingerprint = compute_dataset_fingerprint(data)
    newest_timestamp = compute_newest_timestamp(data, DatasetConfig["date_column"])

    set_training_status("training", dataset_fingerprint)

    try:
        try:
            df = prepare_dataframe(
                data,
                DatasetConfig["date_column"],
                DatasetConfig.get("valid_column"),
            )
        except (KeyError, ValueError) as error:
            set_training_status("error", dataset_fingerprint)
            summary["errors"].append(f"data cleaning failed: {error}")
            return summary

        current_rows = len(df)

        for metric_key, metric in PredictionMetrics.items():
            if metric["field"] not in df.columns and metric["agg"] != "count":
                summary["errors"].append(
                    f"{metric_key}: missing expected column {metric['field']!r}, skipped"
                )
                continue

            for granularity in GRANULARITIES:
                label = f"{metric_key} ({granularity})"

                try:
                    series = build_time_series(df, metric["field"], metric["agg"], granularity)

                    if len(series) < MIN_PERIODS_NEEDED[granularity]:
                        summary["skipped"].append(f"{label}: insufficient history "
                                                    f"({len(series)}/{MIN_PERIODS_NEEDED[granularity]})")
                        continue

                    feature_df = create_features(series, granularity)

                    if len(feature_df) < 5:
                        summary["skipped"].append(f"{label}: not enough usable rows after feature engineering")
                        continue

                    retrain, reason = should_retrain(
                        metric_key, granularity, current_rows, dataset_fingerprint, newest_timestamp
                    )
                    if not retrain:
                        summary["kept"].append(f"{label}: up to date")
                        continue

                    model, metrics = train_model(feature_df)
                    old_metrics = get_model_metrics(dataset_fingerprint, metric_key, granularity)

                    if compare_models(metrics, old_metrics):
                        save_model(model, metric_key, granularity, dataset_fingerprint)
                        update_metadata(
                            metric_key,
                            granularity,
                            current_rows,
                            metrics,
                            dataset_fingerprint,
                            newest_timestamp=newest_timestamp,
                            algorithm="linear_regression",
                            feature_count=len(FEATURE_COLUMNS),
                        )
                        summary["updated"].append(f"{label}: retrained ({reason}); R²={metrics.get('r2')}")

                        old_r2 = (old_metrics or {}).get("r2")
                        new_r2 = metrics.get("r2")
                        if old_r2 is not None and new_r2 is not None and old_r2 > 0:
                            drop = (old_r2 - new_r2) / abs(old_r2)
                            if drop > 0.15:
                                summary["errors"].append(
                                    f"{label}: validation R² dropped {drop:.0%} vs previous model "
                                    f"({old_r2} -> {new_r2}); investigate before trusting this forecast"
                                )
                    else:
                        summary["kept"].append(
                            f"{label}: retrained candidate scored worse than saved model, kept existing "
                            f"(old R²={(old_metrics or {}).get('r2')}, candidate R²={metrics.get('r2')})"
                        )

                except Exception as error:
                    summary["errors"].append(f"{label}: {error}")

        set_training_status("ready", dataset_fingerprint)

    except Exception as error:
        set_training_status("error", dataset_fingerprint)
        summary["errors"].append(f"unexpected failure: {error}")
        raise
    finally:
        pass

    return summary


def train_in_background(data_path: Path | None = None) -> threading.Thread:
    """
    Fire-and-forget training for use from an API layer -- starts training
    on a separate thread and returns immediately instead of blocking the
    request. Callers should poll model_manager.get_training_status().
    """
    thread = threading.Thread(target=check_and_train_models, args=(data_path,), daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    result = check_and_train_models()
    for key, items in result.items():
        print(f"\n{key.upper()} ({len(items)}):")
        for item in items:
            print(f"  - {item}")