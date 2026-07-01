"""Validation-metric-only model trial selection helpers."""

from __future__ import annotations

from collections.abc import Collection

import pandas as pd


def validate_trial_metric_columns(
    *,
    selection_metric: str,
    validation_metric_columns: Collection[str],
    test_metric_columns: Collection[str],
) -> None:
    """Ensure model selection uses validation metrics only."""

    validation_metrics = set(validation_metric_columns)
    test_metrics = set(test_metric_columns)
    if selection_metric not in validation_metrics:
        raise ValueError("selection_metric_must_be_validation_metric")
    if selection_metric in test_metrics:
        raise ValueError("selection_metric_must_not_be_test_metric")


def select_best_validation_trial(
    trials: pd.DataFrame,
    *,
    trial_id_column: str,
    selection_metric: str,
    maximize: bool,
    validation_metric_columns: Collection[str],
    test_metric_columns: Collection[str],
) -> str:
    """Select the best trial using a validation metric."""

    validate_trial_metric_columns(
        selection_metric=selection_metric,
        validation_metric_columns=validation_metric_columns,
        test_metric_columns=test_metric_columns,
    )
    if trials.empty:
        raise ValueError("empty_trials")
    missing = sorted({trial_id_column, selection_metric}.difference(trials.columns))
    if missing:
        raise ValueError(f"missing_trial_selection_columns: {', '.join(missing)}")
    if bool(trials[trial_id_column].duplicated().any()):
        raise ValueError("duplicate_trial_id")
    metric_values = pd.to_numeric(trials[selection_metric], errors="coerce")
    if bool(metric_values.isna().any()):
        raise ValueError("nan_trial_selection_metric")

    ordered = trials.assign(_selection_metric=metric_values).sort_values(
        ["_selection_metric", trial_id_column],
        ascending=[not maximize, True],
        kind="mergesort",
    )
    return str(ordered.iloc[0][trial_id_column])


__all__ = ["select_best_validation_trial", "validate_trial_metric_columns"]
