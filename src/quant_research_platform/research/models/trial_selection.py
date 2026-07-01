"""Validation-metric-only model trial selection helpers."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd

from quant_research_platform.contracts.predictions import (
    MODEL_SELECTION_METRIC_DIRECTIONS,
)

VALIDATION_METRIC_COLUMNS = ("validation_ic", "validation_rankic", "validation_mse")
TEST_METRIC_COLUMNS = ("test_ic", "test_rankic", "test_mse")


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


def evaluate_model_trial_metrics(
    frame: pd.DataFrame,
    *,
    model_id: str,
    label_column: str,
    prediction_column: str,
) -> dict[str, object]:
    """Evaluate validation-only trial metrics."""

    missing = sorted({label_column, prediction_column}.difference(frame.columns))
    if missing:
        raise ValueError(f"missing_trial_metric_columns: {', '.join(missing)}")
    usable = frame.loc[:, [label_column, prediction_column]].dropna()
    if usable.empty:
        raise ValueError("empty_model_selection_validation_window")
    label_values = pd.to_numeric(usable[label_column], errors="coerce")
    prediction_values = pd.to_numeric(usable[prediction_column], errors="coerce")
    if bool(label_values.isna().any()) or bool(prediction_values.isna().any()):
        raise ValueError("invalid_trial_metric_values")
    errors = prediction_values - label_values
    return {
        "model_id": model_id,
        "validation_rows": int(len(usable)),
        "validation_ic": _correlation(label_values, prediction_values, rank=False),
        "validation_rankic": _correlation(label_values, prediction_values, rank=True),
        "validation_mse": float(np.square(errors).mean()),
    }


def build_model_selection_metadata(
    *,
    configured_model_id: str,
    selected_model_id: str,
    selected_model_input_columns: tuple[str, ...],
    candidate_model_ids: tuple[str, ...],
    selection_metric: str,
    validation_start: str,
    validation_end: str,
    trial_metrics: pd.DataFrame,
) -> dict[str, object]:
    """Build prediction model selection metadata from validation metrics."""

    if selection_metric not in MODEL_SELECTION_METRIC_DIRECTIONS:
        raise ValueError("invalid_model_selection_metric")
    if selected_model_id not in candidate_model_ids:
        raise ValueError("selected_model_not_in_candidates")
    records = trial_metrics.loc[:, ("model_id", *VALIDATION_METRIC_COLUMNS)].to_dict(
        "records"
    )
    for record in records:
        record["validation_rows"] = int(
            trial_metrics.loc[
                trial_metrics["model_id"] == record["model_id"],
                "validation_rows",
            ].iloc[0]
        )
    return {
        "status": "selected",
        "configured_model_id": configured_model_id,
        "selected_model_id": selected_model_id,
        "selected_model_input_columns": list(selected_model_input_columns),
        "candidate_model_ids": list(candidate_model_ids),
        "selection_metric": selection_metric,
        "selection_direction": MODEL_SELECTION_METRIC_DIRECTIONS[selection_metric],
        "validation_start": validation_start,
        "validation_end": validation_end,
        "candidate_count": len(candidate_model_ids),
        "trial_metrics": records,
    }


def _correlation(
    labels: pd.Series,
    predictions: pd.Series,
    *,
    rank: bool,
) -> float:
    if len(labels) < 2:
        return 0.0
    left = labels.rank(method="average") if rank else labels
    right = predictions.rank(method="average") if rank else predictions
    value = left.corr(right)
    if pd.isna(value):
        return 0.0
    return float(value)


__all__ = [
    "VALIDATION_METRIC_COLUMNS",
    "TEST_METRIC_COLUMNS",
    "build_model_selection_metadata",
    "evaluate_model_trial_metrics",
    "select_best_validation_trial",
    "validate_trial_metric_columns",
]
