"""Validation-metric-only model trial selection helpers."""

from __future__ import annotations

from collections.abc import Collection
from contextlib import ExitStack

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _pearson_correlation_result,
)
from quant_research_platform.contracts.predictions import (
    MODEL_SELECTION_METRIC_DIRECTIONS,
)
from quant_research_platform.research._memory import _selected_frame_payload_bytes
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
)

VALIDATION_METRIC_COLUMNS = ("validation_ic", "validation_rankic", "validation_mse")
VALIDATION_METRIC_REASON_COLUMNS = (
    "validation_ic_reason",
    "validation_rankic_reason",
)
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
    metric_series = trials[selection_metric]
    if pd.api.types.is_bool_dtype(metric_series):
        raise ValueError("invalid_trial_selection_metric")
    if not pd.api.types.is_numeric_dtype(metric_series):
        if not bool(metric_series.isna().all()):
            raise ValueError("invalid_trial_selection_metric")
        metric_values = pd.Series(
            np.nan,
            index=metric_series.index,
            dtype="float64",
        )
    else:
        metric_values = metric_series.astype("float64")
    if bool((~metric_values.isna() & ~np.isfinite(metric_values)).any()):
        raise ValueError("invalid_trial_selection_metric")
    defined = metric_values.notna()
    if not bool(defined.any()):
        raise ValueError("all_validation_metrics_undefined")

    ordered = (
        trials.loc[defined]
        .assign(_selection_metric=metric_values[defined])
        .sort_values(
            ["_selection_metric", trial_id_column],
            ascending=[not maximize, True],
            kind="mergesort",
        )
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
    selected_output_bound = _selected_frame_payload_bytes(
        frame,
        [label_column, prediction_column],
    )
    usable_output_bound = selected_output_bound
    numeric_vector_count = 6
    numeric_working_bytes = len(frame) * 8 * numeric_vector_count
    mask_working_bytes = len(frame) * 3
    with ExitStack() as reservation:
        policy = current_runtime_resource_policy()
        if policy is not None:
            reservation.enter_context(
                policy.lease(
                    "research.model.selection.metric",
                    selected_output_bound
                    + usable_output_bound
                    + numeric_working_bytes
                    + mask_working_bytes,
                )
            )
        usable = None
        label_values = None
        prediction_values = None
        errors = None
        try:
            usable = frame.loc[:, [label_column, prediction_column]]
            if usable.empty:
                raise ValueError("empty_model_selection_validation_window")
            for column in (label_column, prediction_column):
                values = usable[column]
                if not pd.api.types.is_numeric_dtype(
                    values
                ) or pd.api.types.is_bool_dtype(values):
                    raise ValueError("invalid_trial_metric_values")
            label_values = usable[label_column].astype("float64")
            prediction_values = usable[prediction_column].astype("float64")
            if (
                bool(label_values.isna().any())
                or bool(prediction_values.isna().any())
                or not bool(np.isfinite(label_values).all())
                or not bool(np.isfinite(prediction_values).all())
            ):
                raise ValueError("invalid_trial_metric_values")
            errors = prediction_values - label_values
            validation_ic, validation_ic_reason = _validation_correlation(
                label_values,
                prediction_values,
                rank=False,
            )
            validation_rankic, validation_rankic_reason = _validation_correlation(
                label_values,
                prediction_values,
                rank=True,
            )
            record = {
                "model_id": model_id,
                "validation_rows": int(len(usable)),
                "validation_ic": validation_ic,
                "validation_ic_reason": validation_ic_reason,
                "validation_rankic": validation_rankic,
                "validation_rankic_reason": validation_rankic_reason,
                "validation_mse": float(np.square(errors).mean()),
            }
        except BaseException:
            usable = None
            label_values = None
            prediction_values = None
            errors = None
            raise
        usable = None
        label_values = None
        prediction_values = None
        errors = None
        return record


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
    records = trial_metrics.loc[
        :, ("model_id", *VALIDATION_METRIC_COLUMNS, *VALIDATION_METRIC_REASON_COLUMNS)
    ].to_dict(
        "records"
    )
    for record in records:
        for key in (
            "validation_ic",
            "validation_rankic",
            "validation_ic_reason",
            "validation_rankic_reason",
        ):
            if pd.isna(record[key]):
                record[key] = None
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


def _validation_correlation(
    labels: pd.Series,
    predictions: pd.Series,
    *,
    rank: bool,
) -> tuple[float | None, str | None]:
    if len(labels) < 2:
        return None, "insufficient_observations"
    left = labels.rank(method="average") if rank else labels
    right = predictions.rank(method="average") if rank else predictions
    return _pearson_correlation_result(
        left.to_numpy(dtype="float64", copy=False),
        right.to_numpy(dtype="float64", copy=False),
    )
