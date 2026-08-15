"""Prediction contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from quant_research_platform.contracts.filtering import (
    SampleFilterMode,
    validate_sample_filter_mode,
)
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.validation import strict_finite_numeric_series

PREDICTION_SCORE_COLUMN = "prediction_score"
PREDICTION_PRIMARY_KEY = ("timestamp", "asset_id")
MODEL_SELECTION_METRIC_DIRECTIONS = {
    "validation_ic": "maximize",
    "validation_rankic": "maximize",
    "validation_mse": "minimize",
}


@dataclass(frozen=True)
class PredictionSpec:
    prediction_id: str
    model_id: str
    research_dataset_id: str
    train_end: str
    test_start: str
    test_end: str | None = None
    prediction_column: str = PREDICTION_SCORE_COLUMN
    primary_key: tuple[str, ...] = PREDICTION_PRIMARY_KEY
    run_type: str = "prediction"
    prediction_mode: str = "minimal_oos"
    sample_filter_mode: SampleFilterMode = "eligible"
    is_oos: bool = True
    is_final_oos_window: bool = True
    candidate_model_ids: tuple[str, ...] = ()
    model_selection_metric: str = "validation_rankic"
    model_selection_validation_start: str | None = None
    model_selection_validation_end: str | None = None
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_prediction_contract_version")
        validate_run_id(self.prediction_id)
        if (
            self.prediction_column != PREDICTION_SCORE_COLUMN
            or self.primary_key != PREDICTION_PRIMARY_KEY
        ):
            raise ValueError("prediction_output_columns_must_be_fixed")
        if self.prediction_mode != "minimal_oos":
            raise ValueError("unsupported_prediction_mode")
        validate_sample_filter_mode(self.sample_filter_mode, owner="prediction")
        if not self.is_oos:
            raise ValueError("prediction_must_be_oos")
        if self.test_end is None and not self.is_final_oos_window:
            raise ValueError("prediction_open_ended_window_must_be_final")
        if self.test_end is not None and self.is_final_oos_window:
            object.__setattr__(self, "is_final_oos_window", False)
        _validate_model_selection_contract(self)

    @property
    def model_selection_enabled(self) -> bool:
        return bool(self.candidate_model_ids)


def _validate_prediction_output_values(
    frame: pd.DataFrame,
    *,
    primary_key: tuple[str, ...],
    prediction_column: str,
) -> None:
    if bool(frame.loc[:, list(primary_key)].isna().any(axis=None)):
        raise ValueError("invalid_prediction_keys")
    if bool(frame.duplicated(list(primary_key)).any()):
        raise ValueError("duplicate_prediction_keys")
    strict_finite_numeric_series(
        frame[prediction_column],
        error="invalid_prediction_values",
    )


def _validate_model_selection_contract(spec: PredictionSpec) -> None:
    candidate_model_ids = spec.candidate_model_ids
    if any(
        not isinstance(model_id, str) or not model_id
        for model_id in candidate_model_ids
    ):
        raise ValueError("candidate_model_ids_must_be_non_empty_strings")
    if len(set(candidate_model_ids)) != len(candidate_model_ids):
        raise ValueError("candidate_model_ids_must_not_contain_duplicates")
    if spec.model_id in candidate_model_ids:
        raise ValueError("candidate_model_ids_must_not_include_model_id")
    if spec.model_selection_metric not in MODEL_SELECTION_METRIC_DIRECTIONS:
        raise ValueError("invalid_model_selection_metric")
    if not candidate_model_ids:
        return
    if (
        spec.model_selection_validation_start is None
        or spec.model_selection_validation_end is None
    ):
        raise ValueError("model_selection_validation_window_required")
    train_end = _iso_date(spec.train_end, "train_end")
    test_start = _iso_date(spec.test_start, "test_start")
    validation_start = _iso_date(
        spec.model_selection_validation_start,
        "model_selection_validation_start",
    )
    validation_end = _iso_date(
        spec.model_selection_validation_end,
        "model_selection_validation_end",
    )
    if train_end >= validation_start:
        raise ValueError("model_selection_window_must_follow_train_window")
    if validation_start > validation_end:
        raise ValueError("model_selection_window_must_not_be_reversed")
    if validation_end >= test_start:
        raise ValueError("model_selection_window_must_precede_test_window")


def _iso_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid_{field_name}") from exc


__all__ = [
    "PREDICTION_PRIMARY_KEY",
    "PREDICTION_SCORE_COLUMN",
    "MODEL_SELECTION_METRIC_DIRECTIONS",
    "PredictionSpec",
]
