"""Prediction contracts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_research_platform.contracts.filtering import (
    SampleFilterMode,
    validate_sample_filter_mode,
)
from quant_research_platform.contracts.models import MODEL_SELECTION_METRIC_DIRECTIONS
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.time import TimeWindow, time_value_isoformat
from quant_research_platform.contracts.validation import (
    _validate_column_tuple,
    _validate_non_empty_string,
    strict_finite_numeric_series,
)

PREDICTION_SCORE_COLUMN = "prediction_score"
RAW_PREDICTION_COLUMN = "raw_prediction"
ALPHA_SCORE_COLUMN = "alpha_score"
PREDICTION_PRIMARY_KEY = ("timestamp", "asset_id")
A_SHARE_PREDICTION_PRIMARY_KEY = (*PREDICTION_PRIMARY_KEY, "family_id")
PREDICTION_MODES = frozenset({"minimal_oos", "a_share_oos", "crypto_oos"})


def prediction_output_columns(
    prediction_mode: str,
    prediction_column: str,
    primary_key: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    expected_key = (
        A_SHARE_PREDICTION_PRIMARY_KEY
        if prediction_mode == "a_share_oos"
        else PREDICTION_PRIMARY_KEY
    )
    primary_key = expected_key if primary_key is None else primary_key
    if primary_key != expected_key:
        raise ValueError("prediction_output_columns_must_be_fixed")
    if prediction_mode == "a_share_oos":
        if prediction_column != RAW_PREDICTION_COLUMN:
            raise ValueError("prediction_output_columns_must_be_fixed")
        return (
            *primary_key,
            "quarter_id",
            "model_id",
            RAW_PREDICTION_COLUMN,
            ALPHA_SCORE_COLUMN,
        )
    if (
        prediction_mode not in {"minimal_oos", "crypto_oos"}
        or prediction_column != PREDICTION_SCORE_COLUMN
    ):
        raise ValueError("prediction_output_columns_must_be_fixed")
    return (*primary_key, prediction_column)


@dataclass(frozen=True)
class PredictionSpec:
    prediction_id: str
    model_id: str
    research_dataset_id: str
    train_end: str | None
    test_start: str
    test_end: str | None = None
    prediction_column: str = PREDICTION_SCORE_COLUMN
    primary_key: tuple[str, ...] | None = None
    run_type: str = "prediction"
    prediction_mode: str = "minimal_oos"
    sample_filter_mode: SampleFilterMode = "eligible"
    is_oos: bool = True
    is_final_oos_window: bool = True
    candidate_model_ids: tuple[str, ...] = ()
    selection_tie_break_order: tuple[str, ...] = ()
    model_selection_metric: str = "validation_rankic"
    model_selection_validation_start: str | None = None
    model_selection_validation_end: str | None = None
    training_asset_ids: tuple[str, ...] = ()
    evaluation_asset_ids: tuple[str, ...] = ()
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if type(self.contract_version) is not str or self.contract_version != "0.1":
            raise ValueError("unsupported_prediction_contract_version")
        validate_run_id(self.prediction_id)
        _validate_non_empty_string(self.model_id, "model_id_required")
        _validate_non_empty_string(
            self.research_dataset_id,
            "research_dataset_id_required",
        )
        test_start = _time_bound(self.test_start, "test_start")
        if self.train_end is None:
            if (
                self.prediction_mode not in {"a_share_oos", "crypto_oos"}
                or self.candidate_model_ids
            ):
                raise ValueError("prediction_train_end_required")
        else:
            train_end = _time_bound(self.train_end, "train_end")
            _require_same_bound_kind(train_end, test_start)
            object.__setattr__(
                self,
                "train_end",
                time_value_isoformat(train_end, preserve_datetime=True),
            )
        object.__setattr__(
            self,
            "test_start",
            time_value_isoformat(test_start, preserve_datetime=True),
        )
        if self.test_end is not None:
            test_end = _time_bound(self.test_end, "test_end")
            _require_same_bound_kind(test_start, test_end)
            if test_end < test_start:
                raise ValueError("prediction_test_end_must_not_precede_test_start")
            object.__setattr__(
                self,
                "test_end",
                time_value_isoformat(test_end, preserve_datetime=True),
            )
        if self.train_end is not None and train_end >= test_start:
            raise ValueError("prediction_train_end_must_precede_test_start")
        _validate_non_empty_string(self.prediction_column, "prediction_column_required")
        expected_key = (
            A_SHARE_PREDICTION_PRIMARY_KEY
            if self.prediction_mode == "a_share_oos"
            else PREDICTION_PRIMARY_KEY
        )
        if self.primary_key is None:
            object.__setattr__(self, "primary_key", expected_key)
        _validate_column_tuple(self.primary_key, "prediction_primary_key")
        if self.primary_key != expected_key:
            raise ValueError("prediction_output_columns_must_be_fixed")
        if (
            type(self.prediction_mode) is not str
            or self.prediction_mode not in PREDICTION_MODES
        ):
            raise ValueError("unsupported_prediction_mode")
        if self.prediction_mode == "a_share_oos" and (
            self.train_end is not None or self.candidate_model_ids
        ):
            raise ValueError("a_share_persisted_prediction_requires_quarter_lineage")
        if self.prediction_mode == "minimal_oos" and self.train_end is None:
            raise ValueError("prediction_train_end_required")
        expected_prediction_column = (
            RAW_PREDICTION_COLUMN
            if self.prediction_mode == "a_share_oos"
            else PREDICTION_SCORE_COLUMN
        )
        validate_sample_filter_mode(self.sample_filter_mode, owner="prediction")
        if type(self.is_oos) is not bool or type(self.is_final_oos_window) is not bool:
            raise TypeError("prediction_window_flags_must_be_boolean")
        if not self.is_oos:
            raise ValueError("prediction_must_be_oos")
        if self.prediction_column != expected_prediction_column:
            raise ValueError("prediction_output_columns_must_be_fixed")
        if self.test_end is None and not self.is_final_oos_window:
            raise ValueError("prediction_open_ended_window_must_be_final")
        if self.test_end is not None and self.is_final_oos_window:
            object.__setattr__(self, "is_final_oos_window", False)
        _validate_model_selection_contract(self)
        _validate_column_tuple(
            self.training_asset_ids, "training_asset_ids", required=False
        )
        _validate_column_tuple(
            self.evaluation_asset_ids, "evaluation_asset_ids", required=False
        )
        if bool(self.training_asset_ids) != bool(self.evaluation_asset_ids):
            raise ValueError("prediction_asset_scopes_must_be_paired")

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
    if not pd.api.types.is_numeric_dtype(frame[prediction_column]):
        raise ValueError("invalid_prediction_values")
    strict_finite_numeric_series(
        frame[prediction_column],
        error="invalid_prediction_values",
    )


def _validate_model_selection_contract(spec: PredictionSpec) -> None:
    candidate_model_ids = spec.candidate_model_ids
    _validate_column_tuple(
        candidate_model_ids,
        "candidate_model_ids",
        required=False,
    )
    if any(
        not isinstance(model_id, str) or not model_id
        for model_id in candidate_model_ids
    ):
        raise ValueError("candidate_model_ids_must_be_non_empty_strings")
    if len(set(candidate_model_ids)) != len(candidate_model_ids):
        raise ValueError("candidate_model_ids_must_not_contain_duplicates")
    if spec.selection_tie_break_order:
        _validate_column_tuple(
            spec.selection_tie_break_order,
            "selection_tie_break_order",
        )
        if set(spec.selection_tie_break_order) != set(candidate_model_ids):
            raise ValueError("selection_tie_break_order_mismatch")
    if spec.model_id in candidate_model_ids:
        raise ValueError("candidate_model_ids_must_not_include_model_id")
    if (
        type(spec.model_selection_metric) is not str
        or spec.model_selection_metric not in MODEL_SELECTION_METRIC_DIRECTIONS
    ):
        raise ValueError("invalid_model_selection_metric")
    if not candidate_model_ids:
        return
    if (
        spec.model_selection_validation_start is None
        or spec.model_selection_validation_end is None
    ):
        raise ValueError("model_selection_validation_window_required")
    if spec.train_end is None:
        raise ValueError("prediction_train_end_required")
    train_end = _time_bound(spec.train_end, "train_end")
    test_start = _time_bound(spec.test_start, "test_start")
    validation_start = _time_bound(
        spec.model_selection_validation_start,
        "model_selection_validation_start",
    )
    validation_end = _time_bound(
        spec.model_selection_validation_end,
        "model_selection_validation_end",
    )
    _require_same_bound_kind(train_end, validation_start)
    _require_same_bound_kind(validation_start, validation_end)
    _require_same_bound_kind(validation_end, test_start)
    if train_end >= validation_start:
        raise ValueError("model_selection_window_must_follow_train_window")
    if validation_start > validation_end:
        raise ValueError("model_selection_window_must_not_be_reversed")
    if validation_end >= test_start:
        raise ValueError("model_selection_window_must_precede_test_window")


def _time_bound(value: object, field_name: str):
    try:
        return TimeWindow(value, value).start  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_{field_name}") from exc


def _require_same_bound_kind(left: object, right: object) -> None:
    if type(left) is not type(right):
        raise ValueError("mixed_time_window_bounds")
