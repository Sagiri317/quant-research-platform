"""Evaluation contracts for QRP 0.1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from quant_research_platform.contracts.filtering import (
    EvaluationFilterMode,
    validate_evaluation_filter_mode,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.signals import _cross_signal_id_column_overlaps
from quant_research_platform.contracts.validation import _validate_column_tuple

if TYPE_CHECKING:
    from quant_research_platform.contracts.signals import SignalSpec as _SignalSpec


@dataclass(frozen=True)
class EvaluationSpec:
    run_id: str
    market_id: MarketId
    research_dataset_id: str
    signals: tuple[_SignalSpec, ...]
    label_columns: tuple[str, ...]
    filter_mode: EvaluationFilterMode = "eligible"
    n_groups: int = 5
    segment_columns: tuple[str, ...] = ()
    time_breakdowns: tuple[str, ...] = ("year", "month")
    run_type: str = "evaluation"
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_evaluation_contract_version")
        validate_run_id(self.run_id)
        validate_evaluation_filter_mode(self.filter_mode)
        _validate_signals(self.signals)
        _validate_column_tuple(self.label_columns, "label_columns")
        _validate_column_tuple(
            self.segment_columns,
            "segment_columns",
            required=False,
        )
        _validate_column_tuple(
            self.time_breakdowns,
            "time_breakdowns",
            required=False,
        )
        if not isinstance(self.n_groups, int) or isinstance(self.n_groups, bool):
            raise TypeError("n_groups must be an integer")
        if self.n_groups < 2:
            raise ValueError("evaluation_n_groups_too_small")
        _validate_signal_label_separation(self.signals, self.label_columns)


@dataclass(frozen=True)
class EvaluationDiagnostics:
    filter_mode: EvaluationFilterMode
    filter_columns_used: tuple[str, ...]
    filter_definition: str
    rows_input: int
    rows_after_filter_mode: int
    rows_after_signal_label_dropna: int
    metric_rows: int
    grouped_label_value_rows: int
    ic_summary_rows: int
    group_spread_summary_rows: int
    timestamp_count: int
    min_timestamp: str | None
    max_timestamp: str | None
    n_groups: int
    skipped_timestamp_count: int
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_evaluation_contract_version")
        validate_evaluation_filter_mode(self.filter_mode)
        count_fields = (
            "rows_input",
            "rows_after_filter_mode",
            "rows_after_signal_label_dropna",
            "metric_rows",
            "grouped_label_value_rows",
            "ic_summary_rows",
            "group_spread_summary_rows",
            "timestamp_count",
            "skipped_timestamp_count",
        )
        for field_name in count_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"invalid_evaluation_diagnostics_count: {field_name}")
        if not isinstance(self.n_groups, int) or isinstance(self.n_groups, bool):
            raise ValueError("invalid_evaluation_diagnostics_count: n_groups")
        if self.n_groups < 2:
            raise ValueError("evaluation_diagnostics_n_groups_too_small")


def strict_non_negative_count_series(
    values: pd.Series,
    *,
    error: str = "invalid_evaluation_observation_counts",
) -> pd.Series:
    numeric = _strict_numeric_series(values, error=error, allow_null=False)
    if numeric.lt(0).any() or numeric.mod(1).ne(0).any():
        raise ValueError(error)
    if values.map(lambda value: isinstance(value, bool)).any():
        raise ValueError(error)
    return numeric.astype("int64")


def strict_positive_group_series(
    values: pd.Series,
    *,
    error: str = "evaluation_group_out_of_range",
) -> pd.Series:
    numeric = strict_non_negative_count_series(values, error=error)
    if numeric.le(0).any():
        raise ValueError(error)
    return numeric


def strict_non_negative_count_value(
    value: object,
    *,
    error: str = "invalid_evaluation_diagnostics_count",
) -> int:
    series = strict_non_negative_count_series(pd.Series([value]), error=error)
    return int(series.iloc[0])


def strict_finite_or_null_numeric_series(
    values: pd.Series,
    *,
    error: str = "invalid_evaluation_numeric_values",
) -> pd.Series:
    return _strict_numeric_series(values, error=error, allow_null=True)


def _strict_numeric_series(
    values: pd.Series,
    *,
    error: str,
    allow_null: bool,
) -> pd.Series:
    if values.map(lambda value: isinstance(value, bool)).any():
        raise ValueError(error)
    numeric = pd.to_numeric(values, errors="coerce")
    if not allow_null and numeric.isna().any():
        raise ValueError(error)
    finite = numeric.dropna().map(math.isfinite)
    if not bool(finite.all()):
        raise ValueError(error)
    return numeric


def _validate_signals(signals: tuple[_SignalSpec, ...]) -> None:
    if not signals:
        raise ValueError("evaluation_signals_required")
    signal_ids = [signal.signal_id for signal in signals]
    if len(set(signal_ids)) != len(signal_ids):
        raise ValueError("evaluation_signal_ids_must_not_contain_duplicates")
    columns = [signal.source_column for signal in signals]
    if len(set(columns)) != len(columns):
        raise ValueError("evaluation_signal_columns_must_not_contain_duplicates")
    overlaps = _cross_signal_id_column_overlaps(signals)
    if overlaps:
        raise ValueError("evaluation_signal_id_column_overlap: " + ", ".join(overlaps))


def _validate_signal_label_separation(
    signals: tuple[_SignalSpec, ...],
    label_columns: tuple[str, ...],
) -> None:
    signal_names = {signal.signal_id for signal in signals}.union(
        signal.source_column for signal in signals
    )
    invalid = sorted(signal_names.intersection(label_columns))
    if invalid:
        raise ValueError("evaluation_signal_label_overlap: " + ", ".join(invalid))
