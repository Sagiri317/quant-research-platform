"""Evaluation contracts for QRP 0.1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from quant_research_platform.contracts.filtering import (
    EvaluationFilterMode,
    validate_evaluation_filter_mode,
)
from quant_research_platform.contracts.ids import MarketId, validate_market_id
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.signals import (
    SignalSpec,
    _cross_signal_id_column_overlaps,
)
from quant_research_platform.contracts.validation import (
    _validate_column_tuple,
    _validate_non_empty_string,
    strict_finite_numeric_series,
)

if TYPE_CHECKING:
    from quant_research_platform.contracts.signals import SignalSpec as _SignalSpec


CORRELATION_RAW_REASON_CODES = frozenset(
    {"insufficient_observations", "zero_variance", "nonfinite_result"}
)
CORRELATION_SUMMARY_REASON_CODES = frozenset(
    {
        "no_valid_observations",
        "insufficient_observations",
        "zero_variance",
        "nonfinite_result",
    }
)
GROUP_SPREAD_REASON_CODES = frozenset({"no_group_transitions"})


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
        if type(self.contract_version) is not str or self.contract_version != "0.1":
            raise ValueError("unsupported_evaluation_contract_version")
        validate_run_id(self.run_id)
        object.__setattr__(self, "market_id", validate_market_id(self.market_id))
        _validate_non_empty_string(
            self.research_dataset_id,
            "research_dataset_id_required",
        )
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
        if type(self.run_type) is not str or not self.run_type:
            raise ValueError("evaluation_run_type_required")


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
        if type(self.contract_version) is not str or self.contract_version != "0.1":
            raise ValueError("unsupported_evaluation_contract_version")
        validate_evaluation_filter_mode(self.filter_mode)
        _validate_column_tuple(
            self.filter_columns_used,
            "filter_columns_used",
            required=False,
        )
        if type(self.filter_definition) is not str or not self.filter_definition:
            raise ValueError("evaluation_filter_definition_required")
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
        for field_name in ("min_timestamp", "max_timestamp"):
            value = getattr(self, field_name)
            if value is not None:
                try:
                    parsed = date.fromisoformat(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid_evaluation_{field_name}") from exc
                if parsed.isoformat() != value:
                    raise ValueError(f"invalid_evaluation_{field_name}")
        if (
            self.min_timestamp is not None
            and self.max_timestamp is not None
            and self.min_timestamp > self.max_timestamp
        ):
            raise ValueError("invalid_evaluation_timestamp_range")


def strict_non_negative_count_series(
    values: pd.Series,
    *,
    error: str = "invalid_evaluation_observation_counts",
) -> pd.Series:
    numeric = strict_finite_numeric_series(values, error=error)
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


def _validate_signals(signals: tuple[_SignalSpec, ...]) -> None:
    if type(signals) is not tuple:
        raise ValueError("evaluation_signals_must_be_tuple")
    if not signals:
        raise ValueError("evaluation_signals_required")
    if any(type(signal) is not SignalSpec for signal in signals):
        raise ValueError("evaluation_signals_invalid")
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
