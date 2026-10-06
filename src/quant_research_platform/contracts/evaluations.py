"""Evaluation contracts for QRP."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

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
from quant_research_platform.contracts.time import TimeWindow, time_value_isoformat
from quant_research_platform.contracts.validation import (
    _validate_column_tuple,
    _validate_non_empty_string,
    strict_integer_series,
)

if TYPE_CHECKING:
    from quant_research_platform.contracts.signals import SignalSpec as _SignalSpec


EVALUATION_METRICS_COLUMNS = (
    "timestamp",
    "signal",
    "label",
    "ic",
    "rankic",
    "n_obs",
    "ic_reason",
    "rankic_reason",
)
EVALUATION_GROUPED_LABEL_VALUE_COLUMNS = (
    "timestamp",
    "signal",
    "label",
    "group",
    "mean_label_value",
    "n_obs",
)
EVALUATION_IC_SUMMARY_COLUMNS = (
    "signal",
    "label",
    "mean_ic",
    "std_ic",
    "icir",
    "positive_ic_rate",
    "valid_ic_count",
    "mean_rankic",
    "std_rankic",
    "rankicir",
    "positive_rankic_rate",
    "valid_rankic_count",
    "ic_reason",
    "icir_reason",
    "rankic_reason",
    "rankicir_reason",
)
EVALUATION_GROUP_SPREAD_SUMMARY_COLUMNS = (
    "signal",
    "label",
    "top_group_mean_label_value",
    "bottom_group_mean_label_value",
    "top_bottom_label_value_spread_mean",
    "top_bottom_label_value_spread_hit_rate",
    "group_monotonicity_score",
    "group_monotonicity_score_reason",
)
CORRELATION_RAW_REASON_CODES = frozenset(
    {
        "insufficient_observations",
        "zero_variance",
        "nonfinite_result",
        "not_applicable",
    }
)
CORRELATION_SUMMARY_REASON_CODES = frozenset(
    {
        "no_valid_observations",
        "insufficient_observations",
        "zero_variance",
        "nonfinite_result",
        "not_applicable",
    }
)
GROUP_SPREAD_REASON_CODES = frozenset({"no_group_transitions"})
EvaluationAxis = Literal["cross_sectional", "single_asset_time_series"]
EVALUATION_AXES = frozenset({"cross_sectional", "single_asset_time_series"})


@dataclass(frozen=True)
class EvaluationSpec:
    evaluation_id: str
    market_id: MarketId
    research_dataset_id: str
    signals: tuple[_SignalSpec, ...]
    label_columns: tuple[str, ...]
    label_availability_columns: Mapping[str, str] = field(default_factory=dict)
    evaluation_axis: EvaluationAxis = "cross_sectional"
    filter_mode: EvaluationFilterMode = "eligible"
    n_groups: int = 5
    run_type: str = "evaluation"

    def __post_init__(self) -> None:
        validate_run_id(self.evaluation_id)
        object.__setattr__(self, "market_id", validate_market_id(self.market_id))
        _validate_non_empty_string(
            self.research_dataset_id,
            "research_dataset_id_required",
        )
        if (
            type(self.evaluation_axis) is not str
            or self.evaluation_axis not in EVALUATION_AXES
        ):
            raise ValueError("evaluation_axis_invalid")
        validate_evaluation_filter_mode(self.filter_mode)
        _validate_signals(self.signals)
        _validate_column_tuple(self.label_columns, "label_columns")
        availability = dict(self.label_availability_columns)
        if any(
            type(label) is not str
            or label not in self.label_columns
            or type(column) is not str
            or not column
            for label, column in availability.items()
        ):
            raise ValueError("label_availability_columns_invalid")
        object.__setattr__(
            self,
            "label_availability_columns",
            MappingProxyType(dict(sorted(availability.items()))),
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

    def __post_init__(self) -> None:
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
        parsed_bounds: list[date | datetime | None] = []
        for field_name in ("min_timestamp", "max_timestamp"):
            value = getattr(self, field_name)
            parsed: date | datetime | None = None
            if value is not None:
                try:
                    parsed = TimeWindow(value, value).start
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid_evaluation_{field_name}") from exc
                if (
                    time_value_isoformat(
                        parsed,
                        preserve_datetime=isinstance(parsed, datetime),
                    )
                    != value
                ):
                    raise ValueError(f"invalid_evaluation_{field_name}")
            parsed_bounds.append(parsed)
        minimum, maximum = parsed_bounds
        if (
            minimum is not None
            and maximum is not None
            and (
                isinstance(minimum, datetime) != isinstance(maximum, datetime)
                or minimum > maximum
            )
        ):
            raise ValueError("invalid_evaluation_timestamp_range")


def strict_non_negative_count_series(
    values: pd.Series,
    *,
    error: str = "invalid_evaluation_observation_counts",
) -> pd.Series:
    numeric = strict_integer_series(values, error=error)
    if numeric.lt(0).any():
        raise ValueError(error)
    return numeric


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
