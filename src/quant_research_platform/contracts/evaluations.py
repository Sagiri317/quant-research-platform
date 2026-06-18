"""Evaluation contracts for QRP 0.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from quant_research_platform.contracts.filtering import (
    EvaluationFilterMode,
    reject_filter_input_columns,
    validate_evaluation_filter_mode,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.run_id import validate_run_id

SignalSourceKind = Literal["factor", "feature", "prediction"]
SignalDirection = Literal["higher", "lower"]
SignalTransform = Literal["identity", "rank", "zscore"]


@dataclass(frozen=True)
class SignalSpec:
    signal_id: str
    column: str
    source_kind: SignalSourceKind
    direction: SignalDirection = "higher"
    transform: SignalTransform = "identity"
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_signal_contract_version")
        if not isinstance(self.signal_id, str) or not self.signal_id:
            raise ValueError("signal_id_required")
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("signal_column_required")
        if self.source_kind not in {"factor", "feature", "prediction"}:
            raise ValueError("unsupported_signal_source_kind")
        if self.direction not in {"higher", "lower"}:
            raise ValueError("unsupported_signal_direction")
        if self.transform not in {"identity", "rank", "zscore"}:
            raise ValueError("unsupported_signal_transform")
        reject_filter_input_columns((self.signal_id, self.column))


@dataclass(frozen=True)
class EvaluationSpec:
    run_id: str
    market_id: MarketId
    research_dataset_id: str
    signals: tuple[SignalSpec, ...]
    label_columns: tuple[str, ...]
    filter_mode: EvaluationFilterMode = "eligible"
    run_type: str = "evaluation"
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_evaluation_contract_version")
        validate_run_id(self.run_id)
        validate_evaluation_filter_mode(self.filter_mode)
        _validate_signals(self.signals)
        _validate_column_tuple(self.label_columns, "label_columns")
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


def _validate_signals(signals: tuple[SignalSpec, ...]) -> None:
    if not signals:
        raise ValueError("evaluation_signals_required")
    signal_ids = [signal.signal_id for signal in signals]
    if len(set(signal_ids)) != len(signal_ids):
        raise ValueError("evaluation_signal_ids_must_not_contain_duplicates")
    columns = [signal.column for signal in signals]
    if len(set(columns)) != len(columns):
        raise ValueError("evaluation_signal_columns_must_not_contain_duplicates")
    overlaps = _cross_signal_id_column_overlaps(signals)
    if overlaps:
        raise ValueError("evaluation_signal_id_column_overlap: " + ", ".join(overlaps))


def _validate_signal_label_separation(
    signals: tuple[SignalSpec, ...],
    label_columns: tuple[str, ...],
) -> None:
    signal_names = {signal.signal_id for signal in signals}.union(
        signal.column for signal in signals
    )
    invalid = sorted(signal_names.intersection(label_columns))
    if invalid:
        raise ValueError("evaluation_signal_label_overlap: " + ", ".join(invalid))


def _cross_signal_id_column_overlaps(signals: tuple[SignalSpec, ...]) -> list[str]:
    overlaps = {
        signal.signal_id
        for index, signal in enumerate(signals)
        for other_index, other in enumerate(signals)
        if index != other_index and signal.signal_id == other.column
    }
    return sorted(overlaps)


def _validate_column_tuple(columns: tuple[str, ...], field_name: str) -> None:
    if not columns:
        raise ValueError(f"{field_name}_required")
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError(f"{field_name}_must_be_non_empty_strings")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{field_name}_must_not_contain_duplicates")


__all__ = [
    "EvaluationDiagnostics",
    "EvaluationSpec",
    "SignalDirection",
    "SignalSourceKind",
    "SignalSpec",
    "SignalTransform",
]
