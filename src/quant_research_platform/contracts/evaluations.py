"""Evaluation contracts for QRP 0.1."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.contracts.filtering import (
    EvaluationFilterMode,
    reject_filter_feature_columns,
    validate_evaluation_filter_mode,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.run_id import validate_run_id


@dataclass(frozen=True)
class EvaluationSpec:
    run_id: str
    market_id: MarketId
    research_dataset_id: str
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    filter_mode: EvaluationFilterMode = "eligible"
    run_type: str = "evaluation"
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_evaluation_contract_version")
        validate_run_id(self.run_id)
        validate_evaluation_filter_mode(self.filter_mode)
        reject_filter_feature_columns(self.feature_columns)


@dataclass(frozen=True)
class EvaluationDiagnostics:
    filter_mode: EvaluationFilterMode
    filter_columns_used: tuple[str, ...]
    filter_definition: str
    rows_input: int
    rows_after_filter_mode: int
    rows_after_feature_label_dropna: int
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
            "rows_after_feature_label_dropna",
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


__all__ = ["EvaluationSpec", "EvaluationDiagnostics"]
