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


__all__ = ["EvaluationSpec"]
