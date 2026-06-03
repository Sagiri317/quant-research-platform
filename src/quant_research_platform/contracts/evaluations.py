"""Evaluation contracts for QRP 0.1."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.contracts.ids import MarketId


@dataclass(frozen=True)
class EvaluationSpec:
    run_id: str
    market_id: MarketId
    research_dataset_id: str
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    run_type: str = "evaluation"
    contract_version: str = "0.1"


__all__ = ["EvaluationSpec"]
