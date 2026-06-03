"""Prediction contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionSpec:
    prediction_id: str
    model_id: str
    research_dataset_id: str
    prediction_column: str
    primary_key: tuple[str, ...]
    run_type: str = "prediction"
    contract_version: str = "0.1"


__all__ = ["PredictionSpec"]
