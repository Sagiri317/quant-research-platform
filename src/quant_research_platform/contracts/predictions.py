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
    train_end: str
    test_start: str
    run_type: str = "prediction"
    prediction_mode: str = "minimal_oos"
    is_oos: bool = True
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_prediction_contract_version")
        if self.prediction_mode != "minimal_oos":
            raise ValueError("unsupported_prediction_mode")
        if not self.is_oos:
            raise ValueError("prediction_must_be_oos")


__all__ = ["PredictionSpec"]
