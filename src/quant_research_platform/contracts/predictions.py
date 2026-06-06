"""Prediction contracts."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.contracts.filtering import (
    SampleFilterMode,
    validate_sample_filter_mode,
)
from quant_research_platform.contracts.run_id import validate_run_id


@dataclass(frozen=True)
class PredictionSpec:
    prediction_id: str
    model_id: str
    research_dataset_id: str
    prediction_column: str
    primary_key: tuple[str, ...]
    train_end: str
    test_start: str
    test_end: str | None = None
    run_type: str = "prediction"
    prediction_mode: str = "minimal_oos"
    sample_filter_mode: SampleFilterMode = "eligible"
    is_oos: bool = True
    is_final_oos_window: bool = True
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_prediction_contract_version")
        validate_run_id(self.prediction_id)
        if self.prediction_mode not in {"minimal_oos", "oos_smoke"}:
            raise ValueError("unsupported_prediction_mode")
        validate_sample_filter_mode(self.sample_filter_mode, owner="prediction")
        if not self.is_oos:
            raise ValueError("prediction_must_be_oos")
        if self.test_end is None and not self.is_final_oos_window:
            raise ValueError("prediction_open_ended_window_must_be_final")
        if self.test_end is not None and self.is_final_oos_window:
            object.__setattr__(self, "is_final_oos_window", False)


__all__ = ["PredictionSpec"]
