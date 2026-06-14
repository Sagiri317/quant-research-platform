"""Model contracts for non-DL local research."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.contracts.filtering import (
    SampleFilterMode,
    reject_filter_feature_columns,
    validate_sample_filter_mode,
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_family: str
    feature_columns: tuple[str, ...]
    label_column: str
    sample_filter_mode: SampleFilterMode = "eligible"
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_model_contract_version")
        validate_sample_filter_mode(self.sample_filter_mode, owner="model")
        reject_filter_feature_columns(self.feature_columns)


__all__ = ["ModelSpec"]
