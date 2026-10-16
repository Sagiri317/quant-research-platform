"""Response DTOs shared by the public API and CLI adapters."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FactorCatalogItem(_PublicModel):
    factor_id: str
    market_id: str
    factor_set: str
    display_name: str
    description: str
    family: str
    input_columns: list[str]
    output_column: str
    required_windows: list[int]
    lookback_window: int | None
    grain: str
    timing_semantics: str
    null_policy: str
    source_layer: str
    required_input_data_id: str


class LabelCatalogItem(_PublicModel):
    label_id: str
    market_id: str
    family: str
    input_columns: list[str]
    output_column: str
    available_at_column: str
    label_timing: str
    horizon: int
    horizon_unit: str
    lookahead: int
    entry_rule: str
    exit_rule: str


class FactorCatalogResponse(_PublicModel):
    items: list[FactorCatalogItem]


class LabelCatalogResponse(_PublicModel):
    items: list[LabelCatalogItem]


class ResearchDatasetRequirementsResponse(_PublicModel):
    status: str
    mode: str
    dataset_id: str
    market_id: str
    input_data_ids: list[str]
    factor_count: int
    feature_count: int
    label_count: int
