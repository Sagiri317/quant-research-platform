"""Response DTOs shared by the public API and CLI adapters."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FactorCatalogItem(_PublicModel):
    definition_version: str
    factor_id: str
    market_id: str
    factor_set: str
    display_name: str
    description: str
    family: str
    input_columns: list[str]
    base_input_columns: list[str]
    derived_input_columns: list[str]
    output_column: str
    required_windows: list[int]
    lookback_window: int | None
    grain: str
    timing_semantics: str
    null_policy: str
    source_layer: str
    required_input_data_id: str
    window: int | None = None


class LabelCatalogItem(_PublicModel):
    definition_version: str
    label_id: str
    market_id: str
    family: str
    input_columns: list[str]
    output_column: str
    label_timing: str
    horizon: int
    lookahead: int
    entry_rule: str
    exit_rule: str


class FactorCatalogResponse(_PublicModel):
    api_version: str
    items: list[FactorCatalogItem]


class LabelCatalogResponse(_PublicModel):
    api_version: str
    items: list[LabelCatalogItem]


class ResearchDatasetRequirementsResponse(_PublicModel):
    api_version: str
    status: str
    mode: str
    dataset_id: str
    market_id: str
    input_data_ids: list[str]
    factor_count: int
    feature_count: int
    label_count: int
