"""Pydantic models for the QRP 0.1 API surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer


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


class ApiValidationError(_PublicModel):
    type: str
    loc: tuple[str | int, ...]
    msg: Literal["validation_error"] = "validation_error"


class ApiErrorDetails(_PublicModel):
    errors: list[ApiValidationError] = Field(default_factory=list)

    @model_serializer
    def _serialize(self) -> dict[str, object]:
        if not self.errors:
            return {}
        return {"errors": [item.model_dump() for item in self.errors]}


class ApiErrorPayload(_PublicModel):
    """Common API error payload."""

    error_code: str
    message: str
    field: str | None = None
    details: ApiErrorDetails = Field(default_factory=ApiErrorDetails)


class ApiErrorEnvelope(_PublicModel):
    """Common FastAPI error envelope."""

    error: ApiErrorPayload
