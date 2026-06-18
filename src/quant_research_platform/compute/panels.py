"""Compute-local panel specs and validation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator


class ComputePanelSpec(BaseModel):
    """Column contract for long-format compute panels."""

    model_config = ConfigDict(frozen=True)

    feature_columns: tuple[str, ...]
    label_column: str
    timestamp_column: str = "timestamp"
    asset_column: str = "asset"

    @field_validator("feature_columns")
    @classmethod
    def _validate_feature_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("feature_columns must contain at least one column.")
        if any(not isinstance(column, str) or not column for column in value):
            raise ValueError("feature_columns must contain non-empty strings.")
        if len(set(value)) != len(value):
            raise ValueError("feature_columns must not contain duplicates.")
        return value

    @field_validator("label_column", "timestamp_column", "asset_column")
    @classmethod
    def _validate_named_column(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("column names must be non-empty strings.")
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.label_column in self.feature_columns:
            raise ValueError("label_column must not also be a feature column.")
        reserved = {self.timestamp_column, self.asset_column}
        if self.label_column in reserved or any(
            feature in reserved for feature in self.feature_columns
        ):
            raise ValueError(
                "research columns must not reuse timestamp or asset columns."
            )


def validate_compute_panel(
    frame: pd.DataFrame,
    spec: ComputePanelSpec,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Validate and normalize a long-format compute panel."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if not isinstance(spec, ComputePanelSpec):
        raise TypeError("spec must be a ComputePanelSpec.")

    required_columns = [
        spec.timestamp_column,
        spec.asset_column,
        *spec.feature_columns,
        spec.label_column,
    ]
    missing_columns = [
        column for column in required_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"panel is missing columns: {missing_columns}.")
    missing_segments = [column for column in segment_columns if column not in frame]
    if missing_segments:
        raise ValueError(f"segment columns are missing: {missing_segments}.")

    result = frame.copy()
    result[spec.timestamp_column] = pd.to_datetime(result[spec.timestamp_column])
    if result[spec.timestamp_column].isna().any():
        raise ValueError("timestamp column contains missing timestamps.")
    if result[spec.asset_column].isna().any():
        raise ValueError("asset column contains missing assets.")
    result[spec.asset_column] = result[spec.asset_column].astype(str)

    key_columns = [*segment_columns, spec.timestamp_column, spec.asset_column]
    duplicated = result.duplicated(key_columns)
    if duplicated.any():
        if segment_columns:
            raise ValueError("panel rows must have unique segment-timestamp-asset keys.")
        raise ValueError("panel rows must have unique timestamp-asset keys.")

    for column in (*spec.feature_columns, spec.label_column):
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise TypeError(f"column {column!r} must be numeric.")

    return result.sort_values(key_columns, kind="mergesort").reset_index(drop=True)


__all__ = ["ComputePanelSpec", "validate_compute_panel"]
