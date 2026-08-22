"""Compute-local panel specs and validation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_hashable_key_series,
    strict_string_series,
    strict_timestamp_series,
)


class ComputePanelSpec(BaseModel):
    """Column contract for long-format compute panels."""

    model_config = ConfigDict(frozen=True)

    signal_columns: tuple[str, ...]
    label_column: str
    timestamp_column: str = "timestamp"
    asset_column: str = "asset"

    @field_validator("signal_columns")
    @classmethod
    def _validate_signal_columns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("signal_columns must contain at least one column.")
        if any(not isinstance(column, str) or not column for column in value):
            raise ValueError("signal_columns must contain non-empty strings.")
        if len(set(value)) != len(value):
            raise ValueError("signal_columns must not contain duplicates.")
        return value

    @field_validator("label_column", "timestamp_column", "asset_column")
    @classmethod
    def _validate_named_column(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("column names must be non-empty strings.")
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.label_column in self.signal_columns:
            raise ValueError("label_column must not also be a signal column.")
        reserved = {self.timestamp_column, self.asset_column}
        if self.label_column in reserved or any(
            signal in reserved for signal in self.signal_columns
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
        *spec.signal_columns,
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
    if len(set(segment_columns)) != len(segment_columns):
        raise ValueError("segment columns must not contain duplicates.")
    overlap = set(segment_columns).intersection(required_columns)
    if overlap:
        raise ValueError(
            "segment columns must not overlap panel columns: "
            + ", ".join(sorted(overlap))
        )

    result = frame.copy()
    result[spec.timestamp_column] = strict_timestamp_series(
        result[spec.timestamp_column],
        error="timestamp column contains missing timestamps.",
    )
    result[spec.asset_column] = strict_string_series(
        result[spec.asset_column],
        error="asset column contains missing assets.",
        strip=False,
    )
    for column in segment_columns:
        result[column] = strict_hashable_key_series(
            result[column],
            error=f"segment column {column!r} contains invalid keys.",
        )

    key_columns = [*segment_columns, spec.timestamp_column, spec.asset_column]
    duplicated = result.duplicated(key_columns)
    if duplicated.any():
        if segment_columns:
            raise ValueError(
                "panel rows must have unique segment-timestamp-asset keys."
            )
        raise ValueError("panel rows must have unique timestamp-asset keys.")

    for column in (*spec.signal_columns, spec.label_column):
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise TypeError(f"column {column!r} must be numeric.")
        result[column] = strict_finite_numeric_series(
            result[column],
            error=f"column {column!r} must contain finite numeric values.",
            allow_null=True,
        ).astype("float64")

    return result.sort_values(key_columns, kind="mergesort").reset_index(drop=True)


__all__ = ["ComputePanelSpec", "validate_compute_panel"]
