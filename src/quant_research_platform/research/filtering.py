"""Research sample filtering implementation."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)


def apply_filter_mode(
    frame: pd.DataFrame,
    *,
    mode: str,
    metadata_key: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = required_filter_columns(mode)
    masks = {column: _validated_bool_mask(frame, column) for column in columns}
    if mode == "eligible":
        keep = masks["eligible"]
    elif mode == "tradable":
        keep = masks["eligible"] & masks["can_enter"] & masks["can_exit"]
    elif mode == "unfiltered":
        keep = pd.Series(True, index=frame.index)
    else:
        raise ValueError("unsupported_filter_mode")
    filtered = frame.loc[keep].copy()
    metadata = {
        metadata_key: mode,
        "filter_columns_used": list(columns),
        "filter_definition": filter_definition(mode),
        "rows_input": int(len(frame)),
        "rows_after_filter_mode": int(len(filtered)),
    }
    return filtered, metadata


def _validated_bool_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"missing_filter_columns: {column}")
    series = frame[column]
    if bool(series.isna().any()):
        raise ValueError(f"null_filter_column_values: {column}")
    if is_bool_dtype(series):
        return series.astype(bool)
    if is_integer_dtype(series):
        values = set(series.unique().tolist())
        if values.issubset({0, 1}):
            return series.astype(bool)
    raise ValueError(f"invalid_filter_column_dtype: {column}")


__all__ = ["apply_filter_mode"]
