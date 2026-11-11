"""Derive research eligibility from the facts required by each declared sample scope."""

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
    if mode == "unfiltered":
        filtered = frame
    else:
        keep = _validated_bool_mask(frame, columns[0])
        for column in columns[1:]:
            mask = _validated_bool_mask(frame, column)
            keep &= mask

        filtered = frame.loc[keep]

    return filtered, {
        metadata_key: mode,
        "filter_columns_used": list(columns),
        "filter_definition": filter_definition(mode),
        "rows_input": int(len(frame)),
        "rows_after_filter_mode": int(len(filtered)),
    }


def _validated_bool_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"missing_filter_columns: {column}")
    series = frame[column]
    if is_bool_dtype(series):
        return series.fillna(False)
    if is_integer_dtype(series):
        invalid = ~series.dropna().isin((0, 1))
        if not bool(invalid.any()):
            return series.eq(1).fillna(False)
    raise ValueError(f"invalid_filter_column_dtype: {column}")
