"""Research sample filtering implementation."""

from __future__ import annotations

from contextlib import nullcontext

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype

from quant_research_platform.contracts.filtering import (
    filter_definition,
    required_filter_columns,
)
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
)


def apply_filter_mode(
    frame: pd.DataFrame,
    *,
    mode: str,
    metadata_key: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    columns = required_filter_columns(mode)
    policy = current_runtime_resource_policy()
    if mode == "unfiltered":
        filtered = frame
    else:
        mask_bytes = int(frame.index.memory_usage(deep=True)) + len(frame)
        output_bytes = pandas_frame_payload_bytes(frame)
        working_bytes = max(
            output_bytes + mask_bytes,
            2 * mask_bytes,
        )
        if policy is not None and (
            policy.used_memory_bytes + working_bytes
            > policy.memory_budget_bytes
        ):
            raise ValueError("research_filter_working_set_exceeds_task_memory_budget")
        reservation = (
            policy.reserve("research.filter", working_bytes)
            if policy is not None
            else nullcontext()
        )
        with reservation:
            keep = _validated_bool_mask(frame, columns[0])
            for column in columns[1:]:
                mask = _validated_bool_mask(frame, column)
                keep &= mask
            if len(columns) > 1:
                del mask
            filtered = frame.loc[keep]
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
        return series.eq(True)
    if is_integer_dtype(series):
        invalid = ~series.isin((0, 1))
        if not bool(invalid.any()):
            return series.eq(1)
    raise ValueError(f"invalid_filter_column_dtype: {column}")


__all__ = ["apply_filter_mode"]
