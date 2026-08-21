"""Research sample filtering implementation."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

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
    with _filter_mode_scope(
        frame,
        mode=mode,
        metadata_key=metadata_key,
    ) as result:
        return result


@contextmanager
def _filter_mode_scope(
    frame: pd.DataFrame,
    *,
    mode: str,
    metadata_key: str,
) -> Iterator[tuple[pd.DataFrame, dict[str, object]]]:
    columns = required_filter_columns(mode)
    policy = current_runtime_resource_policy()
    lease = None
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
        if policy is not None:
            lease = policy.lease("research.filter", working_bytes)
        try:
            keep = _validated_bool_mask(frame, columns[0])
            for column in columns[1:]:
                mask = _validated_bool_mask(frame, column)
                keep &= mask
            if len(columns) > 1:
                del mask
            filtered = frame.loc[keep]
            del keep
            if lease is not None:
                lease.shrink_to(pandas_frame_payload_bytes(filtered))
                lease.transfer("research.filter.output")
        except BaseException as exc:
            if lease is not None:
                try:
                    lease.close()
                except BaseException as close_error:
                    exc.add_note(
                        "research filter lease release also failed: "
                        f"{type(close_error).__name__}"
                    )
                lease = None
            raise
    try:
        metadata = {
            metadata_key: mode,
            "filter_columns_used": list(columns),
            "filter_definition": filter_definition(mode),
            "rows_input": int(len(frame)),
            "rows_after_filter_mode": int(len(filtered)),
        }
        yield filtered, metadata
    finally:
        primary = sys.exception()
        if lease is not None:
            try:
                lease.close()
            except BaseException as close_error:
                if primary is None:
                    raise
                primary.add_note(
                    "research filter lease release also failed: "
                    f"{type(close_error).__name__}"
                )


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
