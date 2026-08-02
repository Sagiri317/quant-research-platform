"""Generic time-series contiguity helpers."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence

import pandas as pd

from quant_research_platform.contracts.validation import strict_timestamp_series


def contiguous_segment_ids(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = ("asset_id",),
    timestamp_column: str = "timestamp",
    expected_interval: pd.Timedelta,
) -> pd.Series:
    """Return deterministic contiguous segment ids aligned to input rows."""

    interval = _strict_expected_interval(expected_interval)
    if frame.empty:
        return pd.Series([], index=frame.index, dtype="int64")
    missing = [
        column for column in (*group_columns, timestamp_column) if column not in frame
    ]
    if missing:
        raise ValueError("contiguity_columns_missing")
    work = frame.loc[:, [*group_columns, timestamp_column]].copy()
    _validate_group_key_frame(work, group_columns)
    work["_qrp_original_position"] = range(len(frame))
    work[timestamp_column] = strict_timestamp_series(
        work[timestamp_column], error="invalid_grouped_timestamp_values"
    )
    work = work.sort_values(
        [*group_columns, timestamp_column, "_qrp_original_position"],
        kind="mergesort",
    )
    segment_values = pd.Series(index=work["_qrp_original_position"], dtype="int64")
    next_segment_id = 0
    for _, rows in work.groupby(list(group_columns), sort=False, dropna=False):
        deltas = rows[timestamp_column].diff()
        resets = deltas.isna() | deltas.ne(interval)
        local_segments = resets.cumsum().astype("int64") - 1 + next_segment_id
        segment_values.loc[rows["_qrp_original_position"]] = local_segments.to_numpy()
        next_segment_id = int(local_segments.max()) + 1
    aligned = segment_values.sort_index()
    return pd.Series(aligned.to_numpy(dtype="int64"), index=frame.index)


def iter_grouped_timestamps(
    panel_keys: pd.DataFrame | Mapping[object, Sequence[pd.Timestamp]],
    *,
    group_columns: tuple[str, ...] = ("asset_id",),
    timestamp_column: str = "timestamp",
) -> Iterable[pd.Series]:
    """Yield sorted timestamp groups from a frame or mapping."""

    if isinstance(panel_keys, pd.DataFrame):
        if panel_keys.empty:
            return
        missing = [
            column
            for column in (*group_columns, timestamp_column)
            if column not in panel_keys
        ]
        if missing:
            raise ValueError("panel_key_columns_missing")
        keys = panel_keys.loc[:, [*group_columns, timestamp_column]].copy()
        _validate_group_key_frame(keys, group_columns)
        keys[timestamp_column] = strict_timestamp_series(
            keys[timestamp_column], error="invalid_grouped_timestamp_values"
        )
        if keys.duplicated([*group_columns, timestamp_column]).any():
            raise ValueError("duplicate_panel_keys")
        if keys.empty:
            return
        keys = keys.sort_values([*group_columns, timestamp_column], kind="mergesort")
        for _, group in keys.groupby(list(group_columns), sort=False):
            yield group[timestamp_column].reset_index(drop=True)
        return
    for key, timestamps in panel_keys.items():
        _strict_group_key_value(key)
        if timestamps is None:
            raise ValueError("invalid_grouped_timestamp_values")
        dates = strict_timestamp_series(
            pd.Series(tuple(timestamps)), error="invalid_grouped_timestamp_values"
        )
        if dates.duplicated().any():
            raise ValueError("duplicate_panel_keys")
        dates = dates.sort_values(kind="mergesort").reset_index(drop=True)
        if not dates.empty:
            yield dates


def _strict_expected_interval(expected_interval: object) -> pd.Timedelta:
    if not isinstance(expected_interval, pd.Timedelta):
        raise ValueError("invalid_expected_interval")
    if expected_interval <= pd.Timedelta(0):
        raise ValueError("invalid_expected_interval")
    return expected_interval


def _validate_group_key_frame(
    frame: pd.DataFrame,
    group_columns: tuple[str, ...],
) -> None:
    for column in group_columns:
        for value in frame[column].tolist():
            _strict_group_key_value(value)


def _strict_group_key_value(value: object) -> None:
    if isinstance(value, tuple):
        for part in value:
            _strict_group_key_value(part)
        return
    if (
        _is_null_group_key(value)
        or isinstance(value, (list, dict, set))
        or not isinstance(value, Hashable)
    ):
        raise ValueError("invalid_grouped_timestamp_keys")


def _is_null_group_key(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


__all__ = ["contiguous_segment_ids", "iter_grouped_timestamps"]
