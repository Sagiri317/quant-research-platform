"""Generic time-series contiguity helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd


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
    work[timestamp_column] = _strict_timestamps(work[timestamp_column])
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
        keys[timestamp_column] = _strict_timestamps(keys[timestamp_column])
        if keys.empty:
            return
        keys = keys.sort_values([*group_columns, timestamp_column], kind="mergesort")
        for _, group in keys.groupby(list(group_columns), sort=False):
            yield group[timestamp_column].reset_index(drop=True)
        return
    for key, timestamps in panel_keys.items():
        _validate_mapping_group_key(key)
        dates = pd.Series(_strict_timestamps(pd.Series(tuple(timestamps))))
        dates = dates.sort_values(kind="mergesort").reset_index(drop=True)
        if not dates.empty:
            yield dates


def _strict_timestamps(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("invalid_grouped_timestamp_values")
    return timestamps


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
    if frame.loc[:, list(group_columns)].isna().any(axis=None):
        raise ValueError("invalid_grouped_timestamp_keys")


def _validate_mapping_group_key(key: object) -> None:
    if isinstance(key, tuple):
        invalid = any(_is_null_group_key(part) for part in key)
    else:
        invalid = _is_null_group_key(key)
    if invalid:
        raise ValueError("invalid_grouped_timestamp_keys")


def _is_null_group_key(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


__all__ = ["contiguous_segment_ids", "iter_grouped_timestamps"]
