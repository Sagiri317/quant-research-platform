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

    if expected_interval <= pd.Timedelta(0):
        raise ValueError("invalid_expected_interval")
    if frame.empty:
        return pd.Series([], index=frame.index, dtype="int64")
    missing = [
        column for column in (*group_columns, timestamp_column) if column not in frame
    ]
    if missing:
        raise ValueError("contiguity_columns_missing")
    work = frame.loc[:, [*group_columns, timestamp_column]].copy()
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
        resets = deltas.isna() | deltas.ne(expected_interval)
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
        keys[timestamp_column] = _strict_timestamps(keys[timestamp_column])
        keys = keys.dropna(subset=[*group_columns])
        if keys.empty:
            return
        keys = keys.sort_values([*group_columns, timestamp_column], kind="mergesort")
        for _, group in keys.groupby(list(group_columns), sort=False):
            yield group[timestamp_column].reset_index(drop=True)
        return
    for timestamps in panel_keys.values():
        dates = pd.Series(_strict_timestamps(pd.Series(tuple(timestamps))))
        dates = dates.sort_values(kind="mergesort").reset_index(drop=True)
        if not dates.empty:
            yield dates


def _strict_timestamps(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("invalid_grouped_timestamp_values")
    return timestamps


__all__ = ["contiguous_segment_ids", "iter_grouped_timestamps"]
