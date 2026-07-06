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
    work["_qrp_original_index"] = frame.index
    work[timestamp_column] = pd.to_datetime(work[timestamp_column], errors="coerce")
    work = work.sort_values(
        [*group_columns, timestamp_column, "_qrp_original_index"],
        kind="mergesort",
    )
    segment_values = pd.Series(index=work.index, dtype="int64")
    next_segment_id = 0
    for _, rows in work.groupby(list(group_columns), sort=False, dropna=False):
        deltas = rows[timestamp_column].diff()
        resets = deltas.isna() | deltas.ne(expected_interval)
        local_segments = resets.cumsum().astype("int64") - 1 + next_segment_id
        segment_values.loc[rows.index] = local_segments
        next_segment_id = int(local_segments.max()) + 1
    aligned = segment_values.reindex(frame.index)
    return aligned.astype("int64")


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
        keys[timestamp_column] = pd.to_datetime(keys[timestamp_column], errors="coerce")
        keys = keys.dropna(subset=[*group_columns, timestamp_column])
        if keys.empty:
            return
        keys = keys.sort_values([*group_columns, timestamp_column], kind="mergesort")
        for _, group in keys.groupby(list(group_columns), sort=False):
            yield group[timestamp_column].reset_index(drop=True)
        return
    for timestamps in panel_keys.values():
        dates = pd.Series(pd.to_datetime(tuple(timestamps), errors="coerce"))
        dates = dates.dropna().sort_values(kind="mergesort").reset_index(drop=True)
        if not dates.empty:
            yield dates


__all__ = ["contiguous_segment_ids", "iter_grouped_timestamps"]
