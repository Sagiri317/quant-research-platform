"""Read-window helpers for price-volume factor inputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd

from quant_research_platform.contracts.time import TimeWindow


def price_volume_valid_bar_read_window(
    panel_keys: pd.DataFrame | Mapping[object, Sequence[pd.Timestamp]],
    *,
    read_window: TimeWindow,
    core_sessions: tuple[pd.Timestamp, ...],
    lookback_bars: int,
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
) -> TimeWindow:
    if lookback_bars <= 0:
        return read_window
    key_groups = tuple(
        _iter_panel_key_groups(
            panel_keys,
            asset_column=asset_column,
            date_column=date_column,
        )
    )
    if not key_groups:
        return read_window
    read_end = pd.Timestamp(read_window.end)
    core_dates = {session.date() for session in core_sessions}
    starts = [pd.Timestamp(read_window.start)]
    saw_core = False
    for asset_dates in key_groups:
        asset_rows = asset_dates.loc[asset_dates.dt.normalize().le(read_end)]
        if asset_rows.empty:
            continue
        asset_core_positions = [
            position
            for position, timestamp in enumerate(asset_rows)
            if timestamp.date() in core_dates
        ]
        if not asset_core_positions:
            continue
        saw_core = True
        read_position = max(0, asset_core_positions[0] - lookback_bars)
        starts.append(asset_rows.iloc[read_position].normalize())
    if not saw_core:
        return read_window
    start = min(starts).date()
    if start >= read_window.start:
        return read_window
    return TimeWindow(start, read_window.end)


def _iter_panel_key_groups(
    panel_keys: pd.DataFrame | Mapping[object, Sequence[pd.Timestamp]],
    *,
    asset_column: str,
    date_column: str,
) -> Iterable[pd.Series]:
    if isinstance(panel_keys, pd.DataFrame):
        if panel_keys.empty:
            return
        keys = panel_keys.loc[:, [asset_column, date_column]].copy()
        keys[date_column] = pd.to_datetime(keys[date_column], errors="coerce")
        keys = keys.dropna(subset=[asset_column, date_column])
        if keys.empty:
            return
        keys = keys.sort_values([asset_column, date_column])
        for _, asset_rows in keys.groupby(asset_column, sort=False):
            yield asset_rows[date_column].reset_index(drop=True)
        return
    for timestamps in panel_keys.values():
        dates = pd.Series(pd.to_datetime(tuple(timestamps), errors="coerce"))
        dates = dates.dropna().sort_values(kind="mergesort").reset_index(drop=True)
        if not dates.empty:
            yield dates

__all__ = ["price_volume_valid_bar_read_window"]
