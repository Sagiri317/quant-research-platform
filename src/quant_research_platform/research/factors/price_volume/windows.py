"""Read-window helpers for price-volume factor inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.time_series.contiguity import (
    iter_grouped_timestamps,
)


def price_volume_valid_bar_read_window(
    panel_keys: pd.DataFrame | Mapping[object, Sequence[pd.Timestamp]],
    *,
    read_window: TimeWindow,
    core_sessions: tuple[pd.Timestamp, ...],
    lookback_bars: int,
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
) -> TimeWindow:
    if (
        not isinstance(lookback_bars, int)
        or isinstance(lookback_bars, bool)
        or lookback_bars < 0
    ):
        raise ValueError("invalid_price_volume_lookback_bars")
    if lookback_bars == 0:
        return read_window
    key_groups = tuple(
        iter_grouped_timestamps(
            panel_keys,
            group_columns=(asset_column,),
            timestamp_column=date_column,
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
