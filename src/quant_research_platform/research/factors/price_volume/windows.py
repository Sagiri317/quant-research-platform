"""Read-window helpers for price-volume factor inputs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import pandas as pd

from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.time_series.contiguity import (
    iter_grouped_timestamps,
)

_PRICE_VOLUME_WINDOW_SUFFIXES: dict[str, DataFrequency] = {
    "d": DataFrequency.DAILY,
    "m": DataFrequency.MINUTE,
    "tick": DataFrequency.TICK,
}
_WINDOWED_FACTOR_PATTERN = re.compile(
    r"^(?P<prefix>[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*)_"
    r"(?P<window>[1-9]\d*)(?P<suffix>d|m|tick)$"
)


def parse_price_volume_windowed_factor_name(
    name: str,
) -> tuple[str, int, DataFrequency] | None:
    match = _WINDOWED_FACTOR_PATTERN.fullmatch(name)
    if match is None:
        return None
    return (
        match.group("prefix"),
        int(match.group("window")),
        _PRICE_VOLUME_WINDOW_SUFFIXES[match.group("suffix")],
    )


def required_windows(factor_id: str) -> tuple[int, ...]:
    parsed = parse_price_volume_windowed_factor_name(factor_id)
    if parsed is None:
        return ()
    _prefix, window, _frequency = parsed
    return (window,)


def lookback_window(factor_id: str) -> int | None:
    windows = required_windows(factor_id)
    if not windows:
        return None
    return max(windows)


def windows_from_requested_factors(requested: set[str]) -> tuple[int, ...]:
    values: set[int] = set()
    for factor in requested:
        parsed = parse_price_volume_windowed_factor_name(factor)
        if parsed is None:
            continue
        _prefix, window, frequency = parsed
        if frequency is DataFrequency.DAILY:
            values.add(window)
    return tuple(sorted(values))


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

__all__ = [
    "price_volume_valid_bar_read_window",
]
