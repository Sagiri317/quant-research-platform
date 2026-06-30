"""Generic forward-return label helpers.

Forward-return horizons use per-asset valid bars. They do not imply
calendar/session windows and do not fill missing or suspended rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_research_platform.contracts.data import (
    DataFrequency,
    horizon_unit_suffix,
)
from quant_research_platform.contracts.time import TimeWindow


def add_forward_return_labels(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    frequency: DataFrequency | str = DataFrequency.DAILY,
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
    execution_price_column: str = "adj_open",
) -> pd.DataFrame:
    """Add per-asset valid-bar next-open to horizon-open labels."""

    result = frame.sort_values([asset_column, date_column]).copy()
    price = pd.to_numeric(result[execution_price_column], errors="coerce")
    grouped_price = price.groupby(result[asset_column], sort=False)

    entry_price = grouped_price.shift(-1)
    entry_price = entry_price.where(np.isfinite(entry_price) & entry_price.ne(0.0))
    for horizon in horizons:
        exit_price = grouped_price.shift(-(horizon + 1))
        exit_price = exit_price.where(np.isfinite(exit_price))
        label = exit_price / entry_price - 1.0
        result[forward_return_label_column(horizon, frequency)] = label.replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return result


def forward_return_label_column(
    horizon: int,
    frequency: DataFrequency | str = DataFrequency.DAILY,
) -> str:
    return f"forward_return_{horizon}{horizon_unit_suffix(frequency)}"


def forward_return_valid_bar_read_window(
    panel_keys: pd.DataFrame | Mapping[object, Sequence[pd.Timestamp]],
    *,
    read_window: TimeWindow,
    core_sessions: tuple[pd.Timestamp, ...],
    lookahead_bars: int,
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
) -> TimeWindow:
    if lookahead_bars <= 0:
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
    read_start = pd.Timestamp(read_window.start)
    core_dates = {session.date() for session in core_sessions}
    ends = [pd.Timestamp(read_window.end)]
    saw_core = False
    for asset_dates in key_groups:
        asset_rows = asset_dates.loc[asset_dates.dt.normalize().ge(read_start)]
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
        read_position = min(
            len(asset_rows) - 1,
            asset_core_positions[-1] + lookahead_bars,
        )
        ends.append(asset_rows.iloc[read_position].normalize())
    if not saw_core:
        return read_window
    end = max(ends).date()
    if end <= read_window.end:
        return read_window
    return TimeWindow(read_window.start, end)


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


__all__ = [
    "add_forward_return_labels",
    "forward_return_label_column",
    "forward_return_valid_bar_read_window",
]
