"""Generic forward-return label helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.time import TimeWindow


def add_forward_return_labels(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 5, 10, 20, 60, 120),
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
    execution_price_column: str = "adj_open",
) -> pd.DataFrame:
    """Add per-asset valid daily bar next-open to horizon-open labels."""

    result = frame.sort_values([asset_column, date_column]).copy()
    price = pd.to_numeric(result[execution_price_column], errors="coerce")
    grouped_price = price.groupby(result[asset_column], sort=False)

    entry_price = grouped_price.shift(-1)
    entry_price = entry_price.where(np.isfinite(entry_price) & entry_price.ne(0.0))
    for horizon in horizons:
        exit_price = grouped_price.shift(-(horizon + 1))
        exit_price = exit_price.where(np.isfinite(exit_price))
        label = exit_price / entry_price - 1.0
        result[f"forward_return_{horizon}d"] = label.replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return result


def forward_return_valid_bar_read_window(
    panel_keys: pd.DataFrame,
    *,
    read_window: TimeWindow,
    core_sessions: tuple[pd.Timestamp, ...],
    lookahead_bars: int,
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
) -> TimeWindow:
    if lookahead_bars <= 0 or panel_keys.empty:
        return read_window
    keys = panel_keys.loc[:, [asset_column, date_column]].copy()
    keys[date_column] = pd.to_datetime(keys[date_column], errors="coerce")
    keys = keys.dropna(subset=[asset_column, date_column])
    if keys.empty:
        return read_window
    read_start = pd.Timestamp(read_window.start)
    keys = keys.loc[keys[date_column].dt.normalize().ge(read_start)].sort_values(
        [asset_column, date_column]
    )
    if keys.empty:
        return read_window
    core_dates = {session.date() for session in core_sessions}
    core_rows = keys.loc[keys[date_column].dt.date.isin(core_dates)]
    if core_rows.empty:
        return read_window
    ends = [pd.Timestamp(read_window.end)]
    for _, asset_rows in keys.groupby(asset_column, sort=False):
        asset_core_positions = [
            position
            for position, timestamp in enumerate(asset_rows[date_column])
            if timestamp.date() in core_dates
        ]
        if not asset_core_positions:
            continue
        read_position = min(
            len(asset_rows) - 1,
            asset_core_positions[-1] + lookahead_bars,
        )
        ends.append(asset_rows.iloc[read_position][date_column].normalize())
    end = max(ends).date()
    if end <= read_window.end:
        return read_window
    return TimeWindow(read_window.start, end)


__all__ = ["add_forward_return_labels", "forward_return_valid_bar_read_window"]
