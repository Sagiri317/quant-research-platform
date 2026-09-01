"""Trading-session rebalance schedules shared by markets."""

from __future__ import annotations

import pandas as pd


def rebalance_dates(
    sessions: pd.DatetimeIndex,
    schedule: str,
) -> pd.DatetimeIndex:
    if schedule not in {
        "daily",
        "weekly-last-trading-day",
        "monthly-last-trading-day",
    }:
        raise ValueError("unsupported_rebalance_schedule")
    values = pd.DatetimeIndex(sessions).drop_duplicates().sort_values()
    if schedule == "daily":
        return values
    period = values.to_period("W" if schedule.startswith("weekly") else "M")
    return values[~period.duplicated(keep="last")]
