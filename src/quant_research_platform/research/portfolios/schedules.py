"""Trading-session rebalance schedules shared by markets."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.portfolios import REBALANCE_SCHEDULES
from quant_research_platform.contracts.time import TimeWindow


def holding_intervals(sessions: pd.DatetimeIndex, window: TimeWindow) -> pd.DataFrame:
    """Actual close decisions and their next-open to following-open intervals."""
    records = [
        (sessions[index], sessions[index + 1], sessions[index + 2])
        for index in range(len(sessions) - 2)
        if sessions[index] >= pd.Timestamp(window.start)
        and sessions[index + 2] <= pd.Timestamp(window.end)
    ]
    return pd.DataFrame(records, columns=("decision_at", "entry_at", "end_at"))


def rebalance_dates(
    sessions: pd.DatetimeIndex,
    schedule: str,
) -> pd.DatetimeIndex:
    _validate_schedule(schedule)
    values = pd.DatetimeIndex(sessions).drop_duplicates().sort_values()
    if values.empty or schedule == "daily":
        return values
    return values[
        [
            is_rebalance_session(timestamp, next_timestamp, schedule)
            for timestamp, next_timestamp in zip(values[:-1], values[1:], strict=True)
        ]
        + [True]
    ]


def is_rebalance_session(
    timestamp: pd.Timestamp,
    next_timestamp: pd.Timestamp,
    schedule: str,
) -> bool:
    """Return whether a session is the last session in its schedule period."""

    _validate_schedule(schedule)
    current = pd.Timestamp(timestamp).normalize()
    following = pd.Timestamp(next_timestamp).normalize()
    if following <= current:
        raise ValueError("rebalance_session_order_invalid")
    if schedule == "daily":
        return True
    frequency = "W" if schedule == "weekly-last-trading-day" else "M"
    return current.to_period(frequency) != following.to_period(frequency)


def _validate_schedule(schedule: str) -> None:
    if type(schedule) is not str or schedule not in REBALANCE_SCHEDULES:
        raise ValueError("unsupported_rebalance_schedule")
