"""Time value marker types for QRP 0.1 contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import NewType

Timestamp = NewType("Timestamp", datetime)
SessionDate = NewType("SessionDate", date)
AvailableAt = NewType("AvailableAt", datetime)


@dataclass(frozen=True)
class TimeWindow:
    """Closed logical time interval used for dataset reads and inspection."""

    start: date
    end: date

    def __post_init__(self) -> None:
        start = _coerce_date(self.start)
        end = _coerce_date(self.end)
        if start > end:
            raise ValueError("invalid_time_window")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError("invalid_time_window_bound")


__all__ = ["AvailableAt", "SessionDate", "Timestamp", "TimeWindow"]
