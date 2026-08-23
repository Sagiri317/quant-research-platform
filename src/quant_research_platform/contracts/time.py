"""Time value marker types for QRP 0.1 contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta


@dataclass(frozen=True)
class TimeWindow:
    """Closed logical time interval used for dataset reads and inspection."""

    start: date | datetime
    end: date | datetime

    def __post_init__(self) -> None:
        start = _coerce_bound(self.start)
        end = _coerce_bound(self.end)
        if isinstance(start, datetime) != isinstance(end, datetime):
            raise ValueError("mixed_time_window_bounds")
        if start > end:
            raise ValueError("invalid_time_window")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


def is_intraday_time_window(time_window: TimeWindow) -> bool:
    return isinstance(time_window.start, datetime)


def require_date_window(time_window: TimeWindow, *, context: str) -> None:
    if is_intraday_time_window(time_window):
        raise ValueError(f"{context}_requires_date_time_window")


def time_window_date_bounds(time_window: TimeWindow) -> tuple[date, date]:
    start = (
        time_window.start.date()
        if isinstance(time_window.start, datetime)
        else time_window.start
    )
    end = (
        time_window.end.date()
        if isinstance(time_window.end, datetime)
        else time_window.end
    )
    return start, end


def time_window_timestamp_interval(
    time_window: TimeWindow,
) -> tuple[datetime, datetime, bool]:
    """Return timestamp filter bounds and whether the upper bound is closed."""

    if isinstance(time_window.start, datetime) and isinstance(
        time_window.end,
        datetime,
    ):
        return time_window.start, time_window.end, True
    start = datetime.combine(time_window.start, time.min)
    end = datetime.combine(time_window.end + timedelta(days=1), time.min)
    return start, end, False


def _coerce_bound(value: date | datetime | str) -> date | datetime:
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            if "T" in value or " " in value:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return _normalize_datetime(parsed)
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("invalid_time_window_bound") from exc
    raise ValueError("invalid_time_window_bound")


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)
