"""Time value marker types for QRP contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

import pandas as pd


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


def is_datetime_window(time_window: TimeWindow) -> bool:
    return isinstance(time_window.start, datetime)


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


def time_value_isoformat(
    value: date | datetime,
    *,
    preserve_datetime: bool = False,
    date_only: bool = False,
) -> str:
    """Format one normalized contract time without losing intraday precision."""

    if isinstance(value, datetime):
        normalized = _normalize_datetime(value)
        if date_only or (
            not preserve_datetime
            and normalized == datetime.combine(normalized.date(), time.min)
        ):
            return normalized.date().isoformat()
        return normalized.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raise ValueError("invalid_time_value")


def is_datetime_value(value: date | datetime | str) -> bool:
    return isinstance(_coerce_bound(value), datetime)


def _coerce_bound(value: date | datetime | str) -> date | datetime:
    if value is pd.NaT:
        raise ValueError("invalid_time_window_bound")
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            if "T" in value or " " in value:
                parsed = pd.to_datetime(value, format="ISO8601")
                return _normalize_datetime(parsed)
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("invalid_time_window_bound") from exc
    raise ValueError("invalid_time_window_bound")


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def a_share_analysis_windows(
    manifest: Mapping[str, object], requested: TimeWindow
) -> tuple[TimeWindow, TimeWindow]:
    """Resolve full-development factor history and out-of-sample model dates."""
    if manifest["access_window"] != "development":
        return requested, requested
    development = TimeWindow(**manifest["development_window"])
    factor_history = TimeWindow(
        manifest["formal_dataset_window"]["start"], development.end
    )
    if pd.Timestamp(requested.end) == pd.Timestamp(development.end) and pd.Timestamp(
        requested.start
    ) in {pd.Timestamp(development.start), pd.Timestamp(factor_history.start)}:
        return factor_history, development
    return requested, requested


def crypto_execution_windows(plan, output_window, *, access_window):
    """Execution consumes the frozen final test and never validation."""
    if access_window != "holdout":
        raise PermissionError("crypto_execution_requires_frozen_test")
    boundary = TimeWindow(**plan["test"])
    if pd.Timestamp(output_window.start) < pd.Timestamp(boundary.start) or pd.Timestamp(
        output_window.end
    ) > pd.Timestamp(boundary.end):
        raise PermissionError("crypto_execution_test_window_invalid")
    return ((plan["folds"][-1]["fold_id"], output_window),)
