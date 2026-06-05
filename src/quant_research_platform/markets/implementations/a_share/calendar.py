"""Internal A-share trading calendar helpers."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

CURRENT_A_SHARE_MARKET_TYPES = frozenset({1, 4, 16, 32})
NORMALIZED_CALENDAR_COLUMNS = frozenset(
    {"calendar_date", "source_market_type", "is_trading_day"}
)


def a_share_unified_trading_sessions(
    calendar: pd.DataFrame,
    *,
    market_types: Iterable[int] = CURRENT_A_SHARE_MARKET_TYPES,
) -> pd.DatetimeIndex:
    """Return sorted unique current-scope A-share trading sessions."""

    normalized = _require_normalized_calendar(calendar)
    source_market_types = pd.to_numeric(
        normalized["source_market_type"],
        errors="coerce",
    )
    mask = source_market_types.isin(set(market_types)) & normalized[
        "is_trading_day"
    ].astype(bool)
    return _sorted_unique_sessions(normalized.loc[mask, "calendar_date"])


def market_type_trading_sessions(
    calendar: pd.DataFrame,
    market_type: int,
) -> pd.DatetimeIndex:
    """Return sorted unique trading sessions for one source market type."""

    normalized = _require_normalized_calendar(calendar)
    source_market_types = pd.to_numeric(
        normalized["source_market_type"],
        errors="coerce",
    )
    mask = source_market_types.eq(int(market_type)) & normalized[
        "is_trading_day"
    ].astype(bool)
    return _sorted_unique_sessions(normalized.loc[mask, "calendar_date"])


def _require_normalized_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(NORMALIZED_CALENDAR_COLUMNS.difference(calendar.columns))
    if missing:
        raise ValueError("calendar_missing_normalized_columns: " + ", ".join(missing))
    result = calendar.copy()
    result["calendar_date"] = pd.to_datetime(result["calendar_date"], errors="raise")
    return result


def _sorted_unique_sessions(values: pd.Series) -> pd.DatetimeIndex:
    dates = pd.to_datetime(values, errors="raise").dt.normalize()
    return pd.DatetimeIndex(sorted(dates.dropna().unique()))


__all__ = [
    "CURRENT_A_SHARE_MARKET_TYPES",
    "a_share_unified_trading_sessions",
    "market_type_trading_sessions",
]
