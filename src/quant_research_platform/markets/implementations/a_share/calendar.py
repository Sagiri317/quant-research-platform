"""Internal A-share trading calendar helpers."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

CURRENT_MARKET_TYPES = frozenset({1, 4, 16, 32})
NORMALIZED_CALENDAR_COLUMNS = frozenset(
    {"calendar_date", "source_market_type", "is_trading_day"}
)


def a_share_unified_trading_sessions(
    calendar: pd.DataFrame,
    *,
    market_types: Iterable[int] = CURRENT_MARKET_TYPES,
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


def warmup_start_date_for_window(
    calendar: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
    max_window: int,
) -> pd.Timestamp:
    """Return the first warmup input session for a requested start date."""

    if max_window < 0:
        raise ValueError("invalid_factor_warmup_window")
    normalized = _require_normalized_calendar(calendar)
    if bool(
        normalized.duplicated(["source_market_type", "calendar_date"], keep=False).any()
    ):
        raise ValueError("duplicate_trading_calendar_key")
    sessions = a_share_unified_trading_sessions(normalized)
    if sessions.empty:
        raise ValueError("trading_calendar_no_unified_sessions")
    requested_start = pd.Timestamp(start_date).normalize()
    start_index = int(sessions.searchsorted(requested_start, side="left"))
    if start_index >= len(sessions):
        raise ValueError("trading_calendar_start_date_out_of_range")
    warmup_start_index = max(0, start_index - max_window)
    return pd.Timestamp(sessions[warmup_start_index])


def compute_listing_age_trading_days(
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    observation_date_column: str,
    listing_date_column: str,
    market_type_column: str | None = None,
) -> pd.Series:
    """Count open trading sessions from listing date through observation date."""

    for column in (observation_date_column, listing_date_column):
        if column not in frame.columns:
            raise ValueError(f"missing_listing_age_column: {column}")
    normalized = _require_normalized_calendar(calendar)
    if bool(
        normalized.duplicated(["source_market_type", "calendar_date"], keep=False).any()
    ):
        raise ValueError("duplicate_trading_calendar_key")

    observation_dates = pd.to_datetime(
        frame[observation_date_column],
        errors="coerce",
    ).dt.normalize()
    listing_dates = pd.to_datetime(
        frame[listing_date_column],
        errors="coerce",
    ).dt.normalize()
    valid = (
        listing_dates.notna()
        & observation_dates.notna()
        & observation_dates.ge(listing_dates)
    )
    result = pd.Series(0, index=frame.index, dtype="int64")
    if not bool(valid.any()):
        return result

    assigned = pd.Series(False, index=frame.index, dtype=bool)
    market_types = (
        pd.to_numeric(frame[market_type_column], errors="coerce")
        if market_type_column is not None and market_type_column in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="Float64")
    )

    available_market_types = pd.to_numeric(
        normalized.loc[normalized["is_trading_day"].astype(bool), "source_market_type"],
        errors="coerce",
    ).dropna()
    for market_type in sorted(int(value) for value in available_market_types.unique()):
        market_mask = valid & market_types.eq(market_type)
        if not bool(market_mask.any()):
            continue
        sessions = market_type_trading_sessions(normalized, market_type)
        if sessions.empty:
            continue
        result.loc[market_mask] = _count_sessions_between(
            sessions,
            listing_dates.loc[market_mask],
            observation_dates.loc[market_mask],
        )
        assigned.loc[market_mask] = True

    fallback_mask = valid & ~assigned
    if bool(fallback_mask.any()):
        unified_sessions = a_share_unified_trading_sessions(normalized)
        if not unified_sessions.empty:
            result.loc[fallback_mask] = _count_sessions_between(
                unified_sessions,
                listing_dates.loc[fallback_mask],
                observation_dates.loc[fallback_mask],
            )
    return result


def _require_normalized_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(NORMALIZED_CALENDAR_COLUMNS.difference(calendar.columns))
    if missing:
        raise ValueError("calendar_missing_normalized_columns: " + ", ".join(missing))
    result = calendar.copy()
    result["calendar_date"] = pd.to_datetime(result["calendar_date"], errors="raise")
    result["calendar_date"] = result["calendar_date"].dt.normalize()
    return result


def _sorted_unique_sessions(values: pd.Series) -> pd.DatetimeIndex:
    dates = pd.to_datetime(values, errors="raise").dt.normalize()
    return pd.DatetimeIndex(sorted(dates.dropna().unique()))


def _count_sessions_between(
    sessions: pd.DatetimeIndex,
    listing_dates: pd.Series,
    observation_dates: pd.Series,
) -> pd.Series:
    left = sessions.searchsorted(listing_dates.to_numpy(dtype="datetime64[ns]"), "left")
    right = sessions.searchsorted(
        observation_dates.to_numpy(dtype="datetime64[ns]"),
        "right",
    )
    ages = (right - left).clip(min=0)
    return pd.Series(ages, index=listing_dates.index, dtype="int64")


__all__ = [
    "CURRENT_MARKET_TYPES",
    "a_share_unified_trading_sessions",
    "compute_listing_age_trading_days",
    "market_type_trading_sessions",
    "warmup_start_date_for_window",
]
