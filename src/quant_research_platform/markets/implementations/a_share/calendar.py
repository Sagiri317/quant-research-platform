"""Derive A-share trading sessions, listing age, and intraday session membership."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_integer_series,
    strict_timestamp_series,
)

CURRENT_MARKET_TYPES = frozenset({1, 4, 16, 32})
NORMALIZED_CALENDAR_COLUMNS = frozenset(
    {"calendar_date", "source_market_type", "is_trading_day"}
)


def unified_trading_sessions(
    calendar: pd.DataFrame,
    *,
    market_types: Iterable[int] = CURRENT_MARKET_TYPES,
) -> pd.DatetimeIndex:
    """Return sorted unique current-scope A-share trading sessions."""

    normalized = _require_normalized_calendar(calendar)
    source_market_types = strict_integer_series(
        normalized["source_market_type"],
        error="invalid_source_market_type",
    )
    trading = strict_bool_series(
        normalized["is_trading_day"],
        error="invalid_is_trading_day",
    )
    mask = source_market_types.isin(set(market_types)) & trading
    return _sorted_unique_sessions(normalized.loc[mask, "calendar_date"])


def market_type_calendar_state(calendar: pd.DataFrame, market_type: int) -> pd.Series:
    """Return explicit O/C facts; reindexing an absent date preserves unknown."""

    normalized = _require_normalized_calendar(calendar)
    source_market_types = strict_integer_series(
        normalized["source_market_type"],
        error="invalid_source_market_type",
    )
    trading = strict_bool_series(
        normalized["is_trading_day"],
        error="invalid_is_trading_day",
    )
    requested_market_type = strict_integer_series(
        pd.Series([market_type]),
        error="invalid_source_market_type",
    ).iloc[0]
    selected = source_market_types.eq(requested_market_type)
    dates = pd.DatetimeIndex(normalized.loc[selected, "calendar_date"])
    if dates.has_duplicates:
        raise ValueError("duplicate_trading_calendar_key")
    return pd.Series(trading.loc[selected].to_numpy(), index=dates, dtype="boolean")


def next_trading_sessions(
    calendar: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DatetimeIndex:
    """Prove next sessions within the supplied A-share calendar market scope."""
    normalized = _require_normalized_calendar(calendar)
    selected = normalized.loc[normalized.source_market_type.isin(CURRENT_MARKET_TYPES)]
    if selected.duplicated(["calendar_date", "source_market_type"]).any():
        raise ValueError("duplicate_trading_calendar_key")
    states = selected.pivot(
        index="calendar_date", columns="source_market_type", values="is_trading_day"
    ).astype("boolean")
    opened = states.any(axis=1)
    known = opened | states.notna().all(axis=1)
    sessions = states.index[opened]
    positions = sessions.get_indexer(dates)
    result = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = (positions >= 0) & (positions + 1 < len(sessions))
    rows = np.flatnonzero(valid)
    successors = sessions.take(positions[valid] + 1)
    known_dates = states.index[known]
    count = known_dates.searchsorted(
        successors, side="right"
    ) - known_dates.searchsorted(dates[valid], side="left")
    proved = count == ((successors - dates[valid]).days + 1)
    result[rows[proved]] = successors[proved].to_numpy(dtype="datetime64[ns]")
    return pd.DatetimeIndex(result)


def compute_listing_session_facts(
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    market_history: pd.DataFrame,
    *,
    observed_market_sessions: dict[int, pd.DatetimeIndex] | None = None,
) -> pd.DataFrame:
    """Derive exact age and proven eligibility from evidenced market membership."""
    normalized = _require_normalized_calendar(calendar)
    if normalized.duplicated(["source_market_type", "calendar_date"]).any():
        raise ValueError("duplicate_trading_calendar_key")
    observations = strict_timestamp_series(
        frame["timestamp"], error="invalid_listing_age_timestamp"
    ).dt.normalize()
    listings = strict_timestamp_series(
        frame["listing_date"], error="invalid_listing_age_timestamp", allow_null=True
    ).dt.normalize()
    result = pd.DataFrame(
        {
            "listing_age_trading_days": pd.Series(
                pd.NA, index=frame.index, dtype="Int64"
            ),
            "has_252_trading_sessions": pd.Series(
                pd.NA, index=frame.index, dtype="boolean"
            ),
        }
    )
    before_listing = observations.lt(listings)
    result.loc[before_listing, "listing_age_trading_days"] = 0
    result.loc[before_listing, "has_252_trading_sessions"] = False
    trading = strict_bool_series(
        normalized["is_trading_day"], error="invalid_is_trading_day"
    )
    market_types = strict_integer_series(
        normalized["source_market_type"], error="invalid_source_market_type"
    )
    sessions_by_market = {
        market: pd.DatetimeIndex(
            sorted(
                normalized.loc[
                    trading & market_types.eq(market), "calendar_date"
                ].unique()
            )
        )
        for market in market_types.unique()
    }
    unconfirmed_sessions = {}
    for market, observed in (observed_market_sessions or {}).items():
        original = sessions_by_market.get(market, pd.DatetimeIndex([]))
        unconfirmed_sessions[market] = observed.difference(original)
        sessions_by_market[market] = original.union(observed)
    dates_by_market = {
        market: pd.DatetimeIndex(
            sorted(normalized.loc[market_types.eq(market), "calendar_date"].unique())
        )
        for market in market_types.unique()
    }
    history_groups = market_history.groupby("asset_id", sort=False).indices
    for (asset, listing), indices in frame.groupby(
        ["asset_id", "listing_date"], sort=False, dropna=False
    ).groups.items():
        history = market_history.iloc[history_groups.get(asset, [])].sort_values(
            "timestamp", kind="mergesort"
        )
        if pd.notna(listing):
            history = history.loc[history.timestamp.ge(listing)]
        if history.empty:
            continue
        records = tuple(history.itertuples(index=False))
        counts = pd.Series(0, index=indices, dtype="Int64")
        exact = (
            listings.loc[indices].ge(records[0].timestamp)
            & ~before_listing.loc[indices]
        )
        for position, record in enumerate(records):
            active = observations.loc[indices].ge(record.timestamp)
            sessions = sessions_by_market.get(record.market_type, pd.DatetimeIndex([]))
            starts = (
                listings.loc[indices]
                .fillna(record.timestamp)
                .clip(lower=record.timestamp)
            )
            left = sessions.searchsorted(
                starts.to_numpy(dtype="datetime64[ns]"), side="left"
            )
            right = sessions.searchsorted(
                observations.loc[indices].to_numpy(dtype="datetime64[ns]"), side="right"
            )
            if position + 1 < len(records):
                right = right.clip(
                    max=sessions.searchsorted(
                        records[position + 1].timestamp, side="left"
                    )
                )
            counts += (right - left).clip(min=0)
            ends = observations.loc[indices]
            if position + 1 < len(records):
                ends = ends.clip(
                    upper=records[position + 1].timestamp - pd.Timedelta(days=1)
                )
            if record.market_type not in dates_by_market:
                exact &= ~active
                continue
            known_dates = dates_by_market[record.market_type]
            required_count = (ends - starts).dt.days + 1
            covered_count = known_dates.searchsorted(
                ends, side="right"
            ) - known_dates.searchsorted(starts, side="left")
            covered = (
                starts.ge(known_dates[0])
                & ends.le(known_dates[-1])
                & (required_count == covered_count)
            )
            unconfirmed = unconfirmed_sessions.get(
                record.market_type, pd.DatetimeIndex([])
            )
            covered &= (
                unconfirmed.searchsorted(ends, side="right")
                - unconfirmed.searchsorted(starts, side="left")
            ) == 0
            exact &= ~active | covered
        result.loc[indices[exact], "listing_age_trading_days"] = counts.loc[exact]
        result.loc[indices[exact], "has_252_trading_sessions"] = counts.loc[exact].ge(
            252
        )
        proven = counts.ge(252) & ~before_listing.loc[indices]
        result.loc[indices[proven], "has_252_trading_sessions"] = True
    return result


def _require_normalized_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(NORMALIZED_CALENDAR_COLUMNS.difference(calendar.columns))
    if missing:
        raise ValueError("calendar_missing_normalized_columns: " + ", ".join(missing))
    result = calendar.copy()
    result["calendar_date"] = strict_timestamp_series(
        result["calendar_date"],
        error="invalid_calendar_date",
    )
    result["calendar_date"] = result["calendar_date"].dt.normalize()
    return result


def _sorted_unique_sessions(values: pd.Series) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(values.dropna().unique()))


def continuous_session(timestamps, exchange, instrument_types=None):
    """Classify timestamps into continuous trading sessions under historical
    exchange hours.
    Shanghai stocks gained a closing auction in 2018; funds followed in 2026.
    """
    local = (
        pd.to_datetime(timestamps).dt.tz_localize("UTC").dt.tz_convert("Asia/Shanghai")
    )
    seconds = (
        local.dt.hour * 3600
        + local.dt.minute * 60
        + local.dt.second
        + local.dt.microsecond / 1e6
    )
    closing = _continuous_closing_seconds(local, exchange, instrument_types)
    return pd.Series(
        np.select(
            [
                seconds.ge(9 * 3600 + 30 * 60) & seconds.lt(11 * 3600 + 30 * 60),
                seconds.ge(13 * 3600) & seconds.lt(closing),
            ],
            ["am", "pm"],
            default="",
        ),
        index=timestamps.index,
    )


def continuous_session_end(
    timestamps, exchange, instrument_types, *, break_at_lunch=False
):
    """Return daily continuous closes, or current am/pm ends, in UTC."""
    local = (
        pd.to_datetime(timestamps).dt.tz_localize("UTC").dt.tz_convert("Asia/Shanghai")
    )
    seconds = _continuous_closing_seconds(local, exchange, instrument_types)
    if break_at_lunch:
        seconds = np.where(local.dt.hour.lt(12), 11 * 3600 + 30 * 60, seconds)
    return (
        (local.dt.normalize() + pd.to_timedelta(seconds, unit="s"))
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )


def _continuous_closing_seconds(local, exchange, instrument_types):
    etf = (
        False
        if instrument_types is None
        else instrument_types.isin(("equity_etf", "gold_etf"))
    )
    return np.where(
        exchange.eq("SSE")
        & (
            local.dt.date.lt(pd.Timestamp("2018-08-20").date())
            | (etf & local.dt.date.lt(pd.Timestamp("2026-07-06").date()))
        ),
        15 * 3600,
        14 * 3600 + 57 * 60,
    )
