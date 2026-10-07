"""Internal A-share trading calendar helpers."""

from __future__ import annotations

from collections.abc import Iterable

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


def market_type_trading_sessions(
    calendar: pd.DataFrame,
    market_type: int,
) -> pd.DatetimeIndex:
    """Return sorted unique trading sessions for one source market type."""

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
    mask = source_market_types.eq(requested_market_type) & trading
    return _sorted_unique_sessions(normalized.loc[mask, "calendar_date"])


def compute_listing_session_facts(
    frame: pd.DataFrame,
    calendar: pd.DataFrame,
    market_history: pd.DataFrame,
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
    calendar_dates = pd.DatetimeIndex(sorted(normalized["calendar_date"].unique()))
    dates_by_market = {
        market: pd.DatetimeIndex(
            sorted(normalized.loc[market_types.eq(market), "calendar_date"].unique())
        )
        for market in market_types.unique()
    }
    history_groups = market_history.groupby("asset_id", sort=False).indices
    for asset, indices in frame.groupby("asset_id", sort=False).groups.items():
        history = market_history.iloc[history_groups.get(asset, [])].sort_values(
            "timestamp", kind="mergesort"
        )
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
            if record.market_type not in dates_by_market:
                exact &= ~active
                continue
            sessions = sessions_by_market[record.market_type]
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
            known_dates = dates_by_market[record.market_type]
            required_count = calendar_dates.searchsorted(
                ends, side="right"
            ) - calendar_dates.searchsorted(starts, side="left")
            covered_count = known_dates.searchsorted(
                ends, side="right"
            ) - known_dates.searchsorted(starts, side="left")
            covered = (
                starts.ge(known_dates[0])
                & ends.le(known_dates[-1])
                & (required_count == covered_count)
            )
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
    dates = strict_timestamp_series(
        values,
        error="invalid_calendar_date",
    ).dt.normalize()
    return pd.DatetimeIndex(sorted(dates.dropna().unique()))
