"""Point-in-time stock membership using existing identity and status facts."""

import numpy as np
import pandas as pd

from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
    strict_integer_series,
    strict_timestamp_series,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    CURRENT_MARKET_TYPES,
    continuous_session,
    market_type_calendar_state,
)
from quant_research_platform.markets.implementations.a_share.processed.panels import (
    _asof_join_stock_master,
    _canonical_stock_master,
    _status_event_frontier,
)
from quant_research_platform.storage.datasets.reading import (
    _dataset_input_options,
    materialize_dataset_session,
    open_dataset_session,
)

_A_SHARE_FUSION_ELIGIBILITY_FACTS = (
    "is_nonfinancial_industry",
    "has_legal_fundamental_pit",
)


def a_share_eligibility_columns(
    *,
    minimum_listing_sessions: int = 252,
    exclude_st: bool = True,
    exclude_pt: bool = True,
) -> tuple[str, ...]:
    """Select only the facts used by the requested research scope."""
    if (
        type(minimum_listing_sessions) is not int
        or minimum_listing_sessions < 0
        or type(exclude_st) is not bool
        or type(exclude_pt) is not bool
    ):
        raise ValueError("invalid_a_share_eligibility_parameters")
    columns = ["is_a_share_scope"]
    if exclude_st:
        columns.append("is_st")
    if exclude_pt:
        columns.append("is_pt")
    if minimum_listing_sessions:
        columns.append(
            "has_252_trading_sessions"
            if minimum_listing_sessions == 252
            else "listing_age_trading_days"
        )
    return tuple(columns)


A_SHARE_ELIGIBILITY_FACTS = a_share_eligibility_columns()


def a_share_eligibility(
    frame: pd.DataFrame,
    recipe: str,
    *,
    minimum_listing_sessions: int = 252,
    exclude_st: bool = True,
    exclude_pt: bool = True,
    preserve_unknown: bool = False,
) -> pd.Series:
    """Derive research membership without imposing factor or execution validity.

    Unknown relevant facts are retained in the input and never affirmed as eligible.
    """
    if recipe not in {"a_share_all", "a_share_nonfinancial", "a_share_fundamental"}:
        raise ValueError("invalid_a_share_eligibility_recipe")
    columns = a_share_eligibility_columns(
        minimum_listing_sessions=minimum_listing_sessions,
        exclude_st=exclude_st,
        exclude_pt=exclude_pt,
    )
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError("a_share_eligibility_facts_missing: " + ", ".join(missing))
    eligible = pd.Series(True, index=frame.index, dtype="boolean")
    for column in columns:
        if column == "listing_age_trading_days":
            values = strict_finite_numeric_series(
                frame[column],
                error=f"invalid_a_share_eligibility_fact: {column}",
                allow_null=True,
            )
            if ((values.dropna() < 0) | (values.dropna() % 1 != 0)).any():
                raise ValueError(f"invalid_a_share_eligibility_fact: {column}")
            eligible &= values.ge(minimum_listing_sessions)
        else:
            fact = strict_bool_series(
                frame[column],
                error=f"invalid_a_share_eligibility_fact: {column}",
                allow_null=True,
            )
            eligible &= ~fact if column in {"is_st", "is_pt"} else fact
    if recipe in {"a_share_nonfinancial", "a_share_fundamental"}:
        required = (
            ("is_nonfinancial_industry",)
            if recipe == "a_share_nonfinancial"
            else _A_SHARE_FUSION_ELIGIBILITY_FACTS
        )
        for column in required:
            if column not in frame:
                raise ValueError(f"a_share_eligibility_facts_missing: {column}")
            eligible &= strict_bool_series(
                frame[column],
                error=f"invalid_a_share_eligibility_fact: {column}",
                allow_null=True,
            )
        for column in (
            "industry_standard_id",
            "industry_level1_code",
            "industry_level1_name",
        ):
            if column not in frame:
                raise ValueError(f"a_share_eligibility_facts_missing: {column}")
            values = frame[column].astype("string").str.strip()
            eligible &= values.notna() & values.ne("")
    return eligible if preserve_unknown else eligible.fillna(False).astype(bool)


def _read(inputs, columns=None):
    source = open_dataset_session(**_dataset_input_options(inputs))
    return materialize_dataset_session(
        source if columns is None else source.projected_view(tuple(columns))
    )


def load_stock_status(config):
    """Load identity and state facts without assuming a daily publication clock."""
    config = _strict_config(
        config, {"stock_master", "status", "available_at_column"}, set()
    )
    stocks = _canonical_stock_master(_read(config["stock_master"]))
    undated = stocks.get(
        "timestamp", pd.Series(pd.NaT, index=stocks.index, dtype="datetime64[ns]")
    ).isna()
    # An undated life can identify a listing, not when mutable state was known.
    for name in ("stock_name", "company_status", "is_initial_listing"):
        if name in stocks:
            stocks.loc[undated, name] = pd.NA
    stocks.loc[undated, "delisting_date"] = pd.NaT
    stocks.loc[undated, "delisting_date_unresolved"] = True
    available = config["available_at_column"]
    required = {
        "asset_id",
        "timestamp",
        "listing_date",
        "market_type",
        "is_st",
        "is_pt",
        available,
    }
    source = open_dataset_session(**_dataset_input_options(config["status"]))
    if "is_delisting" in source.logical_columns:
        required.add("is_delisting")
    status = materialize_dataset_session(source.projected_view(tuple(required)))
    if "is_delisting" not in status:
        status["is_delisting"] = pd.Series(pd.NA, index=status.index, dtype="boolean")
    status["asset_id"] = status.asset_id.astype(stocks.asset_id.dtype)
    for name in ("timestamp", "listing_date", available):
        status[name] = strict_timestamp_series(
            status[name], error="stock_pool_status_clock_invalid", allow_null=True
        )
    status["_available_at"] = status[available]
    status["_effective_at"] = status.timestamp
    status = status.loc[
        status[["asset_id", "listing_date", "_available_at", "_effective_at"]]
        .notna()
        .all(axis=1)
    ]
    status["_known_at"] = status[["_effective_at", "_available_at"]].max(axis=1)
    status["market_type"] = strict_integer_series(
        status.market_type, error="stock_pool_market_type_invalid", allow_null=True
    )
    for name in ("is_st", "is_pt", "is_delisting"):
        status[name] = strict_bool_series(
            status[name], error="stock_pool_status_invalid", allow_null=True
        )
    status = _status_event_frontier(
        status,
        keys=("asset_id", "listing_date"),
        effective_at="_effective_at",
        known_at="_known_at",
        columns=("market_type", "is_st", "is_pt", "is_delisting"),
    )
    return {"stocks": stocks, "status": status}


def load_stock_pool(config):
    """Add exchange calendars and declared IPO facts to the shared status input."""
    config = _strict_config(
        config,
        {"stock_master", "status", "available_at_column", "trading_calendar"},
        {"trading_no_limit"},
    )
    state = load_stock_status(
        {
            name: config[name]
            for name in ("stock_master", "status", "available_at_column")
        }
    )
    stocks = state["stocks"]
    calendar = _read(
        config["trading_calendar"],
        ("calendar_date", "source_market_type", "is_trading_day"),
    )
    calendars = {
        market: market_type_calendar_state(calendar, market)
        for market in CURRENT_MARKET_TYPES
    }
    no_limit = None
    if config.get("trading_no_limit") is not None:
        choice = _strict_config(
            config["trading_no_limit"],
            {"inputs", "available_at_column", "ipo_reasons"},
            set(),
        )
        reasons = choice["ipo_reasons"]
        if (
            type(reasons) is not list
            or not reasons
            or any(type(reason) is not str or not reason for reason in reasons)
        ):
            raise ValueError("stock_pool_ipo_reasons_invalid")
        available = choice["available_at_column"]
        no_limit = _read(
            choice["inputs"], ("source_asset_id", "timestamp", "reason", available)
        ).rename(columns={"source_asset_id": "asset_id"})
        no_limit["asset_id"] = no_limit.asset_id.astype(stocks.asset_id.dtype)
        no_limit = no_limit.loc[no_limit.reason.isin(reasons)].copy()
        for name in ("timestamp", available):
            no_limit[name] = strict_timestamp_series(
                no_limit[name],
                error="stock_pool_no_limit_clock_invalid",
                allow_null=True,
            )
        no_limit["_available_at"] = no_limit[available]
        no_limit = no_limit.loc[
            no_limit[["timestamp", "_available_at"]].notna().all(axis=1)
        ]
        no_limit["_known_at"] = no_limit[["timestamp", "_available_at"]].max(axis=1)
        no_limit = _asof_join_stock_master(no_limit, stocks)
    return {
        **state,
        "calendars": calendars,
        "no_limit": no_limit,
    }


def _ipo_no_limit(frame, calendars):
    """Keep unproved regimes unknown; first-five rules apply only to registered IPOs."""
    result = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    dates = frame.timestamp.dt.normalize()
    for market, indices in frame.groupby("market_type", sort=False).groups.items():
        if market not in calendars:
            continue
        calendar = calendars[market]
        open_dates = pd.DatetimeIndex(
            calendar.index[calendar.fillna(False)]
        ).sort_values()
        if open_dates.empty:
            continue
        listings = pd.DatetimeIndex(frame.loc[indices, "listing_date"])
        days = pd.DatetimeIndex(dates.loc[indices])
        counts = open_dates.searchsorted(days, side="right") - open_dates.searchsorted(
            listings
        )
        # Six evidenced exchange sessions suffice to leave any supported IPO window.
        result.loc[indices[counts > 5]] = False
        known_dates = pd.DatetimeIndex(calendar.index[calendar.notna()]).sort_values()
        covered = (
            known_dates.searchsorted(days, side="right")
            - known_dates.searchsorted(listings)
        ) == ((days - listings).days + 1)
        on_session = calendar.reindex(days).fillna(False).to_numpy(dtype=bool)
        exact = covered & on_session & (counts > 0)
        listing_known = (
            (
                frame.loc[indices, "_listing_market_type"].eq(market)
                & frame.loc[indices, "_initial_life"]
            )
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        # Trading rules changed for new IPOs, not retroactively for existing lives.
        first_registered = {
            1: "2023-04-10",
            4: "2023-04-10",
            16: "2020-08-24",
            32: "2019-07-22",
        }[market]
        registered = listings >= pd.Timestamp(first_registered)
        selected = exact & registered & listing_known
        result.loc[indices[selected]] = counts[selected] <= 5
        # These historical IPOs had bounded 64%-144% first-day offer-price bands.
        bounded = (
            (market in {1, 4, 16})
            & (listings >= pd.Timestamp("2013-12-13" if market == 1 else "2014-06-13"))
            & ~registered
            & listing_known
        )
        result.loc[indices[bounded & exact]] = False
    return result


def stock_status_rows(decisions, *, stocks, status):
    """Project state onto actual decision rows, in input order, without a grid."""
    frame = decisions.loc[:, ["asset_id", "timestamp"]].copy()
    frame["asset_id"] = frame.asset_id.astype(stocks.asset_id.dtype)
    frame["timestamp"] = frame.timestamp.dt.as_unit("ns")
    frame["_row"] = np.arange(len(frame))
    frame = (
        _asof_join_stock_master(frame, stocks)
        .sort_values("_row")
        .reset_index(drop=True)
    )
    columns = (
        "_known_at",
        "_effective_at",
        "_available_at",
        "market_type",
        "is_st",
        "is_pt",
        "is_delisting",
    )
    known = frame.listing_date.notna()
    joined = pd.merge_asof(
        frame.loc[known].sort_values("timestamp", kind="stable"),
        status.loc[:, ["asset_id", "listing_date", *columns]],
        left_on="timestamp",
        right_on="_known_at",
        by=["asset_id", "listing_date"],
        direction="backward",
        allow_exact_matches=True,
    ).set_index("_row")
    for name in columns:
        frame[name] = joined[name].reindex(frame.index)
    return frame.drop(columns="_row")


def stock_pool_rows(
    grid, *, stocks, status, calendars, no_limit, require_continuous=True
):
    """Project identity and membership on a bounded decision-clock block."""
    frame = stock_status_rows(
        pd.DataFrame({"asset_id": stocks.asset_id.drop_duplicates()}).merge(
            pd.DataFrame({"timestamp": grid}), how="cross"
        ),
        stocks=stocks,
        status=status,
    )
    frame["_initial_life"] = frame.get(
        "is_initial_listing", pd.Series(pd.NA, index=frame.index, dtype="boolean")
    )
    frame["_row"] = np.arange(len(frame))
    known = frame.listing_date.notna()
    initial = status.loc[status._effective_at.le(status.listing_date)]
    listed = pd.merge_asof(
        frame.loc[known, ["_row", "asset_id", "listing_date", "timestamp"]].sort_values(
            "timestamp", kind="stable"
        ),
        initial.loc[:, ["asset_id", "listing_date", "_known_at", "market_type"]],
        left_on="timestamp",
        right_on="_known_at",
        by=["asset_id", "listing_date"],
        direction="backward",
        allow_exact_matches=True,
    ).set_index("_row")
    frame["_listing_market_type"] = listed.market_type.reindex(frame.index)
    frame["is_a_share_scope"] = (
        frame.market_type.isin(CURRENT_MARKET_TYPES)
        .astype("boolean")
        .where(frame.market_type.notna())
    )
    eligible = a_share_eligibility(
        frame, "a_share_all", minimum_listing_sessions=0, preserve_unknown=True
    )
    day = frame.timestamp.dt.normalize()
    alive = (
        day.ge(frame.listing_date).astype("boolean").where(frame.listing_date.notna())
    )
    alive &= (
        day.le(frame.delisting_date)
        .astype("boolean")
        .where(~frame.delisting_date_unresolved.fillna(True))
    )
    ipo = _ipo_no_limit(frame, calendars)
    if no_limit is not None and len(no_limit):
        events = no_limit.assign(_day=no_limit.timestamp.dt.normalize())
        events = events.groupby(
            ["asset_id", "listing_date", "_day"], sort=False
        )._known_at.min()
        available = events.reindex(
            pd.MultiIndex.from_arrays([frame.asset_id, frame.listing_date, day])
        )
        available.index = frame.index
        ipo.loc[available.notna() & frame.timestamp.ge(available)] = True
    session_open = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    for market, indices in frame.groupby("market_type", sort=False).groups.items():
        if market in calendars:
            session_open.loc[indices] = (
                calendars[market].reindex(day.loc[indices]).to_numpy()
            )
    frame["eligible"] = eligible & alive & ~ipo & session_open & ~frame.is_delisting
    frame["exchange"] = frame.market_type.map(
        {1: "SSE", 32: "SSE", 4: "SZSE", 16: "SZSE"}
    )
    frame["instrument_type"] = "stock"
    frame["session_date"] = day
    frame["market_id"] = "a_share"
    if require_continuous:
        frame["eligible"] &= continuous_session(
            frame.timestamp, frame.exchange, frame.instrument_type
        ).ne("")
    return frame.loc[
        :,
        [
            "market_id",
            "asset_id",
            "timestamp",
            "session_date",
            "exchange",
            "instrument_type",
            "listing_date",
            "eligible",
        ],
    ]
