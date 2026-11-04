"""Explicit small quote and daily evidence fixtures for A-share execution tests."""

import pandas as pd

from quant_research_platform.contracts.portfolios import EQUITY_EVENT_COLUMNS
from quant_research_platform.markets.implementations.a_share.portfolio_sessions import (
    _CORPORATE_EVENT_COLUMNS,
    _PortfolioSession,
)
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.portfolio_inputs import _interim_specs, _processed_specs
from tests.helpers.portfolio_ledger import write_market_evidence


def daily_view(quotes, *, listing=None, corporate_events=None, rights_events=None):
    day = pd.Timestamp(quotes.session_date.iloc[0])
    panel = (
        quotes.sort_values("timestamp")
        .groupby("asset_id", sort=True)
        .agg(
            open=("mid", "first"), close=("mid", "last"), exchange=("exchange", "first")
        )
        .reset_index()
    )
    panel["market_type"] = panel.pop("exchange").map({"SSE": 1, "SZSE": 4})
    panel = panel.assign(
        timestamp=day,
        listing_date=day if listing is None else listing,
        amount=10000000.0,
        is_trading=True,
    )
    return _PortfolioSession(
        day,
        panel,
        panel.copy(),
        frozenset(),
        pd.DataFrame(),
        frozenset(),
        frozenset(),
        frozenset(),
        pd.DataFrame(columns=(*EQUITY_EVENT_COLUMNS, "entitlement_proved"))
        if rights_events is None
        else rights_events,
        pd.DataFrame(columns=_CORPORATE_EVENT_COLUMNS)
        if corporate_events is None
        else corporate_events,
    )


def write_daily_inputs(root, quotes):
    listing = quotes.session_date.min()
    panels = [
        daily_view(frame, listing=listing).panel
        for _, frame in quotes.groupby("session_date")
    ]
    panel = pd.concat(panels, ignore_index=True).assign(
        market_id="a_share",
        period_return=0.0,
        float_market_cap=1e9,
        total_market_cap=1e9,
        industry_standard_id="fixture",
        industry_level1_code="one",
        is_nonfinancial_industry=True,
        is_a_share_scope=True,
        is_st=False,
        is_pt=False,
        listing_age_trading_days=300,
        has_252_trading_sessions=True,
        volume=1000000.0,
        turnover=0.01,
    )
    panel["adj_open"], panel["adj_close"] = panel.open, panel.close
    roots = write_market_evidence(root, panel)
    calendar = (
        panel[["timestamp", "market_type"]]
        .drop_duplicates()
        .rename(columns={"market_type": "source_market_type"})
    )
    calendar["calendar_date"] = calendar.timestamp
    calendar["is_trading_day"] = True
    write_dataset(root / "panel", panel, spec=_processed_specs().get("market_panel_1d"))
    write_dataset(
        root / "calendar", calendar, spec=_interim_specs().get("trading_calendar")
    )
    return {
        "panel": str(root / "panel"),
        "calendar": str(root / "calendar"),
        "market_bars": str(roots["market_bars_1d"]),
        "stock_master": str(roots["stock_master"]),
        "corporate_actions": str(roots["corporate_actions"]),
    }
