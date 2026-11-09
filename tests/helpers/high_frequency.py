"""Explicit small quote and daily evidence fixtures for A-share execution tests."""

from datetime import date

import pandas as pd

from quant_research_platform.contracts.data import InterimDataSpec
from quant_research_platform.contracts.portfolios import EQUITY_EVENT_COLUMNS
from quant_research_platform.markets.implementations.a_share.execution import (
    FeeRule,
    FeeSchedule,
)
from quant_research_platform.markets.implementations.a_share.portfolio_sessions import (
    _CORPORATE_EVENT_COLUMNS,
    _PortfolioSession,
)
from quant_research_platform.research.portfolios.high_frequency import _parameters
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.datasets.reading import open_dataset_session
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


def quotes(
    day, assets=("a", "b"), clocks=("01:30:00", "01:30:03", "01:30:06"), mid=10.0
):
    return pd.DataFrame(
        [
            dict(
                timestamp=pd.Timestamp(f"{day}T{clock}"),
                session_date=pd.Timestamp(day),
                asset_id=asset,
                exchange="SZSE",
                eligible=True,
                segment_id=0,
                mid=mid,
                bid_px1=mid - 0.01,
                ask_px1=mid + 0.01,
                bid_vol1=500.0,
                ask_vol1=500.0,
            )
            for clock in clocks
            for asset in assets
        ]
    )


def parameters(assets=("a", "b"), *, restore_at=None):
    return _parameters(
        dict(
            initial_capital=5000.0,
            initial_inventory=dict.fromkeys(assets, 100),
            maximum_inventory=dict.fromkeys(assets, 300),
            buy_threshold=0.5,
            sell_threshold=-0.5,
            latency_ms=0.0,
            participation_rate=1.0,
            minimum_commission=0.0,
            restore_at=restore_at,
        )
    )


def schedule():
    return FeeSchedule(
        tuple(
            FeeRule(
                code,
                date(2024, 1, 1),
                date(2024, 12, 31),
                "a_share",
                side,
                "amount",
                0.0,
                True,
            )
            for side in ("buy", "sell")
            for code in ("BROKER_COMMISSION", "TRANSFER_FEE", "STAMP_DUTY")
            if code != "STAMP_DUTY" or side == "sell"
        ),
        "synthetic-fees",
    )


def trade_input(root, records):
    frame = pd.DataFrame(
        records, columns=["second", "asset_id", "side", "volume", "price"]
    )
    frame["timestamp"] = pd.Timestamp("2024-01-02T01:30:00") + pd.to_timedelta(
        frame.pop("second"), unit="s"
    )
    frame["source_row"] = range(len(frame))
    frame["event_kind"] = "trade"
    roles = dict.fromkeys(frame, "context")
    roles.update(timestamp="timestamp", asset_id="key", source_row="key")
    write_dataset(
        root,
        frame,
        spec=InterimDataSpec(
            data_id="hft_trades",
            source_id="synthetic",
            frequency="tick",
            grain="trade_event",
            primary_key=("timestamp", "asset_id", "source_row"),
            column_roles=roles,
            column_units={"price": "CNY", "volume": "shares"},
            partition_columns=(),
        ),
    )
    return open_dataset_session(root)
