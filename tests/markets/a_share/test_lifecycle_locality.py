"""Unknown lifecycle bounds never become open-ended admission or erase prices."""

import pandas as pd

from quant_research_platform.markets.implementations.a_share.processed import (
    panels,
    producer,
)
from tests.helpers.a_share_sources import _synthetic_interim_frames


def test_later_unknown_delisting_is_a_local_barrier_without_rewriting_history():
    dates = pd.bdate_range("2020-01-02", periods=6).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    stocks = source["stock_master"].copy()
    stocks["timestamp"] = stocks["listing_date"]
    baseline = panels.compute_market_panel(
        source["market_bars_1d"],
        **panels._prepare_market_panel_inputs(
            stocks=panels._canonical_stock_master(stocks),
            market_cap=source["market_cap"],
            trading_calendar=source["trading_calendar"],
        ),
    )
    later = stocks.loc[stocks.source_asset_id.eq("000001")].copy()
    later["timestamp"] = dates[3]
    later["delisting_date_unresolved"] = True
    stocks = pd.concat([stocks, later], ignore_index=True)
    actual = panels.compute_market_panel(
        source["market_bars_1d"],
        **panels._prepare_market_panel_inputs(
            stocks=panels._canonical_stock_master(stocks),
            market_cap=source["market_cap"],
            trading_calendar=source["trading_calendar"],
        ),
    )
    affected = actual.asset_id.eq("000001") & actual.timestamp.ge(dates[3])
    pd.testing.assert_frame_equal(actual.loc[~affected], baseline.loc[~affected])
    assert actual.loc[affected, "close"].equals(baseline.loc[affected, "close"])
    assert actual.loc[affected, "delisting_date_unresolved"].all()
    assert actual.loc[affected, "delisting_date"].isna().all()
    assert actual.loc[affected, "listing_age_trading_days"].isna().all()
    assert actual.loc[affected, "has_252_trading_sessions"].isna().all()


def test_unknown_listing_date_preserves_observed_prices_and_other_assets():
    dates = pd.bdate_range("2020-01-02", periods=6).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    stocks = source["stock_master"].copy()
    stocks["timestamp"] = stocks["listing_date"]
    bad = stocks.source_asset_id.eq("000001")
    stocks.loc[bad, "timestamp"] = dates[0]
    stocks.loc[bad, "listing_date"] = pd.NaT
    actual = panels.compute_market_panel(
        source["market_bars_1d"],
        **panels._prepare_market_panel_inputs(
            stocks=panels._canonical_stock_master(stocks),
            market_cap=source["market_cap"],
            trading_calendar=source["trading_calendar"],
        ),
    )
    affected = actual.asset_id.eq("000001")
    assert len(actual) == 18
    assert actual.close.notna().all()
    assert actual.loc[affected, "listing_date"].isna().all()
    assert actual.loc[affected, "listing_age_trading_days"].isna().all()
    assert actual.loc[~affected, "listing_date"].notna().all()
    assert not actual.loc[~affected, "delisting_date_unresolved"].any()


def test_unknown_lifecycle_axis_keeps_only_observed_days_and_breaks_fill():
    from tests.markets.a_share.test_calendar_dependency_scope import _calendar
    from tests.markets.a_share.test_prepare_selector_clock import _panel

    dates = pd.bdate_range("2024-01-02", periods=8)
    panel = _panel(dates)
    panel["delisting_date"] = pd.NaT
    panel["delisting_date_unresolved"] = panel.timestamp.ge(dates[3])
    panel = panel.loc[~panel.timestamp.isin([dates[1], dates[5]])]
    calendar = _calendar({1: dates})
    axis = producer._calendar_panel_axis(panel, calendar)
    assert dates[1] in set(axis.timestamp)  # proven lifecycle can fill its clock
    assert dates[5] not in set(axis.timestamp)  # unknown interval cannot
    assert dates[3] in set(axis.timestamp)  # true observations remain represented


def test_execution_unknown_lifecycle_affects_only_its_actual_lookahead():
    from quant_research_platform.markets.implementations.a_share.execution import (
        derive_execution_mask,
    )
    from tests.markets.a_share.test_prepare_selector_clock import _panel

    dates = pd.bdate_range("2024-01-02", periods=6)
    a = _panel(dates)
    a["delisting_date_unresolved"] = a.timestamp.eq(dates[3])
    b = a.assign(asset_id="b", delisting_date_unresolved=False)
    panel = pd.concat([a, b], ignore_index=True)
    baseline = derive_execution_mask(panel.drop(columns="delisting_date_unresolved"))
    actual = derive_execution_mask(panel)
    affected = actual.asset_id.eq("a") & actual.timestamp.isin(dates[2:4])
    assert actual.loc[affected, ["can_enter", "can_exit"]].isna().all().all()
    pd.testing.assert_frame_equal(actual.loc[~affected], baseline.loc[~affected])
