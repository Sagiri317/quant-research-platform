"""Unknown lifecycle bounds never become open-ended admission or erase prices."""

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.processed import (
    panels,
    producer,
)
from tests.helpers.a_share_sources import _synthetic_interim_frames


def test_empty_stock_master_preserves_only_real_observations():
    dates = pd.bdate_range("2020-01-02", periods=6).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    bars = source["market_bars_1d"]
    bars = bars.loc[~(bars.source_asset_id.eq("000001") & bars.timestamp.eq(dates[2]))]
    actual = panels.compute_market_panel(
        bars,
        **panels._prepare_market_panel_inputs(
            stocks=panels._canonical_stock_master(source["stock_master"].iloc[:0]),
            market_cap=source["market_cap"],
            trading_calendar=source["trading_calendar"],
        ),
    )
    expected = bars.rename(columns={"source_asset_id": "asset_id"})
    columns = ["asset_id", "timestamp", "close"]
    observed = actual.loc[:, columns].sort_values(columns[:2]).reset_index(drop=True)
    expected = expected.loc[:, columns].sort_values(columns[:2]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        observed, expected, check_dtype=False, check_exact=True
    )
    pd.testing.assert_extension_array_equal(
        observed.close.array, expected.close.array, check_dtype=False, check_exact=True
    )
    assert len(actual) == 17
    assert actual.delisting_date_unresolved.all()
    assert actual.listing_date.isna().all()
    assert actual.listing_age_trading_days.isna().all()


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
    calculated = ["period_return", "turnover"]
    left, right = (
        panel.loc[~affected].drop(columns=calculated) for panel in (actual, baseline)
    )
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    for column in left.select_dtypes(include="floating"):
        pd.testing.assert_extension_array_equal(
            left[column].array, right[column].array, check_exact=True
        )
    pd.testing.assert_frame_equal(
        actual.loc[~affected, calculated], baseline.loc[~affected, calculated]
    )
    assert actual.close.equals(baseline.close)
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


@pytest.mark.parametrize("missing", ["listing_date", "stock_master"])
def test_production_axis_keeps_observed_assets_without_lifecycle(monkeypatch, missing):
    dates = pd.bdate_range("2020-01-02", periods=6).as_unit("ns")
    source = _synthetic_interim_frames(dates)
    stocks = source["stock_master"].copy()
    stocks["timestamp"] = stocks["listing_date"]
    affected = stocks.source_asset_id.eq("000001")
    if missing == "listing_date":
        stocks.loc[affected, "timestamp"] = dates[0]
        stocks.loc[affected, "listing_date"] = pd.NaT
    else:
        stocks = stocks.loc[~affected]
    stocks = panels._canonical_stock_master(stocks)
    bars = source["market_bars_1d"]
    bars = bars.loc[~(bars.source_asset_id.eq("000001") & bars.timestamp.eq(dates[2]))]

    class Session:
        def projected_view(self, columns):
            return self

    monkeypatch.setattr(producer, "iter_dataset_chunks", lambda *_a, **_k: iter([bars]))
    sessions, assets, history, *_ = producer._processed_axis(
        Session(),
        panel_source=True,
        calendar=source["trading_calendar"],
        stock_master=stocks,
    )
    assert set(assets.index) == set(bars.source_asset_id)
    actual = panels.compute_market_panel(
        bars,
        **panels._prepare_market_panel_inputs(
            stocks=stocks,
            market_cap=source["market_cap"],
            trading_calendar=source["trading_calendar"],
        ),
        session_window=pd.DatetimeIndex(sessions),
        asset_market_types=assets,
        market_history=history,
    )
    unknown = actual.loc[actual.asset_id.eq("000001")]
    assert set(unknown.timestamp) == set(dates) - {dates[2]}
    assert unknown.close.notna().all()
    assert unknown.listing_date.isna().all()
    assert unknown.listing_age_trading_days.isna().all()
    assert len(actual.loc[~actual.asset_id.eq("000001")]) == 12


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
