from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.execution import (
    derive_execution_mask,
)
from quant_research_platform.markets.implementations.a_share.processed.panels import (
    _apply_suspension_semantics,
    _industry_version_active_mask,
    compute_market_panel,
)
from quant_research_platform.research.labels.engine import calculate_labels_for_panel
from quant_research_platform.research.protocols.a_share import label_definitions


@pytest.mark.parametrize("has_raw_history", [False, True])
def test_missing_trading_prices_remain_missing_and_suspension_preserves_units(
    has_raw_history,
) -> None:
    from quant_research_platform.research.datasets.assembler import _a_share_eligibility

    frame = pd.DataFrame(
        {
            "asset_id": "asset",
            "timestamp": pd.bdate_range("2020-01-01", periods=3),
            "filling_flag": [0, 0, 1],
            "close": [10.0 if has_raw_history else np.nan, np.nan, np.nan],
            "adj_close": [100.0, 110.0, np.nan],
            "open": [10.0 if has_raw_history else np.nan, np.nan, np.nan],
            "high": [10.0 if has_raw_history else np.nan, np.nan, np.nan],
            "low": [10.0 if has_raw_history else np.nan, np.nan, np.nan],
            "adj_open": [100.0, np.nan, np.nan],
            "period_return": [0.0, 0.07, np.nan],
            "volume": 100.0,
            "amount": 1000.0,
            "float_market_cap": 1000.0,
            "market_type": 1,
            "is_st": False,
            "is_pt": False,
            "listing_age_trading_days": 300,
        }
    )
    actual = _apply_suspension_semantics(frame)
    assert actual.loc[1, ["close", "open", "high", "low", "adj_open"]].isna().all()
    assert not _a_share_eligibility(actual, "pv_all").iloc[1]
    assert actual.loc[1, "period_return"] == 0.07
    assert actual.loc[2, "adj_close"] == actual.loc[2, "adj_open"] == 110.0
    raw = actual.loc[2, ["close", "open", "high", "low"]]
    # The missing trading quote resets raw prices without resetting adjusted prices.
    assert raw.isna().all()


@pytest.mark.parametrize("missing_caps", [False, True])
def test_market_panel_keeps_suspension_and_excludes_holiday_fill(missing_caps) -> None:
    sessions = pd.DatetimeIndex(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    )
    bars = pd.DataFrame(
        {
            "source_asset_id": ["000001"] * 4,
            "timestamp": sessions,
            "market_type": [1] * 4,
            "trading_status_code": [0] * 4,
            "filling_flag": [0, 2, 1, 0],
            "open": [10.0, 10.5, np.nan, 12.0],
            "high": [10.0, 10.5, np.nan, 12.0],
            "low": [10.0, 10.5, np.nan, 12.0],
            "close": [10.0, 10.5, np.nan, 12.0],
            "volume": [100.0, 100.0, np.nan, 100.0],
            "amount": [1_000.0, 1_050.0, np.nan, 1_200.0],
            "period_return": [0.0, 0.05, np.nan, 0.2],
            "adj_close": [10.0, 10.5, np.nan, 12.0],
            "adj_open": [10.0, 10.5, np.nan, 12.0],
            "float_market_cap": [1_000.0] * 4,
            "total_market_cap": [1_000.0] * 4,
            "prev_close": [9.0, 10.0, np.nan, 10.0],
            "limit_status": [0.0] * 4,
        }
    )
    if missing_caps:
        bars[["float_market_cap", "total_market_cap"]] = np.nan
    stock_master = pd.DataFrame(
        {
            "source_asset_id": ["000001"],
            "timestamp": [sessions[0]],
            "stock_name": ["asset"],
            "listing_date": [sessions[0] - pd.Timedelta(days=500)],
            "delisting_date": [pd.NaT],
            "company_status": ["normal"],
            "industry_standard_id": ["current-only"],
            "industry_level1_code": ["current-only"],
            "industry_level1_name": ["current-only"],
        }
    )
    market_cap = pd.DataFrame(
        {
            "source_asset_id": ["000001"],
            "timestamp": [sessions[0] - pd.Timedelta(days=500)],
            "share_change_date": [sessions[0] - pd.Timedelta(days=500)],
            "total_shares": [100.0],
            "a_float_shares": [100.0],
        }
    )
    calendar = pd.DataFrame(
        {
            "calendar_date": sessions,
            "source_market_type": [1] * 4,
            "is_trading_day": [True, False, True, True],
        }
    )

    result = compute_market_panel(
        bars,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=calendar,
    )

    assert result["timestamp"].tolist() == [sessions[0], sessions[2], sessions[3]]
    suspension = result.loc[result["timestamp"].eq(sessions[2])].iloc[0]
    reopening = result.loc[result["timestamp"].eq(sessions[3])].iloc[0]
    assert bool(suspension["is_trading"]) is False
    assert suspension[["volume", "amount", "turnover"]].tolist() == [0.0] * 3
    assert suspension["adj_close"] == 10.0
    assert suspension["period_return"] == 0.0
    assert reopening["period_return"] == pytest.approx(0.2)
    assert result["industry_level1_code"].isna().all()
    assert "scenario_universe" not in result
    if missing_caps:
        assert result[["float_market_cap", "total_market_cap"]].isna().all().all()
        assert result.loc[result.is_trading, "turnover"].isna().all()
    else:
        assert result.float_market_cap.tolist() == [1000.0] * 3
        assert result.total_market_cap.tolist() == [1000.0] * 3


def test_holiday_fill_on_open_session_fails_without_cross_asset_deletion() -> None:
    sessions = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    bars = pd.DataFrame(
        {
            "source_asset_id": ["A", "A", "B", "B"],
            "timestamp": [sessions[0], sessions[1], sessions[0], sessions[1]],
            "market_type": [1] * 4,
            "trading_status_code": [0] * 4,
            "filling_flag": [0, 0, 2, 0],
            "open": [10.0, 11.0, 20.0, 21.0],
            "high": [10.0, 11.0, 20.0, 21.0],
            "low": [10.0, 11.0, 20.0, 21.0],
            "close": [10.0, 11.0, 20.0, 21.0],
            "volume": [100.0] * 4,
            "amount": [1_000.0, 1_100.0, 2_000.0, 2_100.0],
            "period_return": [0.0] * 4,
            "adj_close": [10.0, 11.0, 20.0, 21.0],
            "adj_open": [10.0, 11.0, 20.0, 21.0],
            "float_market_cap": [1_000.0] * 4,
            "total_market_cap": [1_000.0] * 4,
            "limit_status": [0.0] * 4,
        }
    )
    stock_master = pd.DataFrame(
        {
            "source_asset_id": ["A", "B"],
            "timestamp": [sessions[0], sessions[0]],
            "stock_name": ["A", "B"],
            "listing_date": [sessions[0] - pd.Timedelta(days=500)] * 2,
            "delisting_date": [pd.NaT] * 2,
            "company_status": ["normal"] * 2,
        }
    )
    market_cap = pd.DataFrame(
        {
            "source_asset_id": ["A", "B"],
            "timestamp": [sessions[0] - pd.Timedelta(days=500)] * 2,
            "share_change_date": [sessions[0] - pd.Timedelta(days=500)] * 2,
            "total_shares": [100.0] * 2,
            "a_float_shares": [100.0] * 2,
        }
    )
    calendar = pd.DataFrame(
        {
            "calendar_date": sessions,
            "source_market_type": [1, 1],
            "is_trading_day": [True, True],
        }
    )

    with pytest.raises(ValueError, match="holiday_fill_conflicts_with_open_session"):
        compute_market_panel(
            bars,
            stock_master=stock_master,
            market_cap=market_cap,
            trading_calendar=calendar,
        )


def test_market_panel_physical_order_is_time_major_without_changing_rows() -> None:
    sessions = pd.bdate_range("2024-01-02", periods=2)
    rows = [(asset, timestamp) for asset in ("A", "B") for timestamp in sessions]
    bars = pd.DataFrame(
        {
            "source_asset_id": [asset for asset, _timestamp in rows],
            "timestamp": [timestamp for _asset, timestamp in rows],
            "market_type": 1,
            "trading_status_code": 0,
            "filling_flag": 0,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 100.0,
            "amount": 1_000.0,
            "period_return": 0.0,
            "adj_close": 10.0,
            "adj_open": 10.0,
            "float_market_cap": 1_000.0,
            "total_market_cap": 1_000.0,
            "limit_status": 0.0,
        }
    )
    stock_master = pd.DataFrame(
        {
            "source_asset_id": ["A", "B"],
            "timestamp": sessions[0],
            "stock_name": ["A", "B"],
            "listing_date": sessions[0] - pd.Timedelta(days=500),
            "delisting_date": pd.NaT,
            "company_status": "normal",
        }
    )
    market_cap = pd.DataFrame(
        {
            "source_asset_id": ["A", "B"],
            "timestamp": sessions[0] - pd.Timedelta(days=500),
            "share_change_date": sessions[0] - pd.Timedelta(days=500),
            "total_shares": 100.0,
            "a_float_shares": 100.0,
        }
    )
    calendar = pd.DataFrame(
        {
            "calendar_date": sessions,
            "source_market_type": 1,
            "is_trading_day": True,
        }
    )

    result = compute_market_panel(
        bars,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=calendar,
    )

    assert list(
        result[["timestamp", "asset_id"]].itertuples(index=False, name=None)
    ) == [
        (sessions[0], "A"),
        (sessions[0], "B"),
        (sessions[1], "A"),
        (sessions[1], "B"),
    ]


def test_execution_mask_is_independent_and_uses_signal_row_legality() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=3)
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset",
            "timestamp": sessions,
            "is_trading": True,
            "limit_status": [0, 1, 0],
            "open": [10.0, 11.0, 12.0],
            "high": [10.0, 11.0, 12.0],
            "low": [10.0, 11.0, 12.0],
            "close": [10.0, 11.0, 12.0],
        }
    )

    result = derive_execution_mask(panel)

    assert "scenario_universe" not in result
    assert result["can_enter"].iloc[:2].tolist() == [False, True]
    assert pd.isna(result["can_enter"].iloc[2])
    assert result["can_exit"].iloc[:2].tolist() == [True, True]
    assert pd.isna(result["can_exit"].iloc[2])


def test_execution_mask_and_label_consumer_share_signal_row_timing() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=4)
    panel = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": "asset",
            "timestamp": sessions,
            "is_trading": True,
            "limit_status": [0, 1, 0, 0],
            "open": [10.0, 11.0, 12.0, 13.0],
            "high": [10.0, 11.0, 12.0, 13.0],
            "low": [10.0, 11.0, 12.0, 13.0],
            "close": [10.0, 11.0, 12.0, 13.0],
            "adj_open": [10.0, 11.0, 12.0, 13.0],
        }
    )
    mask = derive_execution_mask(panel)
    labeled = calculate_labels_for_panel(
        panel.merge(mask, on=["market_id", "asset_id", "timestamp"]),
        definitions=(label_definitions()[0],),
        sessions=sessions,
    )

    assert mask["can_enter"].iloc[:3].tolist() == [False, True, True]
    assert pd.isna(mask["can_enter"].iloc[3])
    assert pd.isna(labeled.loc[0, "forward_return_1d"])
    assert labeled.loc[1, "forward_return_1d"] == pytest.approx(13.0 / 12.0 - 1.0)


def test_no_limit_history_overrides_historical_limit_status() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=3)
    bars = pd.DataFrame(
        {
            "source_asset_id": ["000001"] * 3,
            "timestamp": sessions,
            "market_type": 1,
            "trading_status_code": 0,
            "filling_flag": 0,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 100.0,
            "amount": 1_000.0,
            "period_return": 0.0,
            "adj_close": 10.0,
            "adj_open": 10.0,
            "float_market_cap": 1_000.0,
            "total_market_cap": 1_000.0,
            "prev_close": 10.0,
            "limit_status": [0.0, 1.0, 0.0],
        }
    )
    stock_master = pd.DataFrame(
        {
            "source_asset_id": ["000001"],
            "timestamp": [sessions[0]],
            "listing_date": [sessions[0] - pd.Timedelta(days=500)],
            "delisting_date": [pd.NaT],
            "stock_name": ["asset"],
            "company_status": ["normal"],
        }
    )
    market_cap = pd.DataFrame(
        {
            "source_asset_id": ["000001"],
            "timestamp": [sessions[0] - pd.Timedelta(days=500)],
            "share_change_date": [sessions[0] - pd.Timedelta(days=500)],
            "total_shares": [100.0],
            "a_float_shares": [100.0],
        }
    )
    calendar = pd.DataFrame(
        {
            "calendar_date": sessions,
            "source_market_type": 1,
            "is_trading_day": True,
        }
    )
    no_limit = pd.DataFrame(
        {
            "source_asset_id": ["000001"],
            "timestamp": [sessions[1]],
            "reason": ["historical exception"],
        }
    )

    panel = compute_market_panel(
        bars,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=calendar,
        no_limit_history=no_limit,
    )
    mask = derive_execution_mask(panel)

    assert panel.loc[panel["timestamp"].eq(sessions[1]), "limit_status"].iloc[0] == 0.0
    assert mask.loc[mask["timestamp"].eq(sessions[0]), "can_enter"].iloc[0]


def test_missing_listing_lifecycle_does_not_enter_the_panel() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=2)
    bars = pd.DataFrame(
        {
            "source_asset_id": ["000001"] * 2,
            "timestamp": sessions,
            "market_type": 1,
            "trading_status_code": 0,
            "filling_flag": 0,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 100.0,
            "amount": 1_000.0,
            "period_return": 0.0,
            "adj_close": 10.0,
            "adj_open": 10.0,
            "float_market_cap": 1_000.0,
            "total_market_cap": 1_000.0,
            "prev_close": 10.0,
            "limit_status": 0.0,
        }
    )
    calendar = pd.DataFrame(
        {
            "calendar_date": sessions,
            "source_market_type": 1,
            "is_trading_day": True,
        }
    )

    result = compute_market_panel(
        bars,
        stock_master=pd.DataFrame(
            columns=["source_asset_id", "listing_date", "delisting_date"]
        ),
        market_cap=pd.DataFrame(
            {
                "source_asset_id": ["000001"],
                "timestamp": [sessions[0]],
            }
        ),
        trading_calendar=calendar,
    )

    assert result.empty


def test_industry_history_uses_effective_standard_and_set_hierarchy() -> None:
    sessions = pd.DatetimeIndex(
        pd.to_datetime(["2022-12-30", "2023-01-03", "2023-01-04"])
    )
    bars = pd.DataFrame(
        {
            "source_asset_id": ["000001"] * 3,
            "timestamp": sessions,
            "market_type": 1,
            "trading_status_code": 0,
            "filling_flag": 0,
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 100.0,
            "amount": 1_000.0,
            "period_return": 0.0,
            "adj_close": 10.0,
            "adj_open": 10.0,
            "float_market_cap": 1_000.0,
            "total_market_cap": 1_000.0,
            "prev_close": 10.0,
            "limit_status": 0.0,
        }
    )
    stock_master = pd.DataFrame(
        {
            "source_asset_id": ["000001"],
            "timestamp": [pd.Timestamp("2020-01-01")],
            "listing_date": [pd.Timestamp("2020-01-01")],
            "delisting_date": [pd.NaT],
            "stock_name": ["asset"],
            "company_status": ["normal"],
        }
    )
    market_cap = pd.DataFrame(
        {
            "source_asset_id": ["000001"],
            "timestamp": [pd.Timestamp("2020-01-01")],
            "share_change_date": [pd.Timestamp("2020-01-01")],
            "total_shares": [100.0],
            "a_float_shares": [100.0],
        }
    )
    calendar = pd.DataFrame(
        {
            "calendar_date": sessions,
            "source_market_type": 1,
            "is_trading_day": True,
        }
    )
    history = pd.DataFrame(
        {
            "source_asset_id": ["000001", "000001"],
            "timestamp": [pd.Timestamp("2022-12-29"), pd.Timestamp("2023-01-01")],
            "industry_standard_id": ["P0207", "P0221"],
            "industry_level1_code": ["A01", "B01"],
            "industry_level1_name": ["not authoritative", "not authoritative"],
        }
    )
    industry_sets = pd.DataFrame(
        {
            "industry_standard_id": ["P0207", "P0221"],
            "industry_code": ["A01", "B01"],
            "industry_name": ["CSRC industry", "CAPCO industry"],
            "parent_industry_code": ["", ""],
            "industry_level": ["1", "1"],
        }
    )
    industry_versions = pd.DataFrame(
        {
            "industry_standard_id": ["P0207", "P0221"],
            "implement_date": [
                pd.Timestamp("2012-10-26"),
                pd.Timestamp("2023-01-01"),
            ],
            "abandon_date": [pd.Timestamp("2022-12-31"), pd.NaT],
        }
    )

    result = compute_market_panel(
        bars,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=calendar,
        industry_history=history,
        industry_sets=industry_sets,
        industry_versions=industry_versions,
    )

    assert result["industry_standard_id"].tolist() == [
        "CSRC2012",
        "CAPCO",
        "CAPCO",
    ]
    assert result["industry_level1_name"].tolist() == [
        "CSRC industry",
        "CAPCO industry",
        "CAPCO industry",
    ]
    missing_transition = compute_market_panel(
        bars,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=calendar,
        industry_history=history.iloc[[0]],
        industry_sets=industry_sets.iloc[[0]],
        industry_versions=industry_versions.iloc[[0]],
    )
    assert (
        missing_transition.loc[
            missing_transition["timestamp"].ge(pd.Timestamp("2023-01-01")),
            "industry_standard_id",
        ]
        .isna()
        .all()
    )

    abandoned = compute_market_panel(
        bars,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=calendar,
        industry_history=history,
        industry_sets=industry_sets,
        industry_versions=industry_versions.assign(
            abandon_date=[pd.Timestamp("2023-01-04"), pd.Timestamp("2023-01-04")]
        ),
    )
    assert (
        abandoned.loc[
            abandoned["timestamp"].eq(pd.Timestamp("2023-01-04")),
            [
                "industry_standard_id",
                "industry_level1_code",
                "is_nonfinancial_industry",
            ],
        ]
        .isna()
        .all()
        .all()
    )


def test_industry_version_mask_preserves_interval_and_positional_boundaries():
    versions = pd.DataFrame(
        {
            "industry_standard_id": ["CSRC2012", "P0207", "CAPCO", "CSRC2012", "P0221"],
            "implement_date": pd.to_datetime(
                ["2010-01-01", "2012-01-01", "2023-01-01", "2011-06-01", "2020-01-01"]
            ),
            "abandon_date": pd.to_datetime(
                ["2012-01-01", "2014-01-01", None, "2011-09-01", "2020-01-01"]
            ),
        }
    )
    cases = (
        (" P0207 ", "2010-01-01", "2010-01-01", True, True),
        ("CSRC2012", "2009-12-31", "2010-01-01", False, False),
        ("CSRC2012", "2011-12-31", "2010-01-01", True, True),
        ("P0207", "2012-01-01", "2011-12-31", False, True),
        ("P0207", "2012-01-01", "2012-01-01", True, True),
        ("CSRC2012", "2014-01-01", "2013-12-31", False, False),
        ("CSRC2012", "2011-06-01", "2011-05-31", False, False),
        ("CSRC2012", "2011-09-01", "2011-06-01", False, True),
        ("CSRC2012", "2011-09-01", "2011-09-01", True, True),
        ("P0221", "2023-01-01", "2023-01-01", True, True),
        ("CAPCO", "2050-01-01", "2023-01-01", True, True),
        ("CAPCO", "2020-01-01", "2020-01-01", False, False),
        ("UNKNOWN", "2023-01-01", "2023-01-01", False, False),
        (None, "2023-01-01", "2023-01-01", False, False),
        ("CSRC2012", None, "2010-01-01", False, False),
        ("CSRC2012", "2010-01-01", None, False, True),
    )
    index = pd.Index(np.arange(len(cases)) * 3 + 1, name="source_row")
    standards = pd.Series([row[0] for row in cases], index=index)
    timestamps = pd.Series(pd.to_datetime([row[1] for row in cases]), index=index[::-1])
    events = pd.Series(pd.to_datetime([row[2] for row in cases]), index=index + 1000)
    before = [
        value.copy(deep=True) for value in (standards, timestamps, events, versions)
    ]
    actual = _industry_version_active_mask(
        standards, timestamps, versions, event_timestamps=events
    )
    pd.testing.assert_series_equal(
        actual, pd.Series([row[3] for row in cases], index=index, dtype=bool)
    )
    pd.testing.assert_series_equal(
        _industry_version_active_mask(standards, timestamps, versions),
        pd.Series([row[4] for row in cases], index=index, dtype=bool),
    )
    pd.testing.assert_series_equal(
        _industry_version_active_mask(
            standards, timestamps, versions.iloc[:0], event_timestamps=events
        ),
        pd.Series(False, index=index, dtype=bool),
    )
    pd.testing.assert_series_equal(
        _industry_version_active_mask(
            standards.iloc[:0],
            timestamps.iloc[:0],
            versions,
            event_timestamps=events.iloc[:0],
        ),
        pd.Series(False, index=index[:0], dtype=bool),
    )
    with pytest.raises(ValueError):
        _industry_version_active_mask(standards, timestamps.iloc[:-1], versions)
    with pytest.raises(ValueError):
        _industry_version_active_mask(
            standards, timestamps, versions, event_timestamps=events.iloc[:-1]
        )
    for original, current in zip(
        before[:3], (standards, timestamps, events), strict=True
    ):
        pd.testing.assert_series_equal(original, current)
    pd.testing.assert_frame_equal(before[3], versions)
