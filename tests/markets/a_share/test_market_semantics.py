from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.execution import (
    derive_execution_mask,
)
from quant_research_platform.markets.implementations.a_share.processed.panels import (
    compute_market_panel,
)
from quant_research_platform.research.labels.engine import calculate_labels_for_panel
from quant_research_platform.research.protocols.a_share import label_definitions


def test_market_panel_keeps_suspension_and_excludes_holiday_fill() -> None:
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
            "is_trading_day": [True] * 4,
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
    assert result["can_enter"].tolist() == [False, True, False]
    assert result["can_exit"].tolist() == [True, True, False]


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

    assert mask["can_enter"].tolist() == [False, True, True, False]
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
