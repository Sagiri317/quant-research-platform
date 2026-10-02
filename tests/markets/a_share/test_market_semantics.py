from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share.execution import (
    derive_execution_mask,
)
from quant_research_platform.markets.implementations.a_share.processed.panels import (
    _apply_special_treatment,
    _apply_suspension_semantics,
    _asof_industry,
    _industry_version_active_mask,
    compute_market_panel,
)
from quant_research_platform.research.labels.engine import calculate_labels_for_panel
from quant_research_platform.research.protocols.a_share import label_definitions


@pytest.mark.parametrize("has_raw_history", [False, True])
def test_missing_trading_prices_remain_missing_and_suspension_preserves_units(
    has_raw_history,
) -> None:
    from quant_research_platform.research.filtering import a_share_eligibility

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
    assert not a_share_eligibility(actual, "pv_all").iloc[1]
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


def test_industry_history_uses_effective_standard_and_set_hierarchy(tmp_path) -> None:
    sessions = pd.DatetimeIndex(
        pd.to_datetime(
            ["2022-12-30", "2023-01-03", "2023-04-28", "2023-05-04", "2023-05-05"]
        )
    )
    bars = pd.DataFrame(
        {
            "source_asset_id": ["000001"] * len(sessions),
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
            "timestamp": [pd.Timestamp("2022-12-29"), pd.Timestamp("2023-05-01")],
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
                pd.Timestamp("2023-05-01"),
            ],
            "abandon_date": [pd.NaT, pd.NaT],
        }
    )

    changes = pd.DataFrame(
        {
            "source_asset_id": history["source_asset_id"],
            "effective_date": history["timestamp"],
            "announcement_date": history["timestamp"] - pd.Timedelta(days=1),
            "changed_item": history["industry_standard_id"] + "行业代码",
            "value_after": history["industry_level1_name"],
        }
    )
    result = compute_market_panel(
        bars,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=calendar,
        industry_history=history,
        industry_changes=changes,
        industry_sets=industry_sets,
        industry_versions=industry_versions,
    )

    assert result["industry_standard_id"].tolist() == [
        "CSRC2012",
        "CSRC2012",
        "CSRC2012",
        "CAPCO",
        "CAPCO",
    ]
    assert result["industry_level1_name"].tolist() == [
        "CSRC industry",
        "CSRC industry",
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
        industry_changes=changes,
        industry_sets=industry_sets.iloc[[0]],
        industry_versions=industry_versions,
    )
    assert (
        missing_transition.loc[
            missing_transition["timestamp"].ge(pd.Timestamp("2023-05-01")),
            "industry_standard_id",
        ]
        .eq("CSRC2012")
        .all()
    )

    from quant_research_platform.profiles.research import research_profile
    from quant_research_platform.storage.datasets import read_dataset, write_dataset

    spec = research_profile().processed.get("market_panel_1d")
    from quant_research_platform.storage.layout import StorageLayout

    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(
        root, missing_transition.loc[:, list(spec.required_columns)], spec=spec
    )
    restored = read_dataset(root)
    assert len(restored) == len(sessions)
    for column in (
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    ):
        assert str(restored[column].dtype) == "string"
        assert restored[column].isna().sum() == 0
        pd.testing.assert_series_equal(restored[column], missing_transition[column])

    abandoned = compute_market_panel(
        bars,
        stock_master=stock_master,
        market_cap=market_cap,
        trading_calendar=calendar,
        industry_history=history,
        industry_changes=changes,
        industry_sets=industry_sets,
        industry_versions=industry_versions.assign(
            abandon_date=[pd.Timestamp("2023-05-05"), pd.Timestamp("2023-05-05")]
        ),
    )
    assert (
        abandoned.loc[
            abandoned["timestamp"].eq(pd.Timestamp("2023-05-05")),
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


@pytest.mark.parametrize("cutover", ["2023-05-01", "2024-02-01"])
def test_industry_cutover_uses_valid_versions_without_backfill(cutover) -> None:
    start = pd.Timestamp(cutover)
    retired = start + pd.Timedelta(days=30)
    resumed = start + pd.Timedelta(days=60)
    panel = pd.DataFrame(
        {
            "asset_id": ["000001"] * 4,
            "timestamp": pd.Series(
                [start - pd.Timedelta(days=1), start, retired, resumed],
                dtype="datetime64[ns]",
            ),
        }
    )
    for column in (
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    ):
        panel[column] = pd.Series(pd.NA, index=panel.index, dtype="string")
    panel["is_nonfinancial_industry"] = pd.Series(
        pd.NA, index=panel.index, dtype="boolean"
    )
    history = pd.DataFrame(
        {
            "source_asset_id": ["000001"] * 3,
            "timestamp": pd.Series(
                [start - pd.Timedelta(days=10), start - pd.Timedelta(days=2), start],
                dtype="datetime64[ns]",
            ),
            "industry_standard_id": ["P0207", "P0221", "P0221"],
            "industry_level1_code": ["A", "C", "B"],
            "industry_level1_name": ["old", "pre-effective", "current"],
        }
    )
    versions = pd.DataFrame(
        {
            "industry_standard_id": ["P0207", "P0221", "P0221", "P0221"],
            "implement_date": pd.Series(
                [
                    pd.Timestamp("2012-10-26"),
                    pd.Timestamp("2020-01-01"),
                    start,
                    resumed,
                ],
                dtype="datetime64[ns]",
            ),
            "abandon_date": pd.Series(
                [pd.NaT, pd.Timestamp("2020-01-01"), retired, pd.NaT],
                dtype="datetime64[ns]",
            ),
        }
    )
    changes = pd.DataFrame(
        {
            "source_asset_id": history["source_asset_id"],
            "effective_date": history["timestamp"],
            "announcement_date": history["timestamp"] - pd.Timedelta(days=1),
            "changed_item": history["industry_standard_id"] + "行业代码",
            "value_after": history["industry_level1_name"],
        }
    )
    before = [frame.copy(deep=True) for frame in (panel, history, versions)]

    result = _asof_industry(
        panel,
        history,
        industry_sets=None,
        industry_versions=versions,
        industry_changes=changes,
    )
    assert result["industry_standard_id"].iloc[:2].tolist() == ["CSRC2012", "CAPCO"]
    assert result["industry_level1_code"].iloc[:2].tolist() == ["A", "B"]
    assert result["industry_standard_id"].iloc[2:].isna().all()
    for actual, expected in zip((panel, history, versions), before, strict=True):
        pd.testing.assert_frame_equal(actual, expected)

    early_only = _asof_industry(
        panel,
        history.iloc[:2],
        industry_sets=None,
        industry_versions=versions,
        industry_changes=changes,
    )
    assert early_only["industry_standard_id"].tolist() == ["CSRC2012"] * len(panel)
    missing_versions = _asof_industry(
        panel, history, industry_sets=None, industry_versions=None
    )
    pd.testing.assert_frame_equal(missing_versions, panel)
    csrc_only = _asof_industry(
        panel,
        history.iloc[:1],
        industry_sets=None,
        industry_versions=versions.iloc[:1],
        industry_changes=changes,
    )
    assert csrc_only["industry_standard_id"].tolist() == ["CSRC2012"] * len(panel)


@pytest.mark.parametrize("provider", ["P0207", "P0221"])
@pytest.mark.parametrize("missing_name", [None, "event", "announcement", "both"])
def test_industry_publication_orders_known_events_and_unknown_coverage(
    provider,
    missing_name,
) -> None:
    dates = pd.to_datetime(
        [
            "2023-06-01",
            "2023-06-02",
            "2023-06-05",
            "2023-06-07",
            "2023-06-08",
            "2023-06-09",
        ]
    ).as_unit("ns")
    panel = pd.DataFrame({"asset_id": "000001", "timestamp": dates}).astype(
        {"asset_id": "object"}
    )
    for column in (
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    ):
        panel[column] = pd.Series(pd.NA, index=panel.index, dtype="string")
    panel["is_nonfinancial_industry"] = pd.Series(
        pd.NA, index=panel.index, dtype="boolean"
    )
    history = pd.DataFrame(
        {
            "source_asset_id": ["000001"] * 4,
            "timestamp": pd.to_datetime(
                ["2023-06-01", "2023-06-02", "2023-06-05", "2023-06-09"]
            ).as_unit("ns"),
            "industry_standard_id": provider,
            "industry_level1_code": ["A", "B", "C", "D"],
            "industry_level1_name": ["old", "late older", "unknown", "newer"],
        }
    )
    versions = pd.DataFrame(
        {
            "industry_standard_id": [provider],
            "implement_date": pd.to_datetime(["2011-01-01"]).as_unit("ns"),
            "abandon_date": pd.Series([pd.NaT], dtype="datetime64[ns]"),
        }
    )
    changes = pd.DataFrame(
        {
            "source_asset_id": history["source_asset_id"],
            "effective_date": history["timestamp"],
            "announcement_date": pd.to_datetime(
                ["2023-05-31", "2023-06-07", None, "2023-06-08"]
            ).as_unit("ns"),
            "changed_item": provider + "行业代码",
            "value_after": history["industry_level1_name"],
        }
    )
    if missing_name is not None:
        changes.loc[2, "announcement_date"] = pd.Timestamp("2023-06-04")
        if missing_name in {"event", "both"}:
            history.loc[2, "industry_level1_name"] = pd.NA
        if missing_name in {"announcement", "both"}:
            changes.loc[2, "value_after"] = pd.NA
    dictionary = pd.DataFrame(
        {
            "industry_standard_id": provider,
            "industry_code": ["A", "B", "D"],
            "industry_name": ["old", "late older", "newer"],
            "parent_industry_code": "",
            "industry_level": "1",
        }
    )
    before = [frame.copy(deep=True) for frame in (panel, history, versions, changes)]
    result = _asof_industry(
        panel,
        history,
        industry_sets=dictionary,
        industry_versions=versions,
        industry_changes=changes,
    )
    assert result["industry_level1_code"].fillna("NULL").tolist() == [
        "A",
        "A",
        "NULL",
        "NULL",
        "NULL",
        "D",
    ]
    industry_columns = [
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
        "is_nonfinancial_industry",
    ]
    assert result.loc[2:4, industry_columns].isna().all().all()
    pd.testing.assert_frame_equal(
        result[["asset_id", "timestamp"]], panel[["asset_id", "timestamp"]]
    )
    for actual, expected in zip(
        (panel, history, versions, changes), before, strict=True
    ):
        pd.testing.assert_frame_equal(actual, expected)

    known_future = _asof_industry(
        panel,
        history.iloc[:2],
        industry_sets=None,
        industry_versions=versions,
        industry_changes=changes.iloc[:2],
    )
    assert known_future["industry_level1_code"].tolist() == [
        "A",
        "A",
        "A",
        "A",
        "B",
        "B",
    ]
    missing_initial = _asof_industry(
        panel,
        history,
        industry_sets=None,
        industry_versions=versions,
        industry_changes=changes.iloc[0:0],
    )
    assert missing_initial[industry_columns].isna().all().all()
    wrong_item = _asof_industry(
        panel,
        history,
        industry_sets=None,
        industry_versions=versions,
        industry_changes=changes.assign(changed_item=provider),
    )
    assert wrong_item[industry_columns].isna().all().all()
    wrong_value = _asof_industry(
        panel,
        history.iloc[:1],
        industry_sets=None,
        industry_versions=versions,
        industry_changes=changes.iloc[:1].assign(value_after="different native name"),
    )
    assert wrong_value[industry_columns].isna().all().all()


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


@pytest.mark.parametrize("reverse", [False, True])
def test_special_treatment_pit_and_late_older_events_persist(tmp_path, reverse):
    from quant_research_platform.sources.csmar.interim import (
        interim_catalog,
        normalize_raw_frame,
    )
    from quant_research_platform.storage.datasets import read_dataset
    from quant_research_platform.storage.datasets.writing import write_dataset
    from quant_research_platform.storage.layout import StorageLayout

    raw = pd.DataFrame(
        {
            "Symbol": ["000001"] * 3,
            "Chgtype": ["AB", "BA", "AD"],
            "Annoudt": ["2020-01-02", "2020-01-03", "2020-01-05"],
            "Execudt": ["2020-01-01", "2020-01-03", "2020-01-02"],
        }
    )
    history = normalize_raw_frame(
        raw.iloc[::-1] if reverse else raw, "special_treatment_changes"
    )
    spec = interim_catalog().get("special_treatment_changes")
    layout = StorageLayout(project_root=tmp_path)
    root = layout.target_root(
        spec.data_id, layout.interim_data_root(spec.source_id, spec.data_id)
    )
    write_dataset(root, history, spec=spec, resource_policy=None, borrowed_input=True)
    restored = read_dataset(root)
    panel = pd.DataFrame(
        {
            "asset_id": "000001",
            "timestamp": pd.date_range("2020-01-01", periods=6).as_unit("ns"),
            "trading_status_code": 0,
        }
    )
    actual = _apply_special_treatment(panel, restored)
    assert actual["is_st"].tolist() == [False, True, False, False, False, False]
    assert actual["is_pt"].tolist() == [False] * 6
    assert str(restored["is_st"].dtype) == "boolean"


def test_special_treatment_equal_effective_conflict_is_explicit():
    from quant_research_platform.sources.csmar.interim import normalize_raw_frame

    history = normalize_raw_frame(
        pd.DataFrame(
            {
                "Symbol": ["000001", "000001"],
                "Chgtype": ["AB", "BA"],
                "Annoudt": ["2020-01-01", "2020-01-02"],
                "Execudt": ["2020-01-01"] * 2,
            }
        ),
        "special_treatment_changes",
    )
    panel = pd.DataFrame(
        {
            "asset_id": ["000001"],
            "timestamp": pd.to_datetime(["2020-01-01"]).as_unit("ns"),
            "trading_status_code": [0],
        }
    )
    assert _apply_special_treatment(panel, history)["is_st"].tolist() == [True]
    panel["timestamp"] = pd.to_datetime(["2020-01-02"]).as_unit("ns")
    with pytest.raises(ValueError, match="special_treatment_effective_state_ambiguous"):
        _apply_special_treatment(panel, history)


@pytest.mark.parametrize("reverse", [False, True])
def test_special_treatment_shared_availability_and_repeated_state(reverse):
    from quant_research_platform.sources.csmar.interim import normalize_raw_frame

    raw = pd.DataFrame(
        {
            "Symbol": ["000001"] * 3,
            "Chgtype": ["AB", "DB", "BC"],
            "Annoudt": ["2020-01-02", "2020-01-03", "2020-01-03"],
            "Execudt": ["2020-01-01", "2020-01-01", "2020-01-02"],
        }
    )
    history = normalize_raw_frame(
        raw.iloc[::-1] if reverse else raw, "special_treatment_changes"
    )
    panel = pd.DataFrame(
        {
            "asset_id": "000001",
            "timestamp": pd.date_range("2020-01-01", periods=3).as_unit("ns"),
            "trading_status_code": 0,
        }
    )
    actual = _apply_special_treatment(panel, history)
    assert actual["is_st"].tolist() == [False, True, False]
    assert actual["is_pt"].tolist() == [False, False, True]


def test_public_snapshots_preserve_unknown_barriers_and_actual_publication():
    from quant_research_platform.markets.implementations.a_share.processed import panels

    dates = pd.DatetimeIndex(
        ["2023-01-02", "2023-01-03", "2023-01-05", "2023-01-11", "2023-01-12"],
        dtype="datetime64[ns]",
    )
    panel = pd.DataFrame({"asset_id": "000001", "timestamp": dates})
    for column in (
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    ):
        panel[column] = pd.Series(pd.NA, index=panel.index, dtype="string")
    panel["is_nonfinancial_industry"] = pd.Series(
        pd.NA, index=panel.index, dtype="boolean"
    )
    history = pd.DataFrame(
        {
            "source_asset_id": ["000001", "000001"],
            "timestamp": pd.to_datetime(["2023-01-01", "2023-01-05"]),
            "industry_standard_id": ["CSRC2012", "CAPCO"],
            "industry_level1_code": ["A01", "B01"],
            "industry_level1_name": ["old leaf", "new leaf"],
        }
    )
    changes = pd.DataFrame(
        {
            "source_asset_id": ["000001", "000001"],
            "changed_item": ["P0207行业代码", "P0221行业代码"],
            "effective_date": history["timestamp"],
            "announcement_date": pd.to_datetime(["2023-01-01", None]),
            "value_after": history["industry_level1_name"],
        }
    )
    versions = pd.DataFrame(
        {
            "industry_standard_id": ["CSRC2012", "CAPCO"],
            "implement_date": pd.to_datetime(["2012-10-26", "2023-01-03"]),
            "abandon_date": pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]"),
        }
    )
    dictionary = pd.DataFrame(
        {
            "industry_standard_id": ["CSRC2012"] * 2 + ["CAPCO"] * 5,
            "industry_code": ["A", "A01", "B", "B01", "C", "CA", "C01"],
            "industry_name": [
                "old root",
                "old leaf",
                "new root",
                "new leaf",
                "latest root",
                "manufacturing major",
                "latest leaf",
            ],
            "parent_industry_code": ["", "A", "", "B", "", "C", "CA"],
            "industry_level": ["1", "2", "1", "3", "1", "2", "3"],
        }
    )
    stocks = panels._canonical_stock_master(
        pd.DataFrame(
            {
                "source_asset_id": ["000001"],
                "listing_date": pd.to_datetime(["2010-01-01"]),
                "market_type": [1],
            }
        )
    )
    snapshots = pd.DataFrame(
        {
            "source_asset_id": ["000001", "000001"],
            "timestamp": pd.to_datetime(["2023-01-11", "2023-01-12"]),
            "observation_date": pd.to_datetime(["2023-01-04", "2023-01-07"]),
            "industry_standard_id": ["CAPCO", "CAPCO"],
            "industry_category_code": ["B", "C"],
            "industry_major_code": ["", "CA"],
            "industry_code": ["01", "01"],
        }
    )
    before = snapshots.copy(deep=True)
    options = dict(
        industry_sets=dictionary,
        industry_versions=versions,
        industry_changes=changes,
        stock_master=stocks,
    )
    result = panels._asof_industry(
        panel, history, industry_snapshots=snapshots, **options
    )
    assert result["industry_level1_code"].fillna("unknown").tolist() == [
        "A",
        "A",
        "unknown",
        "unknown",
        "C",
    ]
    assert result["industry_standard_id"].fillna("unknown").tolist() == [
        "CSRC2012",
        "CSRC2012",
        "unknown",
        "unknown",
        "CAPCO",
    ]
    earlier = panels._asof_industry(
        panel.iloc[:4], history, industry_snapshots=snapshots.iloc[:1], **options
    )
    pd.testing.assert_frame_equal(result.iloc[:4], earlier)
    pd.testing.assert_frame_equal(snapshots, before)
    conflicting = snapshots.iloc[[1]].assign(
        industry_category_code="B", industry_major_code=""
    )
    with pytest.raises(ValueError, match="industry_history_conflict"):
        panels._asof_industry(
            panel,
            history,
            industry_snapshots=pd.concat([snapshots, conflicting]),
            **options,
        )

    duplicate = snapshots.iloc[[1]].assign(timestamp=pd.Timestamp("2023-01-11"))
    repeated = panels._asof_industry(
        panel,
        history,
        industry_snapshots=pd.concat([snapshots, duplicate], ignore_index=True),
        **options,
    )
    assert repeated.loc[
        repeated.timestamp.eq(pd.Timestamp("2023-01-11")), "industry_level1_code"
    ].tolist() == ["C"]
    same_parent_leaf = dictionary.iloc[[-1]].assign(
        industry_code="C02", industry_name="other leaf"
    )
    same_parent = snapshots.iloc[[1]].assign(industry_code="02")
    with pytest.raises(ValueError, match="industry_history_conflict"):
        panels._asof_industry(
            panel,
            history,
            industry_snapshots=pd.concat([snapshots, same_parent], ignore_index=True),
            **{
                **options,
                "industry_sets": pd.concat(
                    [dictionary, same_parent_leaf], ignore_index=True
                ),
            },
        )
