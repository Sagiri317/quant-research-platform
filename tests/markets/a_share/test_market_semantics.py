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
    _canonical_stock_master,
    _industry_version_active_mask,
    _prepare_industry_inputs,
    _prepare_market_panel_inputs,
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
            "has_252_trading_sessions": True,
        }
    )
    frame["listing_date"] = frame["timestamp"].min()
    actual = _apply_suspension_semantics(frame)
    assert actual.loc[1, ["close", "open", "high", "low", "adj_open"]].isna().all()
    # Research membership does not certify a usable price or an executable trade.
    assert a_share_eligibility(actual, "a_share_all").iloc[1]
    assert actual.loc[1, "period_return"] == 0.07
    assert actual.loc[2, "adj_close"] == actual.loc[2, "adj_open"] == 110.0
    raw = actual.loc[2, ["close", "open", "high", "low"]]
    # The missing trading quote resets raw prices without resetting adjusted prices.
    assert raw.isna().all()


@pytest.mark.parametrize("missing_caps", [False, True])
@pytest.mark.parametrize("other_exchange_open", [False, True])
def test_market_panel_keeps_suspension_and_excludes_holiday_fill(
    missing_caps, other_exchange_open
) -> None:
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

    if other_exchange_open:
        calendar = pd.concat(
            [calendar, calendar.assign(source_market_type=4, is_trading_day=True)],
            ignore_index=True,
        )

    result = compute_market_panel(
        bars,
        **_prepare_market_panel_inputs(
            stocks=_canonical_stock_master(stock_master),
            market_cap=market_cap,
            trading_calendar=calendar,
        ),
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
            **_prepare_market_panel_inputs(
                stocks=_canonical_stock_master(stock_master),
                market_cap=market_cap,
                trading_calendar=calendar,
            ),
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
        **_prepare_market_panel_inputs(
            stocks=_canonical_stock_master(stock_master),
            market_cap=market_cap,
            trading_calendar=calendar,
        ),
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
            "listing_date": pd.Timestamp("2000-01-01"),
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


def test_execution_mask_and_label_consumer_have_independent_price_eligibility() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=4)
    panel = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "market_id": "a_share",
            "asset_id": "asset",
            "timestamp": sessions,
            "is_trading": True,
            "filling_flag": 0,
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
        panel,
        definitions=(label_definitions()[0],),
        sessions=sessions,
    )

    assert mask["can_enter"].iloc[:3].tolist() == [False, True, True]
    assert pd.isna(mask["can_enter"].iloc[3])
    assert labeled.loc[0, "forward_return_1d"] == pytest.approx(12.0 / 11.0 - 1.0)
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
        **_prepare_market_panel_inputs(
            stocks=_canonical_stock_master(stock_master),
            market_cap=market_cap,
            trading_calendar=calendar,
            no_limit_history=no_limit,
        ),
    )
    mask = derive_execution_mask(panel)

    assert panel.loc[panel["timestamp"].eq(sessions[1]), "limit_status"].iloc[0] == 0.0
    assert mask.loc[mask["timestamp"].eq(sessions[0]), "can_enter"].iloc[0]


def test_missing_listing_lifecycle_retains_prices_without_eligibility() -> None:
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
        **_prepare_market_panel_inputs(
            stocks=_canonical_stock_master(
                pd.DataFrame(
                    columns=["source_asset_id", "listing_date", "delisting_date"]
                )
            ),
            market_cap=pd.DataFrame(
                {"source_asset_id": ["000001"], "timestamp": [sessions[0]]}
            ),
            trading_calendar=calendar,
        ),
    )

    assert result.close.tolist() == [10.0, 10.0]
    assert result.listing_date.isna().all()
    assert result.delisting_date_unresolved.all()
    assert result.has_252_trading_sessions.isna().all()
    assert derive_execution_mask(result)[["can_enter", "can_exit"]].isna().all().all()


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
    calendar_dates = pd.date_range(sessions[0], sessions[-1]).as_unit("ns")
    calendar = pd.DataFrame(
        {
            "calendar_date": calendar_dates,
            "source_market_type": 1,
            "is_trading_day": calendar_dates.isin(sessions),
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
        **_prepare_market_panel_inputs(
            stocks=_canonical_stock_master(stock_master),
            market_cap=market_cap,
            trading_calendar=calendar,
            industry_history=history,
            industry_changes=changes,
            industry_sets=industry_sets,
            industry_versions=industry_versions,
        ),
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
        **_prepare_market_panel_inputs(
            stocks=_canonical_stock_master(stock_master),
            market_cap=market_cap,
            trading_calendar=calendar,
            industry_history=history.iloc[[0]],
            industry_changes=changes,
            industry_sets=industry_sets.iloc[[0]],
            industry_versions=industry_versions,
        ),
    )
    assert (
        missing_transition.loc[
            missing_transition["timestamp"].ge(pd.Timestamp("2023-05-01")),
            "industry_standard_id",
        ]
        .eq("CSRC2012")
        .all()
    )

    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.a_share_catalogs import scientific_catalogs

    spec = scientific_catalogs().processed.get("market_panel_1d")
    from tests.helpers.paths import StorageLayout

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
        **_prepare_market_panel_inputs(
            stocks=_canonical_stock_master(stock_master),
            market_cap=market_cap,
            trading_calendar=calendar,
            industry_history=history,
            industry_changes=changes,
            industry_sets=industry_sets,
            industry_versions=industry_versions.assign(
                abandon_date=[pd.Timestamp("2023-05-05"), pd.Timestamp("2023-05-05")]
            ),
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
        **_prepare_industry_inputs(
            history,
            industry_sets=None,
            industry_versions=versions,
            industry_changes=changes,
        ),
    )
    assert result["industry_standard_id"].iloc[:2].tolist() == ["CSRC2012", "CAPCO"]
    assert result["industry_level1_code"].iloc[:2].tolist() == ["A", "B"]
    assert result["industry_standard_id"].iloc[2:].isna().all()
    for actual, expected in zip((panel, history, versions), before, strict=True):
        pd.testing.assert_frame_equal(actual, expected)

    early_only = _asof_industry(
        panel,
        **_prepare_industry_inputs(
            history.iloc[:2],
            industry_sets=None,
            industry_versions=versions,
            industry_changes=changes,
        ),
    )
    assert early_only["industry_standard_id"].tolist() == ["CSRC2012"] * len(panel)
    missing_versions = _asof_industry(
        panel,
        **_prepare_industry_inputs(history, industry_sets=None, industry_versions=None),
    )
    pd.testing.assert_frame_equal(missing_versions, panel)
    csrc_only = _asof_industry(
        panel,
        **_prepare_industry_inputs(
            history.iloc[:1],
            industry_sets=None,
            industry_versions=versions.iloc[:1],
            industry_changes=changes,
        ),
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
        **_prepare_industry_inputs(
            history,
            industry_sets=dictionary,
            industry_versions=versions,
            industry_changes=changes,
        ),
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

    # These auxiliary facts do not depend on the primary row-2 missingness case.
    if missing_name is not None:
        return

    known_future = _asof_industry(
        panel,
        **_prepare_industry_inputs(
            history.iloc[:2],
            industry_sets=None,
            industry_versions=versions,
            industry_changes=changes.iloc[:2],
        ),
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
        **_prepare_industry_inputs(
            history,
            industry_sets=None,
            industry_versions=versions,
            industry_changes=changes.iloc[0:0],
        ),
    )
    assert missing_initial[industry_columns].isna().all().all()
    wrong_item = _asof_industry(
        panel,
        **_prepare_industry_inputs(
            history,
            industry_sets=None,
            industry_versions=versions,
            industry_changes=changes.assign(changed_item=provider),
        ),
    )
    assert wrong_item[industry_columns].isna().all().all()
    wrong_value = _asof_industry(
        panel,
        **_prepare_industry_inputs(
            history.iloc[:1],
            industry_sets=None,
            industry_versions=versions,
            industry_changes=changes.iloc[:1].assign(
                value_after="different native name"
            ),
        ),
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
    from tests.helpers.paths import StorageLayout

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
    root = layout.interim_data_root(spec.source_id, spec.data_id)
    write_dataset(
        root,
        history,
        spec=spec,
    )
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


def test_special_treatment_equal_effective_conflict_is_local_to_unknown_field():
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
    actual = _apply_special_treatment(panel, history)
    assert not actual.is_st.iloc[0]  # independent current-day normal status
    assert not actual.is_pt.iloc[0]  # agreeing event fact
    panel["trading_status_code"] = pd.Series(pd.NA, index=panel.index, dtype="Int64")
    actual = _apply_special_treatment(panel, history)
    assert pd.isna(actual.is_st.iloc[0])
    assert not actual.is_pt.iloc[0]


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
        panel,
        **panels._prepare_industry_inputs(
            history, industry_snapshots=snapshots, **options
        ),
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
        panel.iloc[:4],
        **panels._prepare_industry_inputs(
            history, industry_snapshots=snapshots.iloc[:1], **options
        ),
    )
    pd.testing.assert_frame_equal(result.iloc[:4], earlier)
    pd.testing.assert_frame_equal(snapshots, before)
    conflicting = snapshots.iloc[[1]].assign(
        industry_category_code="B", industry_major_code=""
    )
    industry_columns = [
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
        "is_nonfinancial_industry",
    ]
    mixed_panel = pd.concat(
        [panel.assign(close=10.0), panel.assign(asset_id="000002", close=20.0)],
        ignore_index=True,
    )
    mixed_options = {
        **options,
        "industry_changes": pd.concat(
            [changes, changes.iloc[[0]].assign(source_asset_id="000002")],
            ignore_index=True,
        ),
    }
    mixed_history = pd.concat(
        [history, history.iloc[[0]].assign(source_asset_id="000002")],
        ignore_index=True,
    )
    conflicted = panels._asof_industry(
        mixed_panel,
        **panels._prepare_industry_inputs(
            mixed_history,
            industry_snapshots=pd.concat([snapshots, conflicting]),
            **mixed_options,
        ),
    )
    affected = conflicted.query("asset_id == '000001'").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        affected.iloc[:4].drop(columns="close"), result.iloc[:4]
    )
    assert affected.iloc[[4]][industry_columns].isna().all().all()
    assert (
        conflicted.query("asset_id == '000002'").industry_level1_code.tolist()
        == ["A"] * 5
    )
    pd.testing.assert_series_equal(conflicted.close, mixed_panel.close)

    duplicate = snapshots.iloc[[1]].assign(timestamp=pd.Timestamp("2023-01-11"))
    repeated = panels._asof_industry(
        panel,
        **panels._prepare_industry_inputs(
            history,
            industry_snapshots=pd.concat([snapshots, duplicate], ignore_index=True),
            **options,
        ),
    )
    assert repeated.loc[
        repeated.timestamp.eq(pd.Timestamp("2023-01-11")), "industry_level1_code"
    ].tolist() == ["C"]
    same_parent_leaf = dictionary.iloc[[-1]].assign(
        industry_code="C02", industry_name="other leaf"
    )
    same_parent = snapshots.iloc[[1]].assign(industry_code="02")
    same_parent_conflict = panels._asof_industry(
        mixed_panel,
        **panels._prepare_industry_inputs(
            mixed_history,
            industry_snapshots=pd.concat([snapshots, same_parent], ignore_index=True),
            **{
                **mixed_options,
                "industry_sets": pd.concat(
                    [dictionary, same_parent_leaf], ignore_index=True
                ),
            },
        ),
    )
    affected = same_parent_conflict.query("asset_id == '000001'").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        affected.iloc[:4].drop(columns="close"), result.iloc[:4]
    )
    assert affected.iloc[[4]][industry_columns].isna().all().all()
    assert (
        same_parent_conflict.query("asset_id == '000002'").industry_level1_code.tolist()
        == ["A"] * 5
    )
    pd.testing.assert_series_equal(same_parent_conflict.close, mixed_panel.close)


@pytest.mark.parametrize("damage", ("missing", "cycle", "conflict"))
def test_industry_mapping_gap_preserves_panel_with_local_unknowns(tmp_path, damage):
    from quant_research_platform.markets.implementations.a_share.processed import (
        producer,
    )
    from quant_research_platform.storage.datasets import read_dataset, write_dataset
    from tests.helpers.a_share_catalogs import scientific_catalogs
    from tests.helpers.a_share_sources import _synthetic_interim_frames

    dates = pd.bdate_range("2020-01-02", periods=3)
    source = _synthetic_interim_frames(dates)
    dictionary = source["industry_classification_sets"]
    if damage == "missing":
        source["industry_classification_sets"] = dictionary.iloc[:0].copy()
    elif damage == "cycle":
        source["industry_classification_sets"] = pd.concat(
            [
                dictionary.assign(industry_level="2", parent_industry_code="B"),
                dictionary.assign(
                    industry_code="B", industry_level="2", parent_industry_code="A01"
                ),
            ],
            ignore_index=True,
        )
    else:
        source["industry_classification_sets"] = pd.concat(
            [dictionary, dictionary.assign(industry_name="conflict")], ignore_index=True
        )
    if damage != "missing":
        source["industry_classification_sets"] = pd.concat(
            [
                source["industry_classification_sets"],
                dictionary.assign(
                    industry_code="J",
                    industry_name="finance",
                    industry_level="1",
                    parent_industry_code="",
                ),
                dictionary.assign(
                    industry_code="J66",
                    industry_name="banking",
                    industry_level="2",
                    parent_industry_code="J",
                ),
            ],
            ignore_index=True,
        )
        history = source["industry_classification"]
        history.loc[
            history.source_asset_id.ne("000001"),
            ["industry_level1_code", "industry_level1_name"],
        ] = ["J66", "banking"]
        changes = source["stock_info_changes"]
        changes.loc[changes.source_asset_id.ne("000001"), "value_after"] = "banking"
    spec = scientific_catalogs().processed.get("market_panel_1d")

    def compute():
        return producer._compute_processed_frame(
            spec,
            source={
                **producer._prepare_panel_source(
                    source,
                    producer.panels._canonical_stock_master(source["stock_master"]),
                ),
                "market_bars_1d": source["market_bars_1d"],
            },
            dependencies={},
            trading_sessions=dates,
        )

    frame = compute()
    from tests.helpers.paths import StorageLayout

    root = StorageLayout(tmp_path).processed_data_root(
        spec.market_id, spec.kind, spec.data_id
    )
    write_dataset(root, frame, spec=spec)
    restored = read_dataset(root)
    assert len(restored) == 9
    affected = (
        restored if damage == "missing" else restored.query("asset_id == '000001'")
    )
    assert (
        affected[
            [
                "industry_standard_id",
                "industry_level1_code",
                "industry_level1_name",
                "is_nonfinancial_industry",
            ]
        ]
        .isna()
        .all()
        .all()
    )
    if damage != "missing":
        assert (
            restored.query("asset_id != '000001'").industry_level1_code.tolist()
            == ["J"] * 6
        )
    price_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_open",
        "adj_close",
        "volume",
        "amount",
    ]
    pd.testing.assert_frame_equal(
        restored.set_index(["asset_id", "timestamp"])[price_columns]
        .sort_index()
        .astype("float64"),
        source["market_bars_1d"]
        .set_index(["source_asset_id", "timestamp"])[price_columns]
        .sort_index()
        .rename_axis(["asset_id", "timestamp"]),
        check_index_type=False,
    )


def test_public_suspension_placeholders_keep_fixed_horizon_returns(tmp_path):
    from quant_research_platform.storage.datasets import read_dataset
    from tests.helpers.a_share_sources import (
        _synthetic_interim_frames,
    )

    dates = pd.bdate_range("2020-01-02", periods=9).as_unit("ns")
    frames = _synthetic_interim_frames(dates)
    bars = frames["market_bars_1d"]
    asset = bars.source_asset_id.eq("000001")
    prices = ["open", "high", "low", "close", "adj_open", "adj_close"]
    bars.loc[asset, prices] = 100.0
    suspended = asset & bars.timestamp.between(dates[2], dates[4])
    bars.loc[suspended, "filling_flag"] = 1
    bars.loc[suspended, prices] = 50.0
    bars.loc[asset & bars.timestamp.ge(dates[5]), prices] = 120.0
    from quant_research_platform.markets.prepare import prepare
    from quant_research_platform.storage.datasets import write_dataset
    from tests.helpers.a_share_catalogs import scientific_catalogs

    profile = scientific_catalogs()
    paths = {}
    for data_id, frame in frames.items():
        spec = profile.interim.get(data_id)
        path = tmp_path / data_id
        write_dataset(path, frame.loc[:, list(spec.required_columns)], spec=spec)
        paths[data_id] = str(path)
    window = {"start": str(dates[0].date()), "end": str(dates[-1].date())}
    panel_spec = profile.processed.get("market_panel_1d")
    panel = prepare(
        {
            "market_id": "a_share",
            "data_id": "market_panel_1d",
            "inputs": {
                key: paths[key]
                for key in panel_spec.source_input_data_ids
                if key in paths
            },
            "time_window": window,
            "output_root": str(tmp_path / "panel"),
        }
    )
    execution = prepare(
        {
            "market_id": "a_share",
            "data_id": "execution_mask",
            "inputs": {
                "market_panel_1d": str(panel),
                "trading_calendar": paths["trading_calendar"],
            },
            "time_window": window,
            "output_root": str(tmp_path / "execution"),
        }
    )
    root = prepare(
        {
            "market_id": "a_share",
            "data_id": "forward_returns",
            "inputs": {
                "market_panel_1d": str(panel),
                "trading_calendar": paths["trading_calendar"],
            },
            "time_window": window,
            "output_root": str(tmp_path / "labels"),
        }
    )
    saved = read_dataset(root)
    row = saved.loc[saved.asset_id.eq("000001") & saved.timestamp.eq(dates[0])].iloc[0]
    assert pd.isna(row.forward_return_1d)
    assert pd.isna(row.forward_return_1d_available_at)
    masks = read_dataset(execution)
    entry_into_pause = masks.loc[
        masks.asset_id.eq("000001") & masks.timestamp.eq(dates[1])
    ].iloc[0]
    assert not entry_into_pause.can_enter
    assert row.forward_return_5d == pytest.approx(0.2)
    assert row.forward_return_5d_available_at == dates[6] + pd.Timedelta(hours=7)


def test_listing_age_keeps_unavailable_master_date_unknown():
    from quant_research_platform.markets.implementations.a_share.calendar import (
        compute_listing_session_facts,
    )

    dates = pd.bdate_range("2020-01-02", periods=3)
    calendar = pd.DataFrame(
        {"calendar_date": dates, "source_market_type": 1, "is_trading_day": True}
    )
    observed = pd.DataFrame(
        {
            "asset_id": ["unknown", "known", "future"],
            "timestamp": dates[1],
            "market_type": 1,
            "listing_date": [pd.NaT, dates[0], dates[2]],
        }
    )
    history = pd.DataFrame(
        {
            "asset_id": ["unknown", "known", "future"],
            "timestamp": dates[0],
            "market_type": 1,
        }
    )
    facts = compute_listing_session_facts(observed, calendar, history)
    ages = facts["listing_age_trading_days"]
    assert facts["has_252_trading_sessions"].iloc[1:].tolist() == [False, False]
    assert pd.isna(ages.iloc[0])
    assert ages.iloc[1:].tolist() == [2, 0]


def test_listing_age_history_accepts_calendar_codes_and_absent_closed_assets():
    from quant_research_platform.markets.implementations.a_share.calendar import (
        compute_listing_session_facts,
    )

    dates = pd.bdate_range("2020-01-02", periods=3).as_unit("ns")
    calendar_dates = pd.date_range(dates[0], dates[-1]).as_unit("ns")
    calendar = pd.DataFrame(
        {
            "calendar_date": calendar_dates.tolist() * 2,
            "source_market_type": ["1"] * len(calendar_dates)
            + ["4"] * len(calendar_dates),
            "is_trading_day": [False] * len(calendar_dates)
            + calendar_dates.isin(dates).tolist(),
        }
    )
    observed = pd.DataFrame(
        {
            "asset_id": ["moved"],
            "timestamp": dates[-1],
            "listing_date": dates[0],
            "market_type": 4,
        },
        index=[7],
    )
    history = pd.DataFrame(
        {
            "asset_id": ["absent", "moved", "moved"],
            "timestamp": [dates[0], dates[0], dates[1]],
            "market_type": [1, 1, 4],
        }
    )
    ages = compute_listing_session_facts(
        observed,
        calendar,
        market_history=history,
    )
    pd.testing.assert_series_equal(
        ages["listing_age_trading_days"],
        pd.Series([2], index=[7], dtype="Int64", name="listing_age_trading_days"),
    )
    assert not ages["has_252_trading_sessions"].iloc[0]


def test_listing_facts_do_not_call_partial_market_calendar_an_exact_age():
    from quant_research_platform.markets.implementations.a_share.calendar import (
        compute_listing_session_facts,
    )

    dates = pd.bdate_range("2020-01-02", periods=260).as_unit("ns")
    calendar = pd.DataFrame(
        {
            "calendar_date": dates.tolist() * 2,
            "source_market_type": [1] * 260 + [4] * 260,
            "is_trading_day": True,
        }
    )
    calendar = calendar.loc[
        ~(calendar.source_market_type.eq(1) & calendar.calendar_date.eq(dates[4]))
    ]
    frame = pd.DataFrame(
        {
            "asset_id": ["partial", "unknown_ipo"],
            "timestamp": dates[-1],
            "listing_date": [dates[0], pd.NaT],
        }
    )
    history = pd.DataFrame(
        {
            "asset_id": ["partial", "unknown_ipo"],
            "timestamp": dates[0],
            "market_type": [1, 4],
        }
    )
    facts = compute_listing_session_facts(frame, calendar, history)
    assert facts.listing_age_trading_days.isna().all()
    assert facts.has_252_trading_sessions.tolist() == [True, True]


def test_industry_batch_join_preserves_asset_order_and_unknown_barriers():
    dates = pd.date_range("2020-01-01", periods=3).as_unit("ns")
    panel = pd.DataFrame(
        {
            "asset_id": ["000002", "000001", "000003", "000002", "000001", "000002"],
            "timestamp": [dates[1], dates[1], dates[0], dates[0], dates[0], dates[2]],
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
            "source_asset_id": ["000001", "000002", "000002"],
            "timestamp": [dates[0], dates[0], dates[1]],
            "industry_standard_id": "P0207",
            "industry_level1_code": ["A", "B", "C"],
            "industry_level1_name": ["first", "second", "unknown"],
        }
    )
    versions = pd.DataFrame(
        {
            "industry_standard_id": ["P0207"],
            "implement_date": pd.to_datetime(["2011-01-01"]).as_unit("ns"),
            "abandon_date": pd.Series([pd.NaT], dtype="datetime64[ns]"),
        }
    )
    changes = pd.DataFrame(
        {
            "source_asset_id": history.source_asset_id.iloc[:2],
            "effective_date": history.timestamp.iloc[:2],
            "announcement_date": dates[0] - pd.Timedelta(days=1),
            "changed_item": "P0207行业代码",
            "value_after": history.industry_level1_name.iloc[:2],
        }
    )
    before = panel.copy(deep=True)
    result = _asof_industry(
        panel,
        **_prepare_industry_inputs(
            history, industry_versions=versions, industry_changes=changes
        ),
    )
    assert result.asset_id.tolist() == ["000002"] * 3 + ["000001"] * 2 + ["000003"]
    assert result.timestamp.tolist() == [*dates, dates[0], dates[1], dates[0]]
    assert result.industry_level1_code.fillna("NULL").tolist() == [
        "B",
        "NULL",
        "NULL",
        "A",
        "A",
        "NULL",
    ]
    assert not result.duplicated(["asset_id", "timestamp"]).any()
    assert str(result.industry_level1_code.dtype) == "string"
    assert str(result.is_nonfinancial_industry.dtype) == "boolean"
    pd.testing.assert_frame_equal(panel, before, check_exact=True)


@pytest.mark.parametrize(
    "delisting, error",
    [
        ("2020-01-09", "conflicting_stock_master_lifecycles"),
        (None, "conflicting_stock_master_lifecycles"),
        ("2020-01-01", "invalid_stock_master_lifecycle"),
    ],
)
def test_conflicting_listing_lifecycles_are_rejected(delisting, error):
    from quant_research_platform.markets.implementations.a_share.processed import panels

    with pytest.raises(ValueError, match=error):
        panels._canonical_stock_master(
            pd.DataFrame(
                {
                    "source_asset_id": ["000001", "000001"],
                    "listing_date": ["2020-01-02", "2020-01-09"],
                    "delisting_date": [delisting, None],
                }
            )
        )


@pytest.mark.parametrize(
    "cap,close,volume", ((1e308, 1e-308, 1e308), (1e-308, 1e308, 1e-308))
)
def test_turnover_does_not_lose_finite_value_to_intermediate_share_count(
    cap, close, volume
):
    from fractions import Fraction

    frame = pd.DataFrame(
        {
            "asset_id": "asset",
            "listing_date": pd.Timestamp("2000-01-01"),
            "timestamp": pd.date_range("2020-01-01", periods=2),
            "filling_flag": [0, 1],
            "market_type": 1,
            "close": close,
            "open": close,
            "high": close,
            "low": close,
            "adj_close": 10.0,
            "adj_open": 10.0,
            "period_return": 0.0,
            "volume": volume,
            "amount": 1000.0,
            "float_market_cap": cap,
        }
    )
    actual = _apply_suspension_semantics(frame)
    expected = float(Fraction(volume) * Fraction(close) / Fraction(cap))
    assert np.isfinite(actual.turnover.iloc[0])
    assert actual.turnover.iloc[0] == pytest.approx(expected, rel=1e-15, abs=0)
    assert actual.turnover.iloc[1] == 0.0


@pytest.mark.parametrize(
    ("event_standard", "preserve_known"),
    [
        ("P0217", True),
        ("P0218", True),
        ("P0219", True),
        ("P0207", False),
        ("P0221", False),
    ],
)
def test_industry_unknown_barrier_only_uses_supported_taxonomies(
    event_standard, preserve_known
):
    panel = pd.DataFrame(
        {
            "asset_id": "synthetic",
            "timestamp": pd.to_datetime(["2021-01-02", "2021-01-04"]).as_unit("ns"),
        }
    )
    panel["asset_id"] = panel["asset_id"].astype("object")
    industry_columns = (
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    )
    for column in industry_columns:
        panel[column] = pd.Series(pd.NA, index=panel.index, dtype="string")
    panel["is_nonfinancial_industry"] = pd.Series(
        pd.NA, index=panel.index, dtype="boolean"
    )
    history = pd.DataFrame(
        {
            "source_asset_id": ["synthetic"] * 2,
            "timestamp": pd.to_datetime(["2021-01-01", "2021-01-03"]).as_unit("ns"),
            "industry_standard_id": ["P0207", event_standard],
            "industry_level1_code": ["A01", "B01"],
            "industry_level1_name": ["old leaf", "new leaf"],
        }
    )
    standards = list(dict.fromkeys(["P0207", event_standard]))
    versions = pd.DataFrame(
        {
            "industry_standard_id": standards,
            "implement_date": pd.Timestamp("2020-01-01"),
            "abandon_date": pd.NaT,
        }
    )
    changes = pd.DataFrame(
        {
            "source_asset_id": ["synthetic"],
            "changed_item": ["P0207行业代码"],
            "effective_date": pd.to_datetime(["2021-01-01"]).as_unit("ns"),
            "announcement_date": pd.to_datetime(["2020-12-31"]).as_unit("ns"),
            "value_after": ["old leaf"],
        }
    )
    dictionary = pd.DataFrame(
        {
            "industry_standard_id": ["P0207", "P0207"],
            "industry_code": ["A", "A01"],
            "industry_name": ["old root", "old leaf"],
            "parent_industry_code": ["", "A"],
            "industry_level": ["1", "2"],
        }
    )
    before = [frame.copy(deep=True) for frame in (history, versions, changes)]
    result = _asof_industry(
        panel,
        **_prepare_industry_inputs(
            history,
            industry_sets=dictionary,
            industry_versions=versions,
            industry_changes=changes,
        ),
    )
    assert result.industry_standard_id.iloc[0] == "CSRC2012"
    assert result.industry_level1_code.iloc[0] == "A"
    if preserve_known:
        assert result.industry_standard_id.eq("CSRC2012").all()
        assert result.industry_level1_code.eq("A").all()
    else:
        assert (
            result.loc[1, [*industry_columns, "is_nonfinancial_industry"]].isna().all()
        )
    pd.testing.assert_frame_equal(
        result[["asset_id", "timestamp"]], panel[["asset_id", "timestamp"]]
    )
    for actual, expected in zip((history, versions, changes), before, strict=True):
        pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("newer_available", ["2023-01-09", "2023-01-10", "2023-01-11"])
def test_industry_frontier_discards_superseded_cross_standard_conflicts(
    newer_available,
):
    panel = pd.DataFrame(
        {
            "asset_id": ["a"],
            "timestamp": pd.to_datetime(
                ["2023-01-12" if newer_available == "2023-01-11" else "2023-01-10"]
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
            "asset_id": ["a"] * 3,
            "timestamp": pd.to_datetime(["2023-01-10", "2023-01-10", newer_available]),
            "_industry_event_timestamp": pd.to_datetime(
                ["2023-01-01", "2023-01-01", "2023-01-05"]
            ),
            "industry_standard_id": ["CSRC2012", "CAPCO", "CAPCO"],
            "industry_level1_code": ["A", "B", "C"],
            "industry_level1_name": ["old A", "old B", "new C"],
            "_industry_classification_code": ["A", "B", "C"],
            "_industry_known": [True] * 3,
            "is_nonfinancial_industry": [True] * 3,
        }
    )
    history["asset_id"] = history["asset_id"].astype(object)
    versions = pd.DataFrame(
        {
            "industry_standard_id": ["CSRC2012", "CAPCO"],
            "implement_date": pd.to_datetime(["2012-10-26", "2023-01-01"]),
            "abandon_date": pd.Series([pd.NaT, pd.NaT], dtype="datetime64[ns]"),
        }
    )
    industry_columns = [
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
        "is_nonfinancial_industry",
    ]

    def compute(consumed, facts):
        mixed_panel = pd.concat(
            [consumed.assign(close=10.0), consumed.assign(asset_id="b", close=20.0)],
            ignore_index=True,
        )
        neighbor = history.iloc[[0]].assign(
            asset_id="b", timestamp=pd.Timestamp("2023-01-01")
        )
        mixed_history = pd.concat([facts, neighbor], ignore_index=True)
        result = _asof_industry(
            mixed_panel, mixed_history, industry_sets=None, industry_versions=versions
        )
        assert result.query("asset_id == 'b'").industry_level1_code.tolist() == [
            "A"
        ] * len(consumed)
        pd.testing.assert_series_equal(result.close, mixed_panel.close)
        return result.query("asset_id == 'a'").reset_index(drop=True)

    actual = compute(panel, history)
    assert actual["industry_level1_code"].tolist() == ["C"]
    if newer_available == "2023-01-11":
        # Actual computation history cannot borrow the next day's resolution.
        with_history = pd.concat(
            [panel.assign(timestamp=pd.Timestamp("2023-01-10")), panel],
            ignore_index=True,
        )
        for consumed in (with_history, with_history.iloc[:1]):
            historical = compute(consumed, history)
            assert historical.iloc[[0]][industry_columns].isna().all().all()
            if len(consumed) == 2:
                assert historical.iloc[1].industry_level1_code == "C"
        future = history.copy()
        future.loc[2, "timestamp"] = pd.Timestamp("2023-01-13")
        assert compute(panel, future)[industry_columns].isna().all().all()
    # A current, unresolved cross-standard frontier remains contradictory.
    assert compute(panel, history.iloc[:2])[industry_columns].isna().all().all()
    # A newer frontier restores current knowledge without repairing past conflicts.
    contradictory = pd.concat(
        [history, history.iloc[[0]].assign(_industry_classification_code="D")],
        ignore_index=True,
    )
    assert compute(panel, contradictory).industry_level1_code.tolist() == ["C"]
    historical = compute(
        panel.assign(timestamp=pd.Timestamp("2023-01-10")), contradictory
    )
    assert historical.industry_level1_code.fillna("unknown").tolist() == [
        "unknown" if newer_available == "2023-01-11" else "C"
    ]
    # An unknown latest effective state still blocks older proven classifications.
    history.loc[2, "_industry_known"] = False
    unknown = compute(panel, history)
    assert unknown[industry_columns].isna().all().all()


def test_native_forward_returns_declare_units_and_actual_maturity_columns():
    from quant_research_platform.contracts.columns import ColumnRole
    from quant_research_platform.markets.implementations.a_share.ids import (
        FORWARD_RETURNS_DATA_ID,
    )
    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog,
    )
    from quant_research_platform.research.protocols.a_share import (
        research_dataset_specs,
    )

    spec = catalog.processed_catalog().get(FORWARD_RETURNS_DATA_ID)
    for definition in label_definitions():
        assert spec.column_roles[definition.output_column] is ColumnRole.LABEL
        assert spec.column_roles[definition.available_at_column] is ColumnRole.CONTEXT
        assert spec.column_units[definition.output_column] == "decimal_return"
        assert (
            spec.label_availability_columns[definition.output_column]
            == definition.available_at_column
        )
    for dataset in research_dataset_specs():
        assert len(dataset.label_columns) == len(label_definitions()) + 1
        for label, clock in dataset.label_availability_columns.items():
            assert label in dataset.label_columns
            assert clock in dataset.context_columns
            assert dataset.column_roles[clock] is ColumnRole.CONTEXT
