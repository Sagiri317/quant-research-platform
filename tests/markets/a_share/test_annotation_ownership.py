from __future__ import annotations

import inspect

import pandas as pd

from quant_research_platform.markets.implementations.a_share.processed import (
    annotations,
    panels,
)


def test_daily_panel_schema_excludes_derived_annotation_masks() -> None:
    panel = panels.compute_daily_panel(
        _daily_market_frame(),
        stock_master=_stock_master_frame(),
        market_cap=_market_cap_frame(),
        trading_calendar=_trading_calendar_frame(),
        min_listing_trading_days=1,
    )

    assert not {
        "eligible_base",
        "eligible_after_listing_age",
        "eligible_liquidity",
        "eligible",
        "next_day_is_trading",
        "next_day_limit_status",
        "can_enter",
        "can_exit",
    }.intersection(panel.columns)


def test_annotations_own_eligibility_and_execution_masks() -> None:
    outputs = annotations.derive_annotations(
        _annotation_panel_frame(),
        min_listing_trading_days=120,
    )

    default = outputs["a_share_default"]
    execution = outputs["a_share_execution"]
    expected_default = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share"],
            "asset_id": ["asset_a", "asset_a", "asset_a"],
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-05"]
            ),
            "eligible_base": [True, True, False],
            "eligible_after_listing_age": [True, True, True],
            "eligible_liquidity": [True, True, True],
            "eligible": [True, True, False],
        }
    )
    expected_execution = pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share"],
            "asset_id": ["asset_a", "asset_a", "asset_a"],
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-05"]
            ),
            "next_day_is_trading": [True, True, False],
            "next_day_limit_status": [-1.0, 0.0, float("nan")],
            "can_enter": [True, True, False],
            "can_exit": [False, True, False],
        }
    )

    pd.testing.assert_frame_equal(default, expected_default)
    pd.testing.assert_frame_equal(execution, expected_execution)


def test_daily_context_and_execution_projection_are_separate() -> None:
    outputs = annotations.derive_annotations(_annotation_panel_frame())

    assert outputs["daily_context"].columns.tolist() == [
        "market_id",
        "asset_id",
        "timestamp",
        "market_type",
        "is_st",
        "is_pt",
        "limit_status",
        "listing_age_trading_days",
        "csrc_2012_industry_code",
        "csrc_2012_industry_name",
        "total_market_cap",
        "float_market_cap",
        "amount",
        "turnover",
    ]
    assert outputs["a_share_execution"].columns.tolist() == [
        "market_id",
        "asset_id",
        "timestamp",
        "next_day_is_trading",
        "next_day_limit_status",
        "can_enter",
        "can_exit",
    ]


def test_annotation_outputs_are_joinable_by_primary_key() -> None:
    outputs = annotations.derive_annotations(_annotation_panel_frame())
    base_keys = outputs["a_share_default"].loc[
        :,
        ["market_id", "asset_id", "timestamp"],
    ]

    for data_id, frame in outputs.items():
        keys = frame.loc[:, ["market_id", "asset_id", "timestamp"]]
        assert not keys.duplicated().any(), data_id
        pd.testing.assert_frame_equal(keys, base_keys)


def test_label_calculation_source_does_not_recompute_a_share_trading_rules() -> None:
    from quant_research_platform.markets.implementations.a_share.labels import (
        calculations,
    )

    source = inspect.getsource(calculations)
    forbidden = (
        "is_st",
        "is_pt",
        "is_trading",
        "limit_status",
        "eligible",
        "can_enter",
        "can_exit",
    )
    assert [token for token in forbidden if token in source] == []


def _annotation_panel_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share", "a_share", "a_share"],
            "asset_id": ["asset_a", "asset_a", "asset_a"],
            "timestamp": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-05"]
            ),
            "market_type": [1, 1, 1],
            "is_st": [False, False, True],
            "is_pt": [False, False, False],
            "limit_status": [1.0, -1.0, 0.0],
            "open": [10.0, 9.0, 11.0],
            "high": [10.0, 9.0, 12.0],
            "low": [10.0, 9.0, 10.0],
            "close": [10.0, 9.0, 11.5],
            "is_trading": [True, True, True],
            "is_a_share_scope": [True, True, True],
            "listing_age_trading_days": [130, 131, 132],
            "csrc_2012_industry_code": ["I1", "I1", "I1"],
            "csrc_2012_industry_name": ["Industry", "Industry", "Industry"],
            "total_market_cap": [100.0, 101.0, 102.0],
            "float_market_cap": [90.0, 91.0, 92.0],
            "amount": [1000.0, 1000.0, 1000.0],
            "volume": [100.0, 100.0, 100.0],
            "turnover": [0.1, 0.1, 0.1],
        }
    )


def _daily_market_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["asset_a", "asset_a"],
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "market_type": [1, 1],
            "trading_status_code": [1, 1],
            "filling_flag": ["0", "0"],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "adj_open": [10.0, 11.0],
            "adj_close": [10.2, 11.2],
            "prev_close": [9.9, 10.2],
            "volume": [100.0, 100.0],
            "amount": [1000.0, 1000.0],
            "return_1d": [0.01, 0.02],
            "limit_status": [0, 0],
        }
    )


def _stock_master_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2023-01-01"]),
            "stock_name": ["Asset A"],
            "listing_date": pd.to_datetime(["2023-01-01"]),
            "delisting_date": [pd.NaT],
            "company_status": ["normal"],
            "status_change_date": pd.to_datetime(["2023-01-01"]),
            "csrc_2012_industry_code": ["I1"],
            "csrc_2012_industry_name": ["Industry"],
        }
    )


def _market_cap_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2023-01-01"]),
            "share_change_date": pd.to_datetime(["2023-01-01"]),
            "total_shares": [1000.0],
            "a_float_shares": [500.0],
        }
    )


def _trading_calendar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", "2024-01-03", freq="D"),
            "calendar_date": pd.date_range("2023-01-01", "2024-01-03", freq="D"),
            "source_market_type": [1] * 368,
            "is_trading_day": [True] * 368,
            "source_day_of_week": [1] * 368,
            "source_state": ["O"] * 368,
        }
    )
