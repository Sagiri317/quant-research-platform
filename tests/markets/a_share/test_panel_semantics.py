from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.markets.implementations.a_share import calendar
from quant_research_platform.markets.implementations.a_share.ids import (
    DEFAULT_UNIVERSE_ID,
    EXECUTION_MASK_ID,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    annotations,
)
from quant_research_platform.markets.implementations.a_share.processed.panels import (
    _asof_join_stock_master,
    _dedupe_exact_key_duplicates_or_raise_conflict,
    _first_static_stock_value,
    compute_market_panel,
)
from tests.helpers.synthetic_a_share import (
    make_synthetic_a_share_market_cap,
    make_synthetic_a_share_stock_master,
    make_synthetic_a_share_trading_calendar,
    make_synthetic_market_bars_daily,
    make_synthetic_market_bars_daily_row,
)


def test_a_share_calendar_helpers_use_source_normalized_calendar() -> None:
    frame = pd.DataFrame(
        {
            "calendar_date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-06",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-09",
                ]
            ),
            "source_market_type": [1, 4, 16, 32, 64, 1, 1],
            "is_trading_day": [True, True, False, True, True, False, False],
        }
    )

    assert calendar.unified_trading_sessions(frame).tolist() == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-06"),
    ]
    assert calendar.market_type_trading_sessions(frame, 1).tolist() == [
        pd.Timestamp("2024-01-02")
    ]
    assert calendar.market_type_trading_sessions(frame, 64).tolist() == [
        pd.Timestamp("2024-01-08")
    ]
    with pytest.raises(ValueError, match="calendar_missing_normalized_columns"):
        calendar.unified_trading_sessions(
            pd.DataFrame({"Markettype": [1], "Clddt": ["2024-01-02"]})
        )


def test_a_share_calendar_warmup_start_uses_open_sessions() -> None:
    frame = pd.DataFrame(
        {
            "calendar_date": pd.to_datetime(
                [
                    "2019-12-26",
                    "2019-12-27",
                    "2019-12-28",
                    "2019-12-29",
                    "2019-12-30",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-02",
                ]
            ),
            "source_market_type": [1, 1, 1, 1, 1, 1, 1, 64],
            "is_trading_day": [True, True, False, True, True, False, True, True],
        }
    )

    assert calendar.warmup_start_date_for_window(
        frame,
        start_date="2020-01-01",
        max_window=3,
    ) == pd.Timestamp("2019-12-27")
    with pytest.raises(ValueError, match="invalid_factor_warmup_window"):
        calendar.warmup_start_date_for_window(
            frame,
            start_date="2020-01-01",
            max_window=-1,
        )


def test_compute_market_panel_maps_source_assets_to_a_share_assets() -> None:
    panel = compute_market_panel(
        make_synthetic_market_bars_daily(["2014-01-02", "2014-01-03"]),
        stock_master=make_synthetic_a_share_stock_master(),
        market_cap=make_synthetic_a_share_market_cap(),
        trading_calendar=make_synthetic_a_share_trading_calendar(
            pd.date_range("2010-01-01", "2014-01-03", freq="D")
        ),
    )

    assert panel["market_id"].unique().tolist() == ["a_share"]
    assert panel["asset_id"].unique().tolist() == ["000001"]
    assert pd.api.types.is_numeric_dtype(panel["filling_flag"])
    assert "source_asset_id" not in panel.columns
    assert "source_id" not in panel.columns
    assert "eligible" not in panel.columns
    masks = annotations.derive_annotations(
        panel,
        min_listing_trading_days=1,
    )
    default = masks[DEFAULT_UNIVERSE_ID]
    execution = masks[EXECUTION_MASK_ID]
    assert default["eligible"].tolist() == [True, True]
    assert execution["can_enter"].tolist() == [True, False]
    assert execution["can_exit"].tolist() == [True, False]


def test_compute_market_panel_filters_non_a_share_market_types() -> None:
    panel = compute_market_panel(
        pd.DataFrame(
            [
                make_synthetic_market_bars_daily_row("2014-01-02", market_type=4),
                make_synthetic_market_bars_daily_row("2014-01-03", market_type=32),
                make_synthetic_market_bars_daily_row("2014-01-06", market_type=64),
                make_synthetic_market_bars_daily_row("2014-01-07", market_type=2),
            ]
        ),
        stock_master=make_synthetic_a_share_stock_master(),
        market_cap=make_synthetic_a_share_market_cap(),
        trading_calendar=make_synthetic_a_share_trading_calendar(
            pd.to_datetime(["2014-01-02", "2014-01-03", "2014-01-06", "2014-01-07"])
        ),
    )

    assert len(panel) == 2
    assert panel["market_type"].tolist() == [4, 32]
    assert panel["is_a_share_scope"].tolist() == [True, True]


def test_compute_market_panel_preserves_listing_age_closed_boundary() -> None:
    panel = compute_market_panel(
        make_synthetic_market_bars_daily(["2020-01-01", "2020-01-02"]),
        stock_master=make_synthetic_a_share_stock_master(listing_date="2020-01-01"),
        market_cap=make_synthetic_a_share_market_cap(share_change_date="2020-01-01"),
        trading_calendar=make_synthetic_a_share_trading_calendar(
            pd.to_datetime(["2020-01-01", "2020-01-02"]),
            is_trading_day=True,
        ),
    )

    assert panel["listing_age_trading_days"].tolist() == [1, 2]
    default = annotations.derive_annotations(
        panel,
        min_listing_trading_days=1,
    )[DEFAULT_UNIVERSE_ID]
    assert default["eligible_after_listing_age"].tolist() == [True, True]


def test_compute_market_panel_zero_float_shares_make_turnover_missing() -> None:
    panel = compute_market_panel(
        make_synthetic_market_bars_daily(["2020-01-02"]),
        stock_master=make_synthetic_a_share_stock_master(),
        market_cap=make_synthetic_a_share_market_cap(a_float_shares=0.0),
        trading_calendar=make_synthetic_a_share_trading_calendar(
            pd.to_datetime(["2020-01-02"])
        ),
    )

    assert pd.isna(panel.loc[0, "turnover"])
    default = annotations.derive_annotations(
        panel,
        min_listing_trading_days=1,
    )[DEFAULT_UNIVERSE_ID]
    assert bool(default.loc[0, "eligible_liquidity"]) is False
    assert bool(default.loc[0, "eligible"]) is False


def test_compute_market_panel_preserves_missing_limit_status_context() -> None:
    panel = compute_market_panel(
        pd.DataFrame(
            [make_synthetic_market_bars_daily_row("2020-01-02", limit_status=pd.NA)]
        ),
        stock_master=make_synthetic_a_share_stock_master(),
        market_cap=make_synthetic_a_share_market_cap(),
        trading_calendar=make_synthetic_a_share_trading_calendar(
            pd.to_datetime(["2020-01-02"])
        ),
    )

    assert pd.isna(panel.loc[0, "limit_status"])
    assert bool(panel.loc[0, "is_limit_up"]) is False
    assert bool(panel.loc[0, "is_limit_down"]) is False


def test_compute_market_panel_rejects_invalid_limit_status_context() -> None:
    with pytest.raises(ValueError, match="invalid_limit_status"):
        compute_market_panel(
            pd.DataFrame(
                [make_synthetic_market_bars_daily_row("2020-01-02", limit_status=2)]
            ),
            stock_master=make_synthetic_a_share_stock_master(),
            market_cap=make_synthetic_a_share_market_cap(),
            trading_calendar=make_synthetic_a_share_trading_calendar(
                pd.to_datetime(["2020-01-02"])
            ),
        )


def test_annotations_mask_only_next_bar_one_word_limits() -> None:
    rows = [
        make_synthetic_market_bars_daily_row("2020-01-01"),
        make_synthetic_market_bars_daily_row(
            "2020-01-02",
            limit_status=1,
            open_price=11.0,
            high=11.0,
            low=11.0,
            close=11.0,
        ),
        make_synthetic_market_bars_daily_row(
            "2020-01-03",
            limit_status=1,
            open_price=10.8,
            high=11.0,
            low=10.7,
            close=10.9,
        ),
        make_synthetic_market_bars_daily_row(
            "2020-01-06",
            limit_status=-1,
            open_price=9.0,
            high=9.0,
            low=9.0,
            close=9.0,
        ),
        make_synthetic_market_bars_daily_row(
            "2020-01-07",
            limit_status=-1,
            open_price=9.2,
            high=9.4,
            low=9.0,
            close=9.1,
        ),
    ]

    panel = compute_market_panel(
        pd.DataFrame(rows),
        stock_master=make_synthetic_a_share_stock_master(),
        market_cap=make_synthetic_a_share_market_cap(),
        trading_calendar=make_synthetic_a_share_trading_calendar(
            pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
            )
        ),
    )
    execution = annotations.derive_annotations(
        panel,
        min_listing_trading_days=1,
    )[EXECUTION_MASK_ID].set_index("timestamp")

    assert bool(execution.loc[pd.Timestamp("2020-01-01"), "can_enter"]) is False
    assert bool(execution.loc[pd.Timestamp("2020-01-02"), "can_enter"]) is True
    assert bool(execution.loc[pd.Timestamp("2020-01-03"), "can_exit"]) is False
    assert bool(execution.loc[pd.Timestamp("2020-01-06"), "can_exit"]) is True
    assert bool(execution.loc[pd.Timestamp("2020-01-07"), "can_enter"]) is False
    assert bool(execution.loc[pd.Timestamp("2020-01-07"), "can_exit"]) is False


def test_compute_market_panel_fails_closed_on_conflicting_reference_duplicates() -> (
    None
):
    duplicate_stocks = pd.concat(
        [
            make_synthetic_a_share_stock_master(),
            make_synthetic_a_share_stock_master().assign(company_status="watched"),
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="conflicting_panel_join_duplicate_keys: stock_master",
    ):
        compute_market_panel(
            make_synthetic_market_bars_daily(["2020-01-02"]),
            stock_master=duplicate_stocks,
            market_cap=make_synthetic_a_share_market_cap(),
            trading_calendar=make_synthetic_a_share_trading_calendar(
                pd.to_datetime(["2020-01-02"])
            ),
        )


def test_compute_market_panel_fails_closed_on_missing_source_keys() -> None:
    with pytest.raises(ValueError, match="missing_source_asset_id"):
        compute_market_panel(
            make_synthetic_market_bars_daily(["2020-01-02"]).drop(
                columns=["source_asset_id"]
            ),
            stock_master=make_synthetic_a_share_stock_master(),
            market_cap=make_synthetic_a_share_market_cap(),
            trading_calendar=make_synthetic_a_share_trading_calendar(
                pd.to_datetime(["2020-01-02"])
            ),
        )

    with pytest.raises(ValueError, match="missing_source_timestamp"):
        compute_market_panel(
            make_synthetic_market_bars_daily(["2020-01-02"]).drop(
                columns=["timestamp"]
            ),
            stock_master=make_synthetic_a_share_stock_master(),
            market_cap=make_synthetic_a_share_market_cap(),
            trading_calendar=make_synthetic_a_share_trading_calendar(
                pd.to_datetime(["2020-01-02"])
            ),
        )


def test_compute_market_panel_fails_closed_on_missing_reference_dates() -> None:
    with pytest.raises(ValueError, match="missing_market_cap_panel_timestamp"):
        compute_market_panel(
            make_synthetic_market_bars_daily(["2020-01-02"]),
            stock_master=make_synthetic_a_share_stock_master(),
            market_cap=make_synthetic_a_share_market_cap().assign(timestamp=pd.NaT),
            trading_calendar=make_synthetic_a_share_trading_calendar(
                pd.to_datetime(["2020-01-02"])
            ),
        )

    with pytest.raises(
        ValueError,
        match="missing_stock_master_panel_effective_timestamp",
    ):
        compute_market_panel(
            make_synthetic_market_bars_daily(["2020-01-02"]),
            stock_master=make_synthetic_a_share_stock_master().assign(
                listing_date=pd.NaT,
                status_change_date=pd.NaT,
            ),
            market_cap=make_synthetic_a_share_market_cap(),
            trading_calendar=make_synthetic_a_share_trading_calendar(
                pd.to_datetime(["2020-01-02"])
            ),
        )


def test_compute_market_panel_uses_missing_reference_defaults() -> None:
    panel = compute_market_panel(
        make_synthetic_market_bars_daily(["2020-01-02"]),
        stock_master=make_synthetic_a_share_stock_master(),
        market_cap=make_synthetic_a_share_market_cap().iloc[0:0],
        trading_calendar=make_synthetic_a_share_trading_calendar(
            pd.to_datetime(["2020-01-02"])
        ),
    )

    assert pd.isna(panel.loc[0, "share_change_date"])
    assert pd.isna(panel.loc[0, "total_shares"])
    assert pd.isna(panel.loc[0, "a_float_shares"])
    assert pd.isna(panel.loc[0, "turnover"])


def test_compute_market_panel_fails_closed_on_missing_join_key_columns() -> None:
    with pytest.raises(
        ValueError,
        match="missing_panel_join_key_columns: market_cap: timestamp",
    ):
        _dedupe_exact_key_duplicates_or_raise_conflict(
            pd.DataFrame({"asset_id": ["000001"]}),
            key_columns=("asset_id", "timestamp"),
            dataset="market_cap",
        )


def test_compute_market_panel_fails_closed_on_missing_canonical_after_mapping(
    monkeypatch,
) -> None:
    from quant_research_platform.markets.implementations.a_share.processed import panels

    monkeypatch.setattr(
        panels,
        "MARKET_PANEL_KEY_COLUMNS",
        (*panels.MARKET_PANEL_KEY_COLUMNS, "missing_canonical"),
    )

    with pytest.raises(
        ValueError,
        match="market_bars_interim_missing_canonical_columns: missing_canonical",
    ):
        compute_market_panel(
            make_synthetic_market_bars_daily(["2020-01-02"]),
            stock_master=make_synthetic_a_share_stock_master(),
            market_cap=make_synthetic_a_share_market_cap(),
            trading_calendar=make_synthetic_a_share_trading_calendar(
                pd.to_datetime(["2020-01-02"])
            ),
        )


def test_panel_asof_stock_master_fills_missing_stock_columns() -> None:
    daily = pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset_without_master"],
            "timestamp": pd.to_datetime(["2020-01-02"]),
        }
    )
    stocks = pd.DataFrame(
        {
            "asset_id": pd.Series([], dtype="object"),
            "listing_date": pd.Series([], dtype="datetime64[ns]"),
            "status_change_date": pd.Series([], dtype="datetime64[ns]"),
            "stock_name": pd.Series([], dtype="object"),
        }
    )

    joined = _asof_join_stock_master(daily, stocks)

    assert pd.isna(joined.loc[0, "stock_name"])
    assert _first_static_stock_value(stocks, "missing_column") == {}
