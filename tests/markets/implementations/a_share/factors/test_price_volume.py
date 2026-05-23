from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.markets.implementations.a_share.factors.price_volume import (  # noqa: E501
    DEFAULT_PRICE_VOLUME_WINDOWS,
    FIRST_PRICE_VOLUME_FACTOR_FAMILIES,
    add_forward_return_labels,
    calculate_price_volume_factors,
    expected_price_volume_factor_columns,
)


def make_mock_bar_panel() -> pd.DataFrame:
    rows = []
    for stock_code, base in [("000001.SZ", 10.0), ("600000.SH", 20.0)]:
        for offset in range(6):
            price = base + offset
            rows.append(
                {
                    "stock_code": stock_code,
                    "trade_date": pd.Timestamp("2026-01-01")
                    + pd.offsets.BDay(offset),
                    "adj_open": price + 0.1,
                    "adj_close": price + 0.5,
                    "return_1d": 0.01 * offset,
                    "open": price,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price + 0.5,
                    "prev_close": price - 0.5,
                    "amount": 1000.0 + offset * 10.0,
                    "turnover": 0.01 + offset * 0.001,
                }
            )
    return pd.DataFrame(rows)


def test_price_volume_factors_use_past_window_by_asset() -> None:
    frame = make_mock_bar_panel()

    result = calculate_price_volume_factors(frame, windows=(2,))

    first_asset = result[result["stock_code"] == "000001.SZ"].reset_index(drop=True)
    assert pd.isna(first_asset.loc[0, "mom_2d"])
    assert pd.isna(first_asset.loc[1, "mom_2d"])
    expected_mom = first_asset.loc[2, "adj_close"] / first_asset.loc[0, "adj_close"] - 1
    assert first_asset.loc[2, "mom_2d"] == expected_mom
    assert first_asset.loc[2, "rev_2d"] == -expected_mom
    assert first_asset.loc[2, "vol_2d"] > 0
    assert first_asset.loc[2, "avg_amount_2d"] > 0
    assert first_asset.loc[2, "turnover_2d"] > 0
    assert first_asset.loc[2, "amihud_2d"] >= 0
    assert first_asset.loc[2, "amplitude_2d"] > 0


def test_amihud_factor_treats_zero_amount_as_missing() -> None:
    frame = make_mock_bar_panel()
    frame.loc[frame["stock_code"] == "000001.SZ", "amount"] = 0.0

    result = calculate_price_volume_factors(frame, windows=(2,))

    first_asset = result[result["stock_code"] == "000001.SZ"].reset_index(drop=True)
    assert first_asset["amihud_2d"].isna().all()
    assert np.isfinite(result["amihud_2d"].dropna()).all()


def test_forward_return_labels_use_next_execution_price() -> None:
    frame = make_mock_bar_panel()

    result = add_forward_return_labels(frame, horizons=(1, 2))

    first_asset = result[result["stock_code"] == "000001.SZ"].reset_index(drop=True)
    expected_1d = first_asset.loc[2, "adj_open"] / first_asset.loc[1, "adj_open"] - 1
    expected_2d = first_asset.loc[3, "adj_open"] / first_asset.loc[1, "adj_open"] - 1
    assert first_asset.loc[0, "forward_return_1d"] == expected_1d
    assert first_asset.loc[0, "forward_return_2d"] == expected_2d
    assert pd.isna(first_asset.loc[4, "forward_return_1d"])
    assert pd.isna(first_asset.loc[3, "forward_return_2d"])


def test_first_price_volume_factor_interface_lists_expected_columns() -> None:
    assert DEFAULT_PRICE_VOLUME_WINDOWS == (20, 60, 120)
    assert FIRST_PRICE_VOLUME_FACTOR_FAMILIES == (
        "momentum",
        "reversal",
        "volatility",
        "amount_liquidity",
        "turnover",
        "amihud",
        "amplitude",
    )

    columns = expected_price_volume_factor_columns(windows=(20,))

    assert columns == (
        "mom_20d",
        "rev_20d",
        "vol_20d",
        "avg_amount_20d",
        "turnover_20d",
        "amihud_20d",
        "amplitude_20d",
    )
