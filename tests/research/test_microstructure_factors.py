from __future__ import annotations

import math

import pandas as pd

from quant_research_platform.research.factors.microstructure import (
    MICROSTRUCTURE_FACTOR_COLUMNS,
    calculate_microstructure_factors,
)


def test_microstructure_factor_calculation_outputs_approved_columns() -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["crypto", "crypto"],
            "asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:01", "2026-07-06 00:00"]),
            "best_bid": [101.0, 100.0],
            "best_ask": [103.0, 102.0],
            "bid_quantity": [3.0, 2.0],
            "ask_quantity": [1.0, 2.0],
            "bid_depth": [8.0, 7.0],
            "ask_depth": [4.0, 5.0],
            "mid_price": [102.0, 101.0],
            "trade_count": [4, 3],
            "signed_volume": [2.0, -1.0],
            "volume": [5.0, 10.0],
        }
    )

    result = calculate_microstructure_factors(frame)

    approved_columns = tuple(
        column for column in result.columns if column in MICROSTRUCTURE_FACTOR_COLUMNS
    )
    assert approved_columns == (
        "spread",
        "relative_spread",
        "depth_imbalance",
        "microprice",
        "microprice_deviation",
        "trade_intensity_5m",
        "signed_volume_5m",
        "flow_imbalance_5m",
    )
    assert result["timestamp"].tolist() == sorted(result["timestamp"].tolist())
    first = result.iloc[0]
    assert first["spread"] == 2.0
    assert first["relative_spread"] == 2.0 / 101.0
    assert first["depth_imbalance"] == (7.0 - 5.0) / (7.0 + 5.0)
    assert first["microprice"] == (102.0 * 2.0 + 100.0 * 2.0) / 4.0
    assert first["microprice_deviation"] == first["microprice"] / 101.0 - 1.0
    assert first["trade_intensity_5m"] == 3
    assert first["signed_volume_5m"] == -1.0
    assert first["flow_imbalance_5m"] == -1.0 / 10.0
    second = result.iloc[1]
    assert second["trade_intensity_5m"] == 7
    assert second["signed_volume_5m"] == 1.0
    assert second["flow_imbalance_5m"] == 1.0 / 15.0


def test_microstructure_factor_calculation_handles_missing_and_zero_denominators(
) -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["crypto", "crypto"],
            "asset_id": ["ETHUSDT", "ETHUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00", "2026-07-06 00:00"]),
            "best_bid": [pd.NA, 100.0],
            "best_ask": [101.0, 99.0],
            "bid_quantity": [0.0, 1.0],
            "ask_quantity": [0.0, 1.0],
            "bid_depth": [0.0, 1.0],
            "ask_depth": [0.0, 1.0],
            "mid_price": [pd.NA, 99.5],
            "trade_count": [0, 0],
            "signed_volume": [0.0, 0.0],
            "volume": [0.0, 0.0],
        }
    )

    result = calculate_microstructure_factors(frame)

    assert math.isnan(result.loc[0, "spread"])
    assert math.isnan(result.loc[0, "relative_spread"])
    assert math.isnan(result.loc[0, "depth_imbalance"])
    assert math.isnan(result.loc[0, "microprice"])
    assert math.isnan(result.loc[0, "microprice_deviation"])
    assert math.isnan(result.loc[0, "flow_imbalance_5m"])
    assert result.loc[1, "spread"] == -1.0
    assert result.loc[1, "depth_imbalance"] == 0.0
    assert math.isnan(result.loc[1, "flow_imbalance_5m"])
