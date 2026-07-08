from __future__ import annotations

import math

import pandas as pd
import pytest

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.research.factors.microstructure import (
    MICROSTRUCTURE_FACTOR_COLUMNS,
    calculate_microstructure_factors,
    microstructure_factor_definitions,
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
    assert math.isnan(first["trade_intensity_5m"])
    assert math.isnan(first["signed_volume_5m"])
    assert math.isnan(first["flow_imbalance_5m"])
    second = result.iloc[1]
    assert math.isnan(second["trade_intensity_5m"])
    assert math.isnan(second["signed_volume_5m"])
    assert math.isnan(second["flow_imbalance_5m"])


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


def test_microstructure_rolling_5m_requires_full_contiguous_observed_window() -> None:
    frame = pd.DataFrame(
        {
            "market_id": ["crypto"] * 6,
            "asset_id": ["btc_usdt"] * 6,
            "timestamp": pd.to_datetime(
                [
                    "2026-07-06 00:00",
                    "2026-07-06 00:01",
                    "2026-07-06 00:02",
                    "2026-07-06 00:04",
                    "2026-07-06 00:05",
                    "2026-07-06 00:06",
                ]
            ),
            "best_bid": [100.0] * 6,
            "best_ask": [101.0] * 6,
            "bid_quantity": [1.0] * 6,
            "ask_quantity": [1.0] * 6,
            "bid_depth": [2.0] * 6,
            "ask_depth": [2.0] * 6,
            "mid_price": [100.5] * 6,
            "trade_count": [1] * 6,
            "trade_flow_observed": [True, True, True, False, True, True],
            "signed_volume": [1.0] * 6,
            "volume": [2.0] * 6,
        }
    )

    result = calculate_microstructure_factors(frame)

    assert result["trade_intensity_5m"].isna().all()
    assert result["signed_volume_5m"].isna().all()
    assert result["flow_imbalance_5m"].isna().all()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"rolling_window": 0}, "unsupported_microstructure_rolling_window"),
        ({"rolling_window": 3}, "unsupported_microstructure_rolling_window"),
        ({"min_periods": 4}, "microstructure_rolling_requires_full_window"),
        ({"min_periods": 6}, "microstructure_rolling_requires_full_window"),
        ({"expected_interval": pd.Timedelta(0)}, "invalid_expected_interval"),
    ],
)
def test_microstructure_rolling_contract_fails_closed(kwargs, error) -> None:
    with pytest.raises(ValueError, match=error):
        calculate_microstructure_factors(_microstructure_frame(), **kwargs)


def test_microstructure_requires_explicit_trade_flow_observed_column_when_named(
) -> None:
    frame = _microstructure_frame().drop(columns=["trade_flow_observed"])

    with pytest.raises(ValueError, match="missing_trade_flow_observed_column"):
        calculate_microstructure_factors(
            frame,
            trade_flow_observed_column="trade_flow_observed",
        )


def test_microstructure_rejects_non_bool_trade_flow_observed_column() -> None:
    frame = _microstructure_frame()
    frame["trade_flow_observed"] = ["False"] * len(frame)

    with pytest.raises(ValueError, match="invalid_trade_flow_observed"):
        calculate_microstructure_factors(
            frame,
            trade_flow_observed_column="trade_flow_observed",
        )


def test_microstructure_definitions_keep_point_in_time_factors_unwindowed() -> None:
    definitions = {
        definition.factor_id: definition
        for definition in microstructure_factor_definitions(
            market_id=MarketId("crypto"),
            required_input_data_id="market_panel_1m",
        )
    }

    assert definitions["spread"].required_windows == ()
    assert definitions["spread"].lookback_window is None
    assert definitions["spread"].window is None
    assert definitions["spread"].timing_semantics == "bar_close_asof_quote"
    assert definitions["trade_intensity_5m"].required_windows == (5,)
    assert definitions["trade_intensity_5m"].lookback_window == 5
    assert definitions["trade_intensity_5m"].window == 5
    assert (
        definitions["trade_intensity_5m"].timing_semantics
        == "clock_contiguous_5m_after_bar_close"
    )


def _microstructure_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["crypto"] * 5,
            "asset_id": ["btc_usdt"] * 5,
            "timestamp": pd.date_range("2026-07-06 00:00", periods=5, freq="min"),
            "best_bid": [100.0] * 5,
            "best_ask": [101.0] * 5,
            "bid_quantity": [1.0] * 5,
            "ask_quantity": [1.0] * 5,
            "bid_depth": [2.0] * 5,
            "ask_depth": [2.0] * 5,
            "mid_price": [100.5] * 5,
            "trade_count": [1] * 5,
            "trade_flow_observed": [True] * 5,
            "signed_volume": [1.0] * 5,
            "volume": [2.0] * 5,
        }
    )
