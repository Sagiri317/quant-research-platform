from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.markets.implementations.crypto.processed.panels import (
    compute_market_panel,
)


def test_compute_market_panel_canonicalizes_source_asset_and_asof_quote() -> None:
    bars = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00", "2026-07-06 00:01"]),
            "close_time": pd.to_datetime(
                ["2026-07-06 00:00:59", "2026-07-06 00:01:59"]
            ),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [10.0, 11.0],
            "quote_volume": [1005.0, 1116.5],
            "trade_count": [1, 1],
            "taker_buy_volume": [4.0, 5.0],
            "taker_buy_quote_volume": [402.0, 507.5],
        }
    )
    trades = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:01"]),
            "signed_quantity": [2.0],
        }
    )
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(
                [
                    "2026-07-06 00:00:58",
                    "2026-07-06 00:00:58",
                    "2026-07-06 00:02:01",
                    "2026-07-06 00:02:01",
                ]
            ),
            "side": ["bid", "ask", "bid", "ask"],
            "level": [1, 1, 1, 1],
            "price": [100.0, 100.2, 101.0, 101.2],
            "quantity": [1.5, 1.0, 2.0, 2.0],
        }
    )

    panel = compute_market_panel(
        bars,
        trades=trades,
        depth=depth,
        max_quote_age=pd.Timedelta(seconds=30),
    )

    assert panel["asset_id"].tolist() == ["btc_usdt", "btc_usdt"]
    assert "source_asset_id" not in panel.columns
    assert bool(panel.loc[0, "quote_observed"]) is True
    assert panel.loc[0, "quote_age_seconds"] == 1.0
    assert panel.loc[0, "best_bid"] == 100.0
    assert bool(panel.loc[1, "quote_observed"]) is False
    assert math.isnan(panel.loc[1, "best_bid"])


def test_compute_market_panel_distinguishes_missing_trade_feed_from_zero_flow() -> None:
    bars = pd.DataFrame(
        {
            "source_asset_id": ["ETHUSDT", "ETHUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00", "2026-07-06 00:01"]),
            "close_time": pd.to_datetime(
                ["2026-07-06 00:00:59", "2026-07-06 00:01:59"]
            ),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [0.0, 10.0],
            "quote_volume": [0.0, 1015.0],
            "trade_count": [0, 2],
            "taker_buy_volume": [0.0, 5.0],
            "taker_buy_quote_volume": [0.0, 507.5],
        }
    )

    panel = compute_market_panel(
        bars,
        trades=pd.DataFrame(
            columns=["source_asset_id", "timestamp", "signed_quantity"],
        ),
        depth=pd.DataFrame(),
    )

    assert bool(panel.loc[0, "trade_flow_observed"]) is True
    assert panel.loc[0, "signed_volume"] == 0.0
    assert bool(panel.loc[1, "trade_flow_observed"]) is False
    assert math.isnan(panel.loc[1, "signed_volume"])


def test_one_sided_depth_maps_to_quote_observed_false() -> None:
    panel = compute_market_panel(
        _bars_frame(),
        trades=pd.DataFrame(),
        depth=pd.DataFrame(
            {
                "source_asset_id": ["BTCUSDT"],
                "timestamp": pd.to_datetime(["2026-07-06 00:00:58"]),
                "side": ["bid"],
                "level": [1],
                "price": [100.0],
                "quantity": [1.5],
            }
        ),
    )

    assert bool(panel.loc[0, "quote_observed"]) is False
    assert math.isnan(panel.loc[0, "best_bid"])


def test_one_sided_quote_fields_do_not_set_quote_observed_true() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:58"]),
            "side": ["bid"],
            "level": [1],
            "price": [100.0],
            "quantity": [1.5],
        }
    )

    panel = compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)

    assert bool(panel.loc[0, "quote_observed"]) is False
    assert math.isnan(panel.loc[0, "best_bid"])


def test_crossed_or_invalid_quote_sets_quote_observed_false() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 00:00:58", "2026-07-06 00:00:58"]
            ),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.2, 100.0],
            "quantity": [1.5, 1.0],
        }
    )

    panel = compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)

    assert bool(panel.loc[0, "quote_observed"]) is False
    assert math.isnan(panel.loc[0, "mid_price"])


def test_quote_after_close_not_used() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 00:01:00", "2026-07-06 00:01:00"]
            ),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, 1.0],
        }
    )

    panel = compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)

    assert bool(panel.loc[0, "quote_observed"]) is False
    assert math.isnan(panel.loc[0, "best_bid"])


def test_stale_quote_sets_quote_observed_false() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 00:00:10", "2026-07-06 00:00:10"]
            ),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, 1.0],
        }
    )

    panel = compute_market_panel(
        _bars_frame(),
        trades=pd.DataFrame(),
        depth=depth,
        max_quote_age=pd.Timedelta(seconds=10),
    )

    assert bool(panel.loc[0, "quote_observed"]) is False
    assert math.isnan(panel.loc[0, "quote_age_seconds"])


def test_invalid_ohlc_sets_eligible_false() -> None:
    bars = _bars_frame()
    bars.loc[0, "high"] = 99.0

    panel = compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())

    assert bool(panel.loc[0, "eligible"]) is False


def test_invalid_bar_quantity_sets_eligible_false_and_masks_taker_sell_volume() -> None:
    bars = _bars_frame()
    bars.loc[0, "taker_buy_volume"] = 11.0

    panel = compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())

    assert bool(panel.loc[0, "eligible"]) is False
    assert math.isnan(panel.loc[0, "taker_sell_volume"])


def test_invalid_signed_trade_quantity_fails_closed() -> None:
    trades = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:01"]),
            "signed_quantity": ["bad"],
        }
    )

    with pytest.raises(ValueError, match="invalid_crypto_signed_quantity"):
        compute_market_panel(_bars_frame(), trades=trades, depth=pd.DataFrame())


def test_zero_quantity_quote_is_not_observed() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 00:00:58", "2026-07-06 00:00:58"]
            ),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [0.0, 1.0],
        }
    )

    panel = compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)

    assert bool(panel.loc[0, "quote_observed"]) is False
    assert math.isnan(panel.loc[0, "best_bid"])


def test_invalid_depth_inputs_fail_closed() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 00:00:58", "2026-07-06 00:00:58"]
            ),
            "side": ["bid", "bid"],
            "level": [1, 1],
            "price": [100.0, 99.9],
            "quantity": [1.5, 1.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate_crypto_depth_level"):
        compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)


def test_invalid_bar_timestamp_fails_closed() -> None:
    bars = pd.DataFrame(
        {
            **_bars_frame().to_dict(orient="list"),
            "timestamp": ["bad"],
        }
    )

    with pytest.raises(ValueError, match="invalid_crypto_panel_timestamp"):
        compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("open", "bad"),
        ("high", True),
        ("open", float("inf")),
        ("low", float("-inf")),
        ("close", np.nan),
    ],
)
def test_invalid_bar_ohlc_fails_closed(column: str, value: object) -> None:
    bars = _bars_frame()
    bars[column] = bars[column].astype(object)
    bars.loc[0, column] = value

    with pytest.raises(ValueError, match="invalid_crypto_ohlc"):
        compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("volume", "bad"),
        ("taker_buy_volume", True),
        ("volume", float("inf")),
        ("quote_volume", float("-inf")),
        ("taker_buy_quote_volume", np.nan),
    ],
)
def test_invalid_bar_quantity_inputs_fail_closed(
    column: str,
    value: object,
) -> None:
    bars = _bars_frame()
    bars[column] = bars[column].astype(object)
    bars.loc[0, column] = value

    with pytest.raises(ValueError, match="invalid_crypto_bar_quantity"):
        compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())


def test_bool_trade_count_fails_closed() -> None:
    bars = _bars_frame()
    bars["trade_count"] = bars["trade_count"].astype(object)
    bars.loc[0, "trade_count"] = True

    with pytest.raises(ValueError, match="invalid_crypto_trade_count"):
        compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())


def test_infinite_trade_count_fails_closed() -> None:
    bars = _bars_frame()
    bars["trade_count"] = bars["trade_count"].astype(object)
    bars.loc[0, "trade_count"] = float("inf")

    with pytest.raises(ValueError, match="invalid_crypto_trade_count"):
        compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())


@pytest.mark.parametrize("trade_count", [1.5, -1])
def test_non_integer_or_negative_trade_count_sets_eligible_false(
    trade_count: object,
) -> None:
    bars = _bars_frame()
    bars["trade_count"] = bars["trade_count"].astype(object)
    bars.loc[0, "trade_count"] = trade_count

    panel = compute_market_panel(bars, trades=pd.DataFrame(), depth=pd.DataFrame())

    assert bool(panel.loc[0, "eligible"]) is False


@pytest.mark.parametrize(
    ("column", "value", "error"),
    [
        ("price", "bad", "invalid_crypto_depth_price"),
        ("price", pd.NA, "invalid_crypto_depth_price"),
        ("price", np.nan, "invalid_crypto_depth_price"),
        ("price", float("inf"), "invalid_crypto_depth_price"),
        ("price", True, "invalid_crypto_depth_price"),
        ("quantity", "bad", "invalid_crypto_depth_quantity"),
        ("quantity", pd.NA, "invalid_crypto_depth_quantity"),
        ("quantity", np.nan, "invalid_crypto_depth_quantity"),
        ("quantity", float("inf"), "invalid_crypto_depth_quantity"),
        ("quantity", True, "invalid_crypto_depth_quantity"),
    ],
)
def test_invalid_depth_price_quantity_fail_closed(
    column: str,
    value: object,
    error: str,
) -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 00:00:58", "2026-07-06 00:00:58"]
            ),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, 1.0],
        }
    )
    depth[column] = depth[column].astype(object)
    depth.loc[0, column] = value

    with pytest.raises(ValueError, match=error):
        compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)


@pytest.mark.parametrize("signed_quantity", [True, np.nan, "bad", float("inf")])
def test_invalid_signed_trade_quantity_edge_cases_fail_closed(
    signed_quantity: object,
) -> None:
    trades = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:01"]),
            "signed_quantity": [signed_quantity],
        }
    )

    with pytest.raises(ValueError, match="invalid_crypto_signed_quantity"):
        compute_market_panel(_bars_frame(), trades=trades, depth=pd.DataFrame())


@pytest.mark.parametrize("side", [["bid"], {}, None, "buy"])
def test_invalid_depth_side_values_fail_closed(side: object) -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 00:00:58", "2026-07-06 00:00:58"]
            ),
            "side": [side, "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, 1.0],
        }
    )

    with pytest.raises(ValueError, match="invalid_crypto_depth_side"):
        compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)


def test_valid_depth_side_values_still_pass() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 00:00:58", "2026-07-06 00:00:58"]
            ),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, 1.0],
        }
    )

    panel = compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)

    assert bool(panel.loc[0, "quote_observed"]) is True


def test_invalid_trade_source_asset_fails_closed() -> None:
    trades = pd.DataFrame(
        {
            "source_asset_id": ["unknown"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00:01"]),
            "signed_quantity": [1.0],
        }
    )

    with pytest.raises(ValueError, match="invalid_crypto_source_asset_id"):
        compute_market_panel(_bars_frame(), trades=trades, depth=pd.DataFrame())


def test_quote_observed_and_trade_flow_observed_do_not_control_eligible() -> None:
    panel = compute_market_panel(
        _bars_frame(),
        trades=pd.DataFrame(),
        depth=pd.DataFrame(),
    )

    assert bool(panel.loc[0, "eligible"]) is True
    assert bool(panel.loc[0, "quote_observed"]) is False
    assert bool(panel.loc[0, "trade_flow_observed"]) is False


def _bars_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT"],
            "timestamp": pd.to_datetime(["2026-07-06 00:00"]),
            "close_time": pd.to_datetime(["2026-07-06 00:00:59"]),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10.0],
            "quote_volume": [1005.0],
            "trade_count": [1],
            "taker_buy_volume": [4.0],
            "taker_buy_quote_volume": [402.0],
        }
    )
