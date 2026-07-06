from __future__ import annotations

import math

import pandas as pd

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


def test_incomplete_quote_fields_do_not_set_quote_observed_true() -> None:
    depth = pd.DataFrame(
        {
            "source_asset_id": ["BTCUSDT", "BTCUSDT"],
            "timestamp": pd.to_datetime(
                ["2026-07-06 00:00:58", "2026-07-06 00:00:58"]
            ),
            "side": ["bid", "ask"],
            "level": [1, 1],
            "price": [100.0, 100.2],
            "quantity": [1.5, pd.NA],
        }
    )

    panel = compute_market_panel(_bars_frame(), trades=pd.DataFrame(), depth=depth)

    assert bool(panel.loc[0, "quote_observed"]) is False
    assert math.isnan(panel.loc[0, "best_ask"])


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
        }
    )
