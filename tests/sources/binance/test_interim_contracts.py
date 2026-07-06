from __future__ import annotations

import pytest

from quant_research_platform.contracts.data import DataFrequency, DataGrain
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.sources.binance.interim import (
    interim_catalog,
    normalize_raw_envelopes,
)


def test_binance_interim_catalog_registers_approved_source_specs() -> None:
    catalog = interim_catalog()

    assert tuple(spec.data_id for spec in catalog.list()) == (
        "aggregate_trades_tick",
        "depth_levels_tick",
        "market_bars_1m",
    )
    market_bars = catalog.get("market_bars_1m")
    trades = catalog.get("aggregate_trades_tick")
    depth = catalog.get("depth_levels_tick")

    assert market_bars.source_id == SourceId("binance")
    assert market_bars.frequency is DataFrequency.MINUTE
    assert market_bars.grain is DataGrain.ASSET_TIMESTAMP
    assert market_bars.primary_key == ("source_asset_id", "timestamp")
    assert trades.frequency is DataFrequency.TICK
    assert trades.grain is DataGrain.TRADE_EVENT
    assert trades.primary_key == (
        "source_asset_id",
        "timestamp",
        "aggregate_trade_id",
    )
    assert depth.frequency is DataFrequency.TICK
    assert depth.grain is DataGrain.ORDER_BOOK_LEVEL
    assert depth.primary_key == ("source_asset_id", "timestamp", "side", "level")


def test_binance_interim_fields_are_source_local_only() -> None:
    catalog = interim_catalog()

    for spec in catalog.list():
        assert "source_asset_id" in spec.required_columns
        assert "market_id" not in spec.required_columns
        assert "asset_id" not in spec.required_columns


def test_binance_interim_rejects_data_id_raw_kind_mismatch() -> None:
    with pytest.raises(ValueError, match="binance_raw_data_id_kind_mismatch"):
        normalize_raw_envelopes(
            (
                {
                    "source": "binance",
                    "market_type": "spot",
                    "data_kind": "depth",
                    "symbol": "BTCUSDT",
                    "captured_at": "2026-07-06T00:00:02Z",
                    "payload": {
                        "lastUpdateId": 1,
                        "bids": [["100.0", "1.0"]],
                        "asks": [],
                    },
                },
            ),
            data_id="market_bars_1m",
        )
