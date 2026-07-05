from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_research_platform.sources.binance.interim import materialize_interim_data
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.layout import StorageLayout


def test_binance_interim_materializer_writes_source_datasets(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "binance"
    _write_jsonl(
        raw_root / "spot" / "klines" / "BTCUSDT" / "1m" / "2026-07-06.jsonl",
        {
            "symbol": "BTCUSDT",
            "open_time": 1783296000000,
            "open": "100.0",
            "high": "101.0",
            "low": "99.0",
            "close": "100.5",
            "volume": "10.0",
            "close_time": 1783296059999,
            "quote_volume": "1005.0",
            "trade_count": 7,
            "taker_buy_volume": "4.0",
            "taker_buy_quote_volume": "402.0",
        },
    )
    _write_jsonl(
        raw_root / "spot" / "agg_trades" / "BTCUSDT" / "2026-07-06.jsonl",
        {
            "symbol": "BTCUSDT",
            "aggregate_trade_id": 1,
            "price": "100.1",
            "quantity": "2.0",
            "first_trade_id": 10,
            "last_trade_id": 11,
            "timestamp": 1783296001000,
            "is_buyer_maker": False,
        },
    )
    _write_jsonl(
        raw_root / "spot" / "depth" / "BTCUSDT" / "2026-07-06.jsonl",
        {
            "symbol": "BTCUSDT",
            "timestamp": 1783296002000,
            "bids": [["100.0", "1.5"], ["99.9", "2.0"]],
            "asks": [["100.2", "1.0"]],
        },
    )

    result = materialize_interim_data(layout=StorageLayout(project_root=tmp_path))

    assert sorted(result.artifact_roots) == [
        "aggregate_trades_tick",
        "depth_levels_tick",
        "market_bars_1m",
    ]
    market_bars = read_dataset(result.artifact_roots["market_bars_1m"])
    trades = read_dataset(result.artifact_roots["aggregate_trades_tick"])
    depth = read_dataset(result.artifact_roots["depth_levels_tick"])

    assert list(market_bars.columns) == [
        "source_asset_id",
        "timestamp",
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "source_market_type",
        "interval",
    ]
    assert market_bars.loc[0, "source_asset_id"] == "BTCUSDT"
    assert trades.loc[0, "signed_quantity"] == 2.0
    assert set(depth["side"]) == {"bid", "ask"}
    assert depth["timestamp"].map(type).eq(pd.Timestamp).all()


def _write_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
