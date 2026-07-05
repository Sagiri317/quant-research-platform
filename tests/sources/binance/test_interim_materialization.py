from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_research_platform.sources.binance.interim import materialize_interim_data
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.layout import StorageLayout


def test_binance_interim_materializer_writes_source_datasets(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "binance"
    _append_jsonl(
        raw_root / "spot" / "klines" / "BTCUSDT" / "1m" / "2026-07-06.jsonl",
        _kline_envelope(0),
    )
    _append_jsonl(
        raw_root / "spot" / "agg_trades" / "BTCUSDT" / "2026-07-06.jsonl",
        _trade_envelope(0),
    )
    _append_jsonl(
        raw_root / "spot" / "depth" / "BTCUSDT" / "2026-07-06.jsonl",
        _depth_envelope(0),
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
    assert trades.loc[0, "source_asset_id"] == "BTCUSDT"
    assert trades.loc[0, "signed_quantity"] == 2.0
    assert set(depth["side"]) == {"bid", "ask"}
    assert depth["timestamp"].map(type).eq(pd.Timestamp).all()
    assert depth["timestamp"].nunique() == 1


def test_binance_interim_materializer_writes_multiple_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_root = tmp_path / "data" / "raw" / "binance"
    for offset in range(5):
        _append_jsonl(
            raw_root / "spot" / "klines" / "BTCUSDT" / "1m" / "2026-07-06.jsonl",
            _kline_envelope(offset),
        )
    chunk_lengths: list[int] = []

    def record_write_dataset(dataset_root, data, *, spec, instance_metadata=None):
        chunks = tuple(data)
        chunk_lengths.extend(len(chunk) for chunk in chunks)
        from quant_research_platform.storage.datasets import write_dataset

        return write_dataset(
            dataset_root,
            chunks,
            spec=spec,
            instance_metadata=instance_metadata,
        )

    monkeypatch.setattr(
        "quant_research_platform.sources.binance.interim.write_dataset",
        record_write_dataset,
    )

    result = materialize_interim_data(
        layout=StorageLayout(project_root=tmp_path),
        data_ids=("market_bars_1m",),
        row_batch_size=2,
    )
    market_bars = read_dataset(result.artifact_roots["market_bars_1m"])

    assert chunk_lengths == [2, 2, 1]
    assert len(market_bars) == 5


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _kline_envelope(offset: int) -> dict[str, object]:
    open_time = 1783296000000 + offset * 60_000
    return {
        "source": "binance",
        "market_type": "spot",
        "data_kind": "klines",
        "symbol": "BTCUSDT",
        "interval": "1m",
        "captured_at": "2026-07-06T00:00:02Z",
        "payload": [
            open_time,
            str(100 + offset),
            str(101 + offset),
            str(99 + offset),
            str(100.5 + offset),
            "10.0",
            open_time + 59_999,
            "1005.0",
            7,
            "4.0",
            "402.0",
            "0",
        ],
    }


def _trade_envelope(offset: int) -> dict[str, object]:
    open_time = 1783296000000 + offset * 60_000
    return {
        "source": "binance",
        "market_type": "spot",
        "data_kind": "agg_trades",
        "symbol": "BTCUSDT",
        "captured_at": "2026-07-06T00:00:02Z",
        "payload": {
            "a": offset + 1,
            "p": str(100.1 + offset),
            "q": "2.0",
            "f": offset * 2,
            "l": offset * 2 + 1,
            "T": open_time + 1_000,
            "m": False,
            "M": True,
        },
    }


def _depth_envelope(offset: int) -> dict[str, object]:
    open_time = 1783296000000 + offset * 60_000
    return {
        "source": "binance",
        "market_type": "spot",
        "data_kind": "depth",
        "symbol": "BTCUSDT",
        "captured_at": pd.Timestamp(open_time + 2_000, unit="ms", tz="UTC").isoformat(),
        "payload": {
            "lastUpdateId": offset + 1,
            "bids": [[str(100 + offset), "1.5"], [str(99.9 + offset), "2.0"]],
            "asks": [[str(100.2 + offset), "1.0"]],
        },
    }
