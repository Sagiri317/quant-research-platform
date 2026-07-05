from __future__ import annotations

import json
from pathlib import Path

from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.processed import (
    materializer,
)
from quant_research_platform.sources.binance.interim import materialize_interim_data
from quant_research_platform.storage.datasets import read_dataset
from quant_research_platform.storage.layout import StorageLayout


def test_crypto_processed_materializer_builds_panel_factors_and_labels(
    tmp_path: Path,
) -> None:
    _write_binance_fixture(tmp_path)
    layout = StorageLayout(project_root=tmp_path)
    materialize_interim_data(layout=layout)

    result = materializer.materialize_processed_data(
        layout=layout,
        source_id=SourceId("binance"),
        time_window=TimeWindow("2026-07-06", "2026-07-06"),
        kinds=frozenset(
            {
                ProcessedDataKind.PANEL,
                ProcessedDataKind.FACTOR,
                ProcessedDataKind.LABEL,
            }
        ),
    )

    assert sorted(result) == ["forward_returns", "market_panel_1m", "microstructure"]
    panel = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.PANEL, "market_panel_1m")
    )
    factors = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.FACTOR, "microstructure")
    )
    labels = read_dataset(
        layout.processed_data_root("crypto", ProcessedDataKind.LABEL, "forward_returns")
    )

    assert panel.loc[0, "market_id"] == "crypto"
    assert panel.loc[0, "asset_id"] == "BTCUSDT"
    assert panel["eligible"].all()
    assert "source_asset_id" not in panel.columns
    assert {"spread", "relative_spread", "flow_imbalance_5m"}.issubset(
        set(factors.columns)
    )
    assert {"forward_return_1m", "forward_return_5m", "forward_return_15m"}.issubset(
        set(labels.columns)
    )


def _write_binance_fixture(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "binance"
    start_ms = 1783296000000
    for offset in range(18):
        open_time = start_ms + offset * 60_000
        _append_jsonl(
            raw_root / "spot" / "klines" / "BTCUSDT" / "1m" / "2026-07-06.jsonl",
            {
                "symbol": "BTCUSDT",
                "open_time": open_time,
                "open": str(100 + offset),
                "high": str(101 + offset),
                "low": str(99 + offset),
                "close": str(100.5 + offset),
                "volume": "10.0",
                "close_time": open_time + 59_999,
                "quote_volume": "1005.0",
                "trade_count": 7,
                "taker_buy_volume": "4.0",
                "taker_buy_quote_volume": "402.0",
            },
        )
        _append_jsonl(
            raw_root / "spot" / "agg_trades" / "BTCUSDT" / "2026-07-06.jsonl",
            {
                "symbol": "BTCUSDT",
                "aggregate_trade_id": offset + 1,
                "price": str(100.1 + offset),
                "quantity": "2.0",
                "first_trade_id": offset * 2,
                "last_trade_id": offset * 2 + 1,
                "timestamp": open_time + 1_000,
                "is_buyer_maker": offset % 2 == 0,
            },
        )
        _append_jsonl(
            raw_root / "spot" / "depth" / "BTCUSDT" / "2026-07-06.jsonl",
            {
                "symbol": "BTCUSDT",
                "timestamp": open_time + 2_000,
                "bids": [[str(100 + offset), "1.5"], [str(99.9 + offset), "2.0"]],
                "asks": [[str(100.2 + offset), "1.0"]],
            },
        )


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
