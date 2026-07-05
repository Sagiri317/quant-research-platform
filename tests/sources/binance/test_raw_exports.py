from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_research_platform.sources.binance.raw_exports import (
    dataset_files_by_id,
    iter_raw_rows,
)


def test_binance_dataset_files_by_id_groups_local_raw_jsonl(tmp_path: Path) -> None:
    raw_root = tmp_path / "data" / "raw" / "binance"
    kline_path = raw_root / "spot" / "klines" / "BTCUSDT" / "1m" / "2026-07-06.jsonl"
    trade_path = (
        raw_root / "spot" / "agg_trades" / "BTCUSDT" / "2026-07-06.jsonl"
    )
    depth_path = raw_root / "spot" / "depth" / "BTCUSDT" / "2026-07-06.jsonl"
    for path in (kline_path, trade_path, depth_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"symbol": "BTCUSDT"}) + "\n", encoding="utf-8")

    files = dataset_files_by_id(raw_root)

    assert files == {
        "market_bars_1m": (kline_path,),
        "aggregate_trades_tick": (trade_path,),
        "depth_levels_tick": (depth_path,),
    }


def test_binance_raw_rows_reject_missing_empty_and_malformed_files(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.jsonl"
    empty = tmp_path / "empty.jsonl"
    malformed = tmp_path / "malformed.jsonl"
    empty.write_text("", encoding="utf-8")
    malformed.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        tuple(iter_raw_rows(missing))
    with pytest.raises(ValueError, match="empty_binance_raw_file"):
        tuple(iter_raw_rows(empty))
    with pytest.raises(ValueError, match="malformed_binance_raw_row"):
        tuple(iter_raw_rows(malformed))
