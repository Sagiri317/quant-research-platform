"""Normalized archive identity and bounded causal observation input checks."""

import gzip
import io
import json

import pytest

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.sources.tardis.normalized import (
    download_dataset,
    iter_records,
)


class Response(io.BytesIO):
    status = 200


def test_normalized_original_bytes_nulls_clock_and_hash(tmp_path, monkeypatch):
    raw = gzip.compress(
        b"exchange,symbol,timestamp,local_timestamp,id,side,price,amount\nbybit,BTCUSDT,1785542400000000,1785542400000001,1,buy,100,2\nbybit,BTCUSDT,1785542400000000,1785542400000002,2,sell,100,1\n",
        mtime=0,
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(raw))
    root = tmp_path / "archive"
    manifest = download_dataset(
        root,
        exchange="bybit",
        dataset_type="trades",
        symbol="BTCUSDT",
        date="2026-08-01",
        byte_budget=1024**2,
    )
    assert (root / "data.csv.gz").read_bytes() == raw
    assert manifest["row_count"] == 2
    rows = list(
        iter_records(
            root.parent,
            archive_paths=(root.name,),
            time_window=TimeWindow("2026-08-01", "2026-08-01"),
        )
    )
    assert rows[0]["amount"] == 2.0
    assert rows[0]["timestamp"] < rows[0]["local_timestamp"]
    (root / "data.csv.gz").write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises((ValueError, OSError)):
        list(
            iter_records(
                root.parent,
                archive_paths=(root.name,),
                time_window=TimeWindow("2026-08-01", "2026-08-01"),
            )
        )


def test_normalized_rejects_wrong_schema_and_preserves_existing_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: Response(gzip.compress(b"bad,header\n")),
    )
    with pytest.raises(ValueError, match="schema_invalid"):
        download_dataset(
            tmp_path / "bad",
            exchange="okex-swap",
            dataset_type="quotes",
            symbol="BTC-USDT-SWAP",
            date="2026-08-01",
            byte_budget=1024**2,
        )
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="scope_invalid"):
        download_dataset(
            tmp_path / "other",
            exchange="bybit",
            dataset_type="incremental_book_L2",
            symbol="BTCUSDT",
            date="2026-08-01",
            byte_budget=1024**2,
        )


def test_normalized_manifest_mutation_during_iteration(tmp_path, monkeypatch):
    raw = gzip.compress(
        b"exchange,symbol,timestamp,local_timestamp,id,side,price,amount\nbybit,ETHUSDT,1785542400000000,1785542400000001,1,buy,100,2\n",
        mtime=0,
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Response(raw))
    root = tmp_path / "archive"
    manifest = download_dataset(
        root,
        exchange="bybit",
        dataset_type="trades",
        symbol="ETHUSDT",
        date="2026-08-01",
        byte_budget=1024**2,
    )
    rows = iter_records(
        root.parent,
        archive_paths=(root.name,),
        time_window=TimeWindow("2026-08-01", "2026-08-01"),
    )
    next(rows)
    manifest["coverage"] = "changed"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_changed"):
        rows.close()
