"""Small deterministic BTC/ETH recorder fixture for packaged research smoke."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path

from quant_research_platform.sources.binance.ids import (
    FUTURES_HFT_RECORDING_SCOPE,
    FUTURES_MARKET_TYPE,
    RESEARCH_SYMBOLS,
)


def write_synthetic_hft_recorder_run(root: Path, *, count: int = 60) -> dict:
    """Write the fixed, reviewable two-asset recorder run used by package smoke."""

    root.mkdir(parents=True)
    files = []
    intervals = []
    for symbol, base, sequence_offset in (
        ("BTCUSDT", 100.0, 0),
        ("ETHUSDT", 200.0, 10_000),
    ):
        book_connection = f"{symbol.lower()}-book"
        trade_connection = f"{symbol.lower()}-trades"
        snapshot = [
            _raw(
                sequence_offset + 1,
                {
                    "lastUpdateId": 100,
                    "E": 0,
                    "T": 0,
                    "bids": [[f"{base - 0.5 - level:.1f}", "2"] for level in range(10)],
                    "asks": [[f"{base + 0.5 + level:.1f}", "2"] for level in range(10)],
                },
                connection_id=book_connection,
            )
        ]
        previous_bid = f"{base - 0.5:.1f}"
        previous_ask = f"{base + 0.5:.1f}"
        depth = [
            _raw(
                sequence_offset + 2,
                {
                    "e": "depthUpdate",
                    "E": 1,
                    "T": 0,
                    "s": symbol,
                    "ps": symbol,
                    "st": 1,
                    "U": 90,
                    "u": 100,
                    "pu": 0,
                    "b": [],
                    "a": [],
                },
                connection_id=book_connection,
            )
        ]
        trades = []
        for index in range(1, count + 1):
            bid = f"{base - 0.5 + (index % 3) * 0.1:.1f}"
            ask = f"{base + 0.5 + (index % 3) * 0.1:.1f}"
            trades.append(
                _raw(
                    sequence_offset + index * 2 + 1,
                    {
                        "e": "aggTrade",
                        "E": index * 100 + 1,
                        "T": index * 100,
                        "s": symbol,
                        "ps": symbol,
                        "st": 1,
                        "a": index,
                        "f": index,
                        "l": index,
                        "p": ask,
                        "q": "0.1",
                        "nq": "0.1",
                        "m": False,
                    },
                    segment=False,
                    connection_id=trade_connection,
                )
            )
            depth.append(
                _raw(
                    sequence_offset + index * 2 + 2,
                    {
                        "e": "depthUpdate",
                        "E": index * 100 + 1,
                        "T": index * 100,
                        "s": symbol,
                        "ps": symbol,
                        "st": 1,
                        "U": 100 + index,
                        "u": 100 + index,
                        "pu": 99 + index,
                        "b": [[previous_bid, "0"], [bid, "2"]],
                        "a": [[previous_ask, "0"], [ask, "2"]],
                    },
                    connection_id=book_connection,
                )
            )
            previous_bid, previous_ask = bid, ask
        for data_kind, stream, rows in (
            (
                "metadata",
                "REST:/fapi/v1/exchangeInfo",
                [
                    _raw(
                        sequence_offset + 9_001,
                        {
                            "symbol": symbol,
                            "filters": [
                                {"filterType": "PRICE_FILTER", "tickSize": "0.1"}
                            ],
                        },
                        segment=False,
                        connection_id=f"{symbol.lower()}-metadata",
                    )
                ],
            ),
            ("depth_snapshot", "REST:/fapi/v1/depth", snapshot),
            ("depth", f"{symbol.lower()}@depth@100ms", depth),
            ("agg_trades", f"{symbol.lower()}@aggTrade", trades),
        ):
            files.append(
                _write_file(
                    root,
                    symbol=symbol,
                    data_kind=data_kind,
                    stream=stream,
                    rows=rows,
                )
            )
        for data_kind, stream, connection_id, segment_id in (
            ("depth", f"{symbol.lower()}@depth@100ms", book_connection, "segment-1"),
            ("agg_trades", f"{symbol.lower()}@aggTrade", trade_connection, None),
        ):
            intervals.append(
                {
                    "market": FUTURES_MARKET_TYPE,
                    "data_kind": data_kind,
                    "stream": stream,
                    "connection_id": connection_id,
                    "segment_id": segment_id,
                    "opened_at": "1970-01-01T00:00:00.000000Z",
                    "opened_monotonic_ns": 0,
                    "closed_at": "1970-01-01T00:00:10.000000Z",
                    "closed_monotonic_ns": 10_000_000_000,
                    "close_reason": "stop",
                }
            )
    sidecar = root / "connection_intervals.jsonl.gz"
    with gzip.open(sidecar, "wt", encoding="utf-8") as handle:
        for interval in intervals:
            handle.write(json.dumps(interval, separators=(",", ":")) + "\n")
    sidecar_bytes = sidecar.read_bytes()
    manifest = {
        "schema_id": "qrp-binance-recorder",
        "run_id": "workflow-run",
        "recording_scope": FUTURES_HFT_RECORDING_SCOPE,
        "status": "PASS",
        "market_data_status": "PASS",
        "venue": "binance",
        "symbols": list(RESEARCH_SYMBOLS),
        "market_types": [FUTURES_MARKET_TYPE],
        "market_metadata": {
            symbol: {FUTURES_MARKET_TYPE: {"tick_size": "0.1", "snapshot_limit": 1000}}
            for symbol in RESEARCH_SYMBOLS
        },
        "capture": {
            "connection_intervals": {
                "path": sidecar.name,
                "record_count": len(intervals),
                "compressed_bytes": len(sidecar_bytes),
                "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            }
        },
        "files": files,
    }
    (root / "run.json").write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )
    return manifest


def _raw(
    capture_seq: int,
    payload: dict,
    *,
    segment: bool = True,
    connection_id: str,
) -> dict:
    observed = payload.get("T", payload.get("E", 0))
    received = dt.datetime(1970, 1, 1, tzinfo=dt.UTC) + dt.timedelta(
        milliseconds=int(observed)
    )
    row = {
        "capture_seq": capture_seq,
        "receive_time": received.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "observed_time_ns": int(observed) * 1_000_000,
        "receive_monotonic_ns": capture_seq,
        "connection_id": connection_id,
        "payload": payload,
    }
    if segment:
        row["segment_id"] = "segment-1"
    return row


def _write_file(
    root: Path,
    *,
    symbol: str,
    data_kind: str,
    stream: str,
    rows: list[dict],
) -> dict:
    path = (
        root
        / "date=2026-08-27"
        / "hour=00"
        / f"{symbol}_{FUTURES_MARKET_TYPE}_{data_kind}.jsonl.gz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    maximum = 0
    with gzip.open(path, "wb") as handle:
        for row in rows:
            line = json.dumps(row, separators=(",", ":")).encode() + b"\n"
            maximum = max(maximum, len(line))
            handle.write(line)
    raw = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "venue": "binance",
        "symbol": symbol,
        "market": FUTURES_MARKET_TYPE,
        "stream": stream,
        "data_kind": data_kind,
        "record_count": len(rows),
        "capture_seq_min": rows[0]["capture_seq"],
        "capture_seq_max": rows[-1]["capture_seq"],
        "max_record_bytes": maximum,
        "compressed_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
