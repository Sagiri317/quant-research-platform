from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path

from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE


def _coverage(start_ms: int, end_ms: int) -> tuple[int, int]:
    return start_ms * 1_000_000, end_ms * 1_000_000


def _quote(
    timestamp_ms: int,
    *,
    book_segment_id: str = "book-1",
    trade_connection_id: str = "trade-connection-1",
) -> dict[str, object]:
    return {
        "timestamp_ms": timestamp_ms,
        "asset_id": "btc_usdt",
        "segment_id": f"{book_segment_id}:{trade_connection_id}",
        "book_segment_id": book_segment_id,
        "trade_connection_id": trade_connection_id,
    }


def write_binance_hft_run(root: Path, *, count: int = 60) -> dict:
    root.mkdir(parents=True)
    snapshot_rows = [
        _raw(
            1,
            {
                "lastUpdateId": 100,
                "E": 0,
                "T": 0,
                "bids": [[f"{99.5 - index:.1f}", "2"] for index in range(10)],
                "asks": [[f"{100.5 + index:.1f}", "2"] for index in range(10)],
            },
        )
    ]
    depth_rows = [
        _raw(
            2,
            {
                "e": "depthUpdate",
                "E": 1,
                "T": 0,
                "s": "BTCUSDT",
                "ps": "BTCUSDT",
                "st": 1,
                "U": 90,
                "u": 100,
                "pu": 0,
                "b": [],
                "a": [],
            },
        )
    ]
    trade_rows = []
    previous_bid = "99.5"
    previous_ask = "100.5"
    for index in range(1, count + 1):
        bid = f"{99.5 + (index % 3) * 0.1:.1f}"
        ask = f"{100.5 + (index % 3) * 0.1:.1f}"
        trade_rows.append(
            _raw(
                index * 2 + 1,
                {
                    "e": "aggTrade",
                    "E": index * 100 + 1,
                    "T": index * 100,
                    "s": "BTCUSDT",
                    "ps": "BTCUSDT",
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
            )
        )
        depth_rows.append(
            _raw(
                index * 2 + 2,
                {
                    "e": "depthUpdate",
                    "E": index * 100 + 1,
                    "T": index * 100,
                    "s": "BTCUSDT",
                    "ps": "BTCUSDT",
                    "st": 1,
                    "U": 100 + index,
                    "u": 100 + index,
                    "pu": 99 + index,
                    "b": [[previous_bid, "0"], [bid, "2"]],
                    "a": [[previous_ask, "0"], [ask, "2"]],
                },
            )
        )
        previous_bid, previous_ask = bid, ask
    files = [
        _write_file(
            root,
            data_kind="depth_snapshot",
            stream="REST:/fapi/v1/depth",
            rows=snapshot_rows,
        ),
        _write_file(
            root,
            data_kind="depth",
            stream="btcusdt@depth@100ms",
            rows=depth_rows,
        ),
        _write_file(
            root,
            data_kind="agg_trades",
            stream="btcusdt@aggTrade",
            rows=trade_rows,
        ),
    ]
    connection_intervals = [
        {
            "market": FUTURES_MARKET_TYPE,
            "data_kind": "depth",
            "stream": "btcusdt@depth@100ms",
            "connection_id": "book-connection-1",
            "segment_id": "segment-1",
            "opened_at": "1970-01-01T00:00:00.000000Z",
            "opened_monotonic_ns": 0,
            "closed_at": "1970-01-01T00:00:10.000000Z",
            "closed_monotonic_ns": 10_000_000_000,
            "close_reason": "stop",
        },
        {
            "market": FUTURES_MARKET_TYPE,
            "data_kind": "agg_trades",
            "stream": "btcusdt@aggTrade",
            "connection_id": "trade-connection-1",
            "segment_id": None,
            "opened_at": "1970-01-01T00:00:00.000000Z",
            "opened_monotonic_ns": 0,
            "closed_at": "1970-01-01T00:00:10.000000Z",
            "closed_monotonic_ns": 10_000_000_000,
            "close_reason": "stop",
        },
    ]
    sidecar = root / "connection_intervals.jsonl.gz"
    with gzip.open(sidecar, "wt", encoding="utf-8") as handle:
        for interval in connection_intervals:
            handle.write(json.dumps(interval, separators=(",", ":")))
            handle.write("\n")
    sidecar_bytes = sidecar.read_bytes()
    manifest = {
        "schema_id": "qrp-binance-recorder",
        "run_id": "workflow-run",
        "status": "PASS",
        "market_data_status": "PASS",
        "venue": "binance",
        "symbol": "BTCUSDT",
        "market_types": [FUTURES_MARKET_TYPE],
        "market_metadata": {
            FUTURES_MARKET_TYPE: {"tick_size": "0.1", "snapshot_limit": 1000}
        },
        "capture": {
            "connection_intervals": {
                "path": "connection_intervals.jsonl.gz",
                "record_count": len(connection_intervals),
                "compressed_bytes": len(sidecar_bytes),
                "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            }
        },
        "files": files,
    }
    (root / "run.json").write_text(
        json.dumps(manifest, separators=(",", ":")),
        encoding="utf-8",
    )
    return manifest


def _raw(
    capture_seq: int,
    payload: dict,
    *,
    segment: bool = True,
    connection_id: str | None = None,
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
        "connection_id": connection_id
        or ("book-connection-1" if segment else "trade-connection-1"),
        "payload": payload,
    }
    if segment:
        row["segment_id"] = "segment-1"
    return row


def _write_file(
    root: Path,
    *,
    data_kind: str,
    stream: str,
    rows: list[dict],
) -> dict:
    path = (
        root
        / "date=2026-08-27"
        / "hour=00"
        / f"{FUTURES_MARKET_TYPE}_{data_kind}.jsonl.gz"
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
        "symbol": "BTCUSDT",
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
