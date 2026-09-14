from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
from functools import wraps
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.portfolios import FundingForecast
from quant_research_platform.markets.implementations.crypto.hft import (
    iter_panel_frames,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.binance.ids import (
    FUTURES_HFT_RECORDING_SCOPE,
    FUTURES_MARKET_TYPE,
    RESEARCH_SYMBOLS,
)
from tests.helpers.runtime_resources import capture_test_policy

# Synthetic event fixtures explicitly declare this known no-funding interval.
KNOWN_NO_FUNDING = (FundingForecast(0, 1_000_000_000_000, ()),)


def hft_research_plan():
    """Explicit plan paired with the 90-step, two-asset recorder fixture."""

    def window(start, end):
        return {"start": f"1970-01-01T00:00:{start}", "end": f"1970-01-01T00:00:{end}"}

    return {
        "folds": [
            {
                "fold_id": "fold_1",
                "train": window("00", "02"),
                "validation": window("02.100000", "03.500000"),
                "development_test": window("04", "05"),
            },
            {
                "fold_id": "fold_2",
                "train": window("01", "04"),
                "validation": window("04.100000", "06"),
                "development_test": window("06.500000", "08.500000"),
            },
        ],
        "final_holdout": window("08.600000", "09"),
        "initial_capital": 1000.0,
        "valuation_currency": "USDT",
        "exposure_limits": {
            asset: {"max_order_quantity": 1.0, "max_position_quantity": 2.0}
            for asset in ("btc_usdt", "eth_usdt")
        },
        "execution": {
            "taker_fee_bps": 1.0,
            "latency_ms": 10,
            "markout_horizon_ms": 500,
            "latency_scenarios_ms": [0, 100, 200],
            "fee_scenarios_bps": [0.5, 1.0, 2.0],
            "additional_spread_scenarios_bps": [0.0, 1.0, 2.0],
            "maker": {
                "maker_fee_bps": -1.0,
                "latency_ms": 10,
                "min_calibration_samples": 2,
                "inventory_target": 0.0,
                "inventory_skew": 0.0,
            },
        },
    }


def write_hft_funding_fixture(raw_root: Path) -> None:
    """Explicit synthetic calendar, settlements, and separately known cash forecasts."""
    from quant_research_platform.sources.binance.funding import (
        FUNDING_COLUMNS,
        FUNDING_SCHEMA_ID,
    )
    from quant_research_platform.storage.manifests import read_manifest
    from quant_research_platform.storage.raw_snapshots import (
        iter_csv_snapshot,
        write_csv_snapshot,
    )

    with runtime_resource_policy_scope():
        for symbol, mark in (("BTCUSDT", 100.0), ("ETHUSDT", 200.0)):
            metadata = {
                "schema_id": FUNDING_SCHEMA_ID,
                "symbol": symbol,
                "market_type": FUTURES_MARKET_TYPE,
                "valuation_currency": "USDT",
                "coverage": {
                    "start_ns": 0,
                    "end_ns": 10_000_000_000,
                    "status": "COMPLETE",
                },
                "provenance": "synthetic_explicit_calendar_and_known_forecasts",
                "settlement_source": None,
            }
            rows = (
                ("settlement", 4_800_000_000, None, -0.02, mark, None),
                ("settlement", 5_700_000_000, None, 0.001, mark, None),
                ("settlement", 7_400_000_000, None, 0.02, mark, None),
                (
                    "forecast",
                    0,
                    12_000_000_000,
                    None,
                    None,
                    [
                        [4_800_000_000, mark * 0.02],
                        [5_700_000_000, -mark * 0.001],
                        [7_400_000_000, -mark * 0.02],
                    ],
                ),
            )

            def validate(root, metadata=metadata):
                manifest = read_manifest(root / "response.metadata.json")
                assert manifest["metadata"] == metadata
                for _row in iter_csv_snapshot(
                    root,
                    expected_manifest=manifest,
                    max_bytes=1024 * 1024,
                ):
                    pass

            write_csv_snapshot(
                raw_root.parent / "funding" / symbol,
                columns=FUNDING_COLUMNS,
                rows=rows,
                metadata=metadata,
                max_bytes=1024 * 1024,
                validate=validate,
            )


def with_test_resource_policy(function):  # noqa: ANN001, ANN201
    @wraps(function)
    def invoke(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
        for name in ("on_fill", "on_equity"):
            observer = kwargs.get(name)
            if observer is not None:
                kwargs[name] = lambda value, sink=observer: sink(dict(value))
        with runtime_resource_policy_scope(policy):
            return function(*args, resource_policy=policy, **kwargs)

    return invoke


def _coverage(start_ms: int, end_ms: int) -> tuple[int, int]:
    return start_ms * 1_000_000, end_ms * 1_000_000


def _quote(
    timestamp_ms: int,
    *,
    book_segment_id: str = "book-1",
    trade_observation_id: str = "trade-connection-1",
) -> dict[str, object]:
    return {
        "timestamp_ms": timestamp_ms,
        "asset_id": "btc_usdt",
        "book_segment_id": book_segment_id,
        "trade_observation_id": trade_observation_id,
    }


def _panel_rows(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
    kwargs["trade_coverage"] = coverage_lookup(kwargs["trade_coverage"])
    frames = list(
        iter_panel_frames(
            *args,
            market_id="crypto",
            asset_id="btc_usdt",
            batch_rows=2,
            **kwargs,
        )
    )
    if not frames:
        return []
    return pd.concat(frames, ignore_index=True).to_dict("records")


def _row_timestamp_ms(row: dict[str, object]) -> int:
    return pd.Timestamp(row["timestamp"]).value // 1_000_000


def write_binance_hft_run(
    root: Path, *, count: int = 60, maker_liquidity: bool = False
) -> dict:
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
                    "p": bid if maker_liquidity and index % 2 else ask,
                    "q": "3" if maker_liquidity else "0.1",
                    "nq": "3" if maker_liquidity else "0.1",
                    "m": bool(maker_liquidity and index % 2),
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
            symbol="BTCUSDT",
            data_kind="metadata",
            stream="REST:/fapi/v1/exchangeInfo",
            rows=[
                _raw(
                    9_001,
                    {
                        "symbol": "BTCUSDT",
                        "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.1"}],
                    },
                    segment=False,
                    connection_id="btc-metadata",
                )
            ],
        ),
        _write_file(
            root,
            symbol="BTCUSDT",
            data_kind="depth_snapshot",
            stream="REST:/fapi/v1/depth",
            rows=snapshot_rows,
        ),
        _write_file(
            root,
            symbol="BTCUSDT",
            data_kind="depth",
            stream="btcusdt@depth@100ms",
            rows=depth_rows,
        ),
        _write_file(
            root,
            symbol="BTCUSDT",
            data_kind="agg_trades",
            stream="btcusdt@aggTrade",
            rows=trade_rows,
        ),
        _write_file(
            root,
            symbol="ETHUSDT",
            data_kind="metadata",
            stream="REST:/fapi/v1/exchangeInfo",
            rows=[
                _raw(
                    19_001,
                    {
                        "symbol": "ETHUSDT",
                        "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.1"}],
                    },
                    segment=False,
                    connection_id="eth-metadata",
                )
            ],
        ),
        _write_file(
            root,
            symbol="ETHUSDT",
            data_kind="depth_snapshot",
            stream="REST:/fapi/v1/depth",
            rows=[
                _raw(
                    10_001,
                    {
                        "lastUpdateId": 100,
                        "E": 0,
                        "T": 0,
                        "bids": [[f"{199.5 - index:.1f}", "2"] for index in range(10)],
                        "asks": [[f"{200.5 + index:.1f}", "2"] for index in range(10)],
                    },
                    connection_id="eth-book-connection-1",
                )
            ],
        ),
        _write_file(
            root,
            symbol="ETHUSDT",
            data_kind="depth",
            stream="ethusdt@depth@100ms",
            rows=[
                {
                    **row,
                    "capture_seq": int(row["capture_seq"]) + 10_000,
                    "receive_monotonic_ns": int(row["capture_seq"]) + 10_000,
                    "connection_id": "eth-book-connection-1",
                    "payload": {
                        **row["payload"],
                        "s": "ETHUSDT",
                        "ps": "ETHUSDT",
                        "b": [
                            [f"{float(price) + 100.0:.1f}", quantity]
                            for price, quantity in row["payload"]["b"]
                        ],
                        "a": [
                            [f"{float(price) + 100.0:.1f}", quantity]
                            for price, quantity in row["payload"]["a"]
                        ],
                    },
                }
                for row in depth_rows
            ],
        ),
        _write_file(
            root,
            symbol="ETHUSDT",
            data_kind="agg_trades",
            stream="ethusdt@aggTrade",
            rows=[
                {
                    **row,
                    "capture_seq": int(row["capture_seq"]) + 10_000,
                    "receive_monotonic_ns": int(row["capture_seq"]) + 10_000,
                    "connection_id": "eth-trade-connection-1",
                    "payload": {
                        **row["payload"],
                        "s": "ETHUSDT",
                        "ps": "ETHUSDT",
                        "p": f"{float(row['payload']['p']) + 100.0:.1f}",
                    },
                }
                for row in trade_rows
            ],
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
            "data_kind": "depth",
            "stream": "ethusdt@depth@100ms",
            "connection_id": "eth-book-connection-1",
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
            "stream": "ethusdt@aggTrade",
            "connection_id": "eth-trade-connection-1",
            "segment_id": None,
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
        "recording_scope": FUTURES_HFT_RECORDING_SCOPE,
        "status": "PASS",
        "market_data_status": "PASS",
        "venue": "binance",
        "symbols": list(RESEARCH_SYMBOLS),
        "market_types": [FUTURES_MARKET_TYPE],
        "market_metadata": {
            "BTCUSDT": {
                FUTURES_MARKET_TYPE: {"tick_size": "0.1", "snapshot_limit": 1000}
            },
            "ETHUSDT": {
                FUTURES_MARKET_TYPE: {"tick_size": "0.1", "snapshot_limit": 1000}
            },
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


def coverage_lookup(mapping):
    return lambda start, end: tuple(
        key for key, bounds in mapping.items() if bounds[0] <= start and bounds[1] > end
    )
