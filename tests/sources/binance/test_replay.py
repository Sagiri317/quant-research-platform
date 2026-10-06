from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from datetime import UTC
from decimal import Decimal, localcontext
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.contracts.order_book as order_book
import quant_research_platform.sources.binance.replay as replay
import quant_research_platform.storage.manifests as storage_manifests
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeEvent,
    index_to_price,
    price_to_index,
)
from quant_research_platform.sources.binance.ids import (
    FUTURES_HFT_RECORDING_SCOPE,
    FUTURES_MARKET_TYPE,
    RESEARCH_SYMBOLS,
    SPOT_L2_RECORDING_SCOPE,
    SPOT_MARKET_TYPE,
)
from quant_research_platform.sources.binance.replay import (
    iter_agg_trades,
    iter_reconstructed_depth,
    open_recorder_run,
)
from quant_research_platform.storage.identity import (
    filesystem_glob,
    filesystem_mkdir,
    filesystem_read_bytes,
    filesystem_rename,
    filesystem_stat,
    filesystem_unlink,
    filesystem_write_bytes,
    filesystem_write_text,
    native_filesystem_path,
    open_filesystem_file,
)


def _raw(
    capture_seq: int,
    payload: dict[str, object],
    *,
    segment_id: str,
    connection_id: str = "connection-1",
) -> dict[str, object]:
    return {
        "capture_seq": capture_seq,
        "receive_time": "2026-08-27T00:00:00.123456Z",
        "observed_time_ns": pd.Timestamp("2026-08-27T00:00:00.123456Z").value,
        "receive_monotonic_ns": capture_seq * 10,
        "connection_id": connection_id,
        "segment_id": segment_id,
        "payload": payload,
    }


def test_connection_coverage_gzip_constructor_failure_closes_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = KeyboardInterrupt("gzip_constructor_primary")
    raw_handle = io.BytesIO(b"not-consumed")

    def fail_gzip(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(replay.gzip, "GzipFile", fail_gzip)
    monkeypatch.setattr(
        replay, "open_filesystem_file", lambda *_args, **_kwargs: raw_handle
    )
    index = object.__new__(replay.ObservationCoverageIndex)
    index._path = Path("synthetic-coverage.gz")
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            index._load(markets={FUTURES_MARKET_TYPE})
        assert raised.value is primary
        assert raw_handle.closed
    finally:
        raw_handle.close()


def test_connection_coverage_gzip_close_failure_still_closes_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class RawHandle(io.BytesIO):
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            super().close()

    class FakeGzip:
        close_calls = 0

        def readline(self, _size: int) -> bytes:
            return b""

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 2:
                raise OSError("gzip-close-secondary")

    raw_handle = RawHandle()
    gzip_handle = FakeGzip()
    monkeypatch.setattr(replay.gzip, "GzipFile", lambda **_kwargs: gzip_handle)
    monkeypatch.setattr(
        replay, "open_filesystem_file", lambda *_args, **_kwargs: raw_handle
    )
    monkeypatch.setattr(
        replay, "_verify_raw_file_handle", lambda *_args, **_kwargs: None
    )
    index = object.__new__(replay.ObservationCoverageIndex)
    index._path = Path("synthetic-coverage.gz")
    index.identity = object()
    index._descriptor = {"record_count": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    try:
        with pytest.raises(OSError, match="gzip-close-secondary"):
            index._load(markets={FUTURES_MARKET_TYPE})
        assert raw_handle.close_calls == 1
    finally:
        raw_handle.close()


def test_replay_file_handle_cleanup_preserves_hash_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class FailingHandle:
        def close(self) -> None:
            raise OSError("handle_close")

    monkeypatch.setattr(
        replay, "open_filesystem_file", lambda *_args, **_kwargs: FailingHandle()
    )
    monkeypatch.setattr(
        replay,
        "_verify_raw_file_handle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("hash_primary")
        ),
    )
    with pytest.raises(KeyboardInterrupt, match="hash_primary") as raised:
        replay._sha256_file(Path("synthetic-member.gz"), object())
    assert raised.value.__notes__ == [
        "Binance replay file handle close also failed: OSError"
    ]


def _snapshot(
    capture_seq: int,
    *,
    market_type: str,
    segment_id: str,
    last_update_id: int,
    bids: list[list[str]] | None = None,
    asks: list[list[str]] | None = None,
    connection_id: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "lastUpdateId": last_update_id,
        "bids": bids or [["100.0", "2"], ["99.0", "3"]],
        "asks": asks or [["101.0", "2"], ["102.0", "3"]],
    }
    if market_type == FUTURES_MARKET_TYPE:
        payload.update({"E": 90, "T": 89})
    return _raw(
        capture_seq,
        payload,
        segment_id=segment_id,
        connection_id=connection_id or f"connection-{segment_id}",
    )


def _futures_delta(
    capture_seq: int,
    first: int,
    last: int,
    *,
    previous: int,
    transaction_time: int,
    segment_id: str,
    bids: list[list[str]] | None = None,
    asks: list[list[str]] | None = None,
    connection_id: str | None = None,
) -> dict[str, object]:
    return _raw(
        capture_seq,
        {
            "e": "depthUpdate",
            "E": transaction_time + 2,
            "T": transaction_time,
            "s": "BTCUSDT",
            "ps": "BTCUSDT",
            "st": 1,
            "U": first,
            "u": last,
            "pu": previous,
            "b": bids or [],
            "a": asks or [],
        },
        segment_id=segment_id,
        connection_id=connection_id or f"connection-{segment_id}",
    )


def _spot_delta(
    capture_seq: int,
    first: int,
    last: int,
    *,
    event_time: int,
    segment_id: str,
    connection_id: str | None = None,
) -> dict[str, object]:
    return _raw(
        capture_seq,
        {
            "e": "depthUpdate",
            "E": event_time,
            "s": "BTCUSDT",
            "U": first,
            "u": last,
            "b": [],
            "a": [],
        },
        segment_id=segment_id,
        connection_id=connection_id or f"connection-{segment_id}",
    )


def _agg_trade(
    capture_seq: int, *, quantity: str = "2.0", visible_quantity: str = "1.0"
) -> dict[str, object]:
    return {
        "capture_seq": capture_seq,
        "receive_time": "2026-08-27T00:00:00.123456Z",
        "observed_time_ns": pd.Timestamp("2026-08-27T00:00:00.123456Z").value,
        "receive_monotonic_ns": capture_seq * 10,
        "connection_id": "trade-connection-1",
        "payload": {
            "e": "aggTrade",
            "E": 101,
            "T": 100,
            "s": "BTCUSDT",
            "ps": "BTCUSDT",
            "st": 1,
            "a": 1,
            "f": 1,
            "l": 1,
            "p": "100.0",
            "q": quantity,
            "nq": visible_quantity,
            "m": True,
        },
    }


def _write_file(
    root: Path,
    *,
    market_type: str,
    data_kind: str,
    stream: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    path = root / "date=2026-08-27" / "hour=00" / f"{market_type}_{data_kind}.jsonl.gz"
    filesystem_mkdir(path.parent, parents=True, exist_ok=True)
    max_record_bytes = 0
    with gzip.open(native_filesystem_path(path), "wb") as handle:
        for row in rows:
            line = (
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
                + b"\n"
            )
            max_record_bytes = max(max_record_bytes, len(line))
            handle.write(line)
    raw = filesystem_read_bytes(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "venue": "binance",
        "symbol": "BTCUSDT",
        "market": market_type,
        "stream": stream,
        "data_kind": data_kind,
        "record_count": len(rows),
        "capture_seq_min": min(row["capture_seq"] for row in rows),
        "capture_seq_max": max(row["capture_seq"] for row in rows),
        "max_record_bytes": max_record_bytes,
        "compressed_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _write_run(
    root: Path,
    files: list[tuple[str, str, str, list[dict[str, object]]]],
    *,
    run_id: str = "run-1",
) -> dict[str, object]:
    filesystem_mkdir(root, parents=True, exist_ok=True)
    file_manifests = [
        _write_file(root, market_type=market, data_kind=kind, stream=stream, rows=rows)
        for market, kind, stream, rows in files
    ]
    connection_intervals: list[dict[str, object]] = []
    seen_connections: set[str] = set()
    for market, kind, stream, rows in files:
        if kind not in {"depth", "agg_trades"}:
            continue
        for row in rows:
            connection_id = str(row["connection_id"])
            if connection_id in seen_connections:
                continue
            seen_connections.add(connection_id)
            connection_intervals.append(
                {
                    "market": market,
                    "data_kind": kind,
                    "stream": stream,
                    "connection_id": connection_id,
                    "segment_id": str(row["segment_id"]) if kind == "depth" else None,
                    "opened_at": "2026-08-27T00:00:00.000000Z",
                    "opened_monotonic_ns": 0,
                    "closed_at": "2026-08-27T00:00:01.000000Z",
                    "closed_monotonic_ns": 1000000000,
                    "close_reason": "stop",
                }
            )
    sidecar = root / "connection_intervals.jsonl.gz"
    with gzip.open(native_filesystem_path(sidecar), "wt", encoding="utf-8") as handle:
        for interval in connection_intervals:
            handle.write(json.dumps(interval, separators=(",", ":")))
            handle.write("\n")
    sidecar_bytes = filesystem_read_bytes(sidecar)
    market_types = sorted({market for market, *_ in files})
    recording_scope = (
        SPOT_L2_RECORDING_SCOPE
        if market_types == [SPOT_MARKET_TYPE]
        else FUTURES_HFT_RECORDING_SCOPE
    )
    manifest: dict[str, object] = {
        "schema_id": "qrp-binance-recorder",
        "run_id": run_id,
        "recording_scope": recording_scope,
        "status": "PARTIAL",
        "market_data_status": "PARTIAL",
        "venue": "binance",
        "symbols": list(RESEARCH_SYMBOLS),
        "market_types": market_types,
        "market_metadata": {
            symbol: {
                market: {"tick_size": "0.1", "snapshot_limit": 1000}
                for market, *_ in files
            }
            for symbol in RESEARCH_SYMBOLS
        },
        "capture": {
            "connection_intervals": {
                "path": "connection_intervals.jsonl.gz",
                "record_count": len(connection_intervals),
                "compressed_bytes": len(sidecar_bytes),
                "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            }
        },
        "files": file_manifests,
    }
    filesystem_write_text(
        root / "run.json", json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )
    return manifest


@pytest.mark.parametrize("invalid_record", [False, True])
def test_replay_merge_closes_all_streams_without_collecting_retained_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_record: bool
) -> None:
    root = tmp_path / "run"
    depth = _futures_delta(2, 1, 2, previous=0, transaction_time=100, segment_id="seg")
    if invalid_record:
        depth["receive_monotonic_ns"] = -1
    _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1), _agg_trade(3)],
            ),
            (FUTURES_MARKET_TYPE, "depth", "btcusdt@depth@100ms", [depth]),
        ],
    )
    handles = []
    original_open = replay.open_filesystem_file

    def observe_open(path: Path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if (
            path.name.endswith(".jsonl.gz")
            and path.name != "connection_intervals.jsonl.gz"
        ):
            handles.append(handle)
        return handle

    with open_recorder_run(root) as run:
        monkeypatch.setattr(replay, "open_filesystem_file", observe_open)
        records = run.iter_records(market_type=FUTURES_MARKET_TYPE)
        if invalid_record:
            with pytest.raises(ValueError, match="monotonic") as retained:
                tuple(records)
            assert retained.value is not None
        else:
            assert next(records)["capture_seq"] == 1
            records.close()
        assert handles and all(handle.closed for handle in handles)


@pytest.mark.parametrize(
    ("status", "market_data_status"),
    (("UNKNOWN", "PASS"), ("PASS", "UNKNOWN"), ("PASS", "FAIL")),
)
def test_run_manifest_requires_equal_known_statuses(
    tmp_path: Path, status: str, market_data_status: str
) -> None:
    manifest = _write_run(
        tmp_path / "run",
        [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])],
    )
    manifest["status"] = status
    manifest["market_data_status"] = market_data_status
    with pytest.raises(ValueError, match="binance_replay_run_manifest_invalid"):
        replay._validate_run_manifest(manifest)


@pytest.mark.parametrize("status", ("PARTIAL", "FAIL"))
def test_run_manifest_accepts_equal_known_statuses(tmp_path: Path, status: str) -> None:
    manifest = _write_run(
        tmp_path / "run",
        [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])],
    )
    manifest["status"] = status
    manifest["market_data_status"] = status
    replay._validate_run_manifest(manifest)


def test_pass_manifest_requires_every_btc_eth_stream_tuple(tmp_path: Path) -> None:
    manifest = _write_run(
        tmp_path / "run",
        [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])],
    )
    manifest["status"] = "PASS"
    manifest["market_data_status"] = "PASS"
    with pytest.raises(ValueError, match="^binance_replay_run_completeness_invalid$"):
        replay._validate_run_manifest(manifest)


def test_run_manifest_binds_recording_scope_to_exact_market_set(tmp_path: Path) -> None:
    manifest = _write_run(
        tmp_path / "run",
        [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])],
    )
    manifest["recording_scope"] = SPOT_L2_RECORDING_SCOPE
    with pytest.raises(ValueError, match="binance_replay_run_manifest_invalid"):
        replay._validate_run_manifest(manifest)


def test_agg_trades_normalize_all_clocks_and_visible_quantity(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_run(
        root, [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])]
    )
    with open_recorder_run(root) as run:
        trades = list(iter_agg_trades(run, FUTURES_MARKET_TYPE, symbol="BTCUSDT"))
    assert len(trades) == 1
    trade = trades[0]
    assert isinstance(trade, TradeEvent)
    assert trade.event_time_ms == 101
    assert trade.transaction_time_ms == 100
    assert trade.observed_time_ms == 1787788800123
    assert trade.quantity == 2.0
    assert trade.visible_quantity == 1.0
    assert trade.price_index == 1000


def _futures_run(root: Path, depth_rows: list[dict[str, object]]) -> None:
    segments = dict.fromkeys(str(row["segment_id"]) for row in depth_rows)
    snapshots: list[dict[str, object]] = []
    ordered_depth_rows: list[dict[str, object]] = []
    next_capture_seq = 1
    for index, segment in enumerate(segments):
        snapshots.append(
            _snapshot(
                next_capture_seq,
                market_type=FUTURES_MARKET_TYPE,
                segment_id=segment,
                last_update_id=100 if index == 0 else 200,
            )
        )
        next_capture_seq += 1
        for source in depth_rows:
            if str(source["segment_id"]) != segment:
                continue
            row = dict(source)
            row["capture_seq"] = next_capture_seq
            ordered_depth_rows.append(row)
            next_capture_seq += 1
    _write_run(
        root,
        [
            (FUTURES_MARKET_TYPE, "depth", "btcusdt@depth@100ms", ordered_depth_rows),
            (FUTURES_MARKET_TYPE, "depth_snapshot", "REST:/fapi/v1/depth", snapshots),
        ],
    )


def test_run_identity_binds_exact_manifest_files_and_capture_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    manifest = _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "depth",
                "btcusdt@depth@100ms",
                [
                    _futures_delta(
                        1, 90, 100, previous=0, transaction_time=10, segment_id="s1"
                    ),
                    _futures_delta(
                        3, 101, 101, previous=100, transaction_time=30, segment_id="s1"
                    ),
                ],
            ),
            (
                FUTURES_MARKET_TYPE,
                "depth_snapshot",
                "REST:/fapi/v1/depth",
                [
                    _snapshot(
                        2,
                        market_type=FUTURES_MARKET_TYPE,
                        segment_id="s1",
                        last_update_id=100,
                    )
                ],
            ),
        ],
    )
    with open_recorder_run(root) as run:
        records = list(run.iter_records())
        assert run.source_identity == {
            "schema_id": "qrp-binance-recorder",
            "artifact_id": "run-1",
            "manifest_hash": canonical_json_hash(manifest),
        }
    assert [record["capture_seq"] for record in records] == [1, 2, 3]


@pytest.mark.parametrize(
    ("run_id", "valid"),
    (
        ("../x", False),
        (" x", False),
        ("UPPER", False),
        ("0123456789abcdef0123456789abcdef", True),
        ("legal-lowercase.run_1", True),
    ),
)
def test_replay_run_id_uses_shared_contract(
    tmp_path: Path, run_id: str, valid: bool
) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])],
        run_id=run_id,
    )
    if valid:
        with open_recorder_run(root):
            pass
    else:
        with pytest.raises(ValueError, match="run_manifest_invalid"):
            with open_recorder_run(root):
                pass


def test_run_rejects_unlisted_or_hash_mismatched_file(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "depth",
                "btcusdt@depth@100ms",
                [
                    _futures_delta(
                        1, 90, 100, previous=0, transaction_time=10, segment_id="s1"
                    )
                ],
            )
        ],
    )
    extra = root / "extra.jsonl.gz"
    with gzip.open(native_filesystem_path(extra), "wt", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="file_manifest_mismatch"):
        with open_recorder_run(root):
            pass
    filesystem_unlink(extra)
    declared = next(
        path
        for path in filesystem_glob(root, "**/*.jsonl.gz")
        if path.name == "usd_s_m_futures_depth.jsonl.gz"
    )
    filesystem_write_bytes(declared, filesystem_read_bytes(declared) + b"tampered")
    with pytest.raises(ValueError, match="file_identity_mismatch"):
        with open_recorder_run(root):
            pass


def test_futures_bootstrap_keeps_all_clocks_and_exact_tick_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _futures_run(
        root,
        [_futures_delta(1, 90, 100, previous=0, transaction_time=10, segment_id="s1")],
    )
    with open_recorder_run(root) as run:
        events = list(
            iter_reconstructed_depth(
                run, FUTURES_MARKET_TYPE, symbol="BTCUSDT", required_levels=1
            )
        )
        state = events[0]
        assert isinstance(state, BookState)
        assert state.transaction_time_ms == 10
        assert state.event_time_ms == 12
        assert state.receive_time == "2026-08-27T00:00:00.123456Z"
        assert state.receive_monotonic_ns == 10
        assert state.source_sequence == 2
        assert state.best_bid == (1000, 2.0)
    assert price_to_index("100.0", "0.1", error="bad") == 1000
    assert index_to_price(1000, "0.1") == 100.0
    with pytest.raises(ValueError, match="bad"):
        price_to_index("100.05", "0.1", error="bad")


@pytest.mark.parametrize("precision", (1, 7, 28))
def test_tick_conversion_is_exact_independently_of_decimal_context(
    precision: int,
) -> None:
    price = "12345678901234567890.123456789"
    with localcontext() as context:
        context.prec = precision
        ticks = price_to_index(price, "0.000000001", error="off_tick")
        assert ticks == 12345678901234567890123456789
        assert index_to_price(ticks, "0.000000001") == float(Decimal(price))
        assert index_to_price(-ticks, "0.000000001") == -float(Decimal(price))
        assert index_to_price(0, "0.000000001") == 0.0
        assert price_to_index("5e-324", "5e-324", error="off_tick") == 1
        assert index_to_price(1, "5e-324") == 5e-324
        with pytest.raises(ValueError, match="off_tick"):
            replay._levels(
                [["100.00000000000000000000000000001", "1"]],
                price_unit="0.1",
                error="off_tick",
                allow_zero=False,
            )
        assert replay._levels(
            [[price, "1"]],
            price_unit="0.000000001",
            error="off_tick",
            allow_zero=False,
        ) == {ticks: 1.0}


@pytest.mark.parametrize("value", ("1e1000000000", "1e-1000000000"))
@pytest.mark.parametrize("is_tick", (False, True))
def test_extreme_decimal_exponents_reject_before_integer_expansion(
    monkeypatch: pytest.MonkeyPatch, value: str, is_tick: bool
) -> None:
    monkeypatch.setattr(
        order_book,
        "Fraction",
        lambda *_args: pytest.fail("integer expansion before float-range admission"),
    )
    with pytest.raises(
        ValueError, match="order_book_price_unit_invalid" if is_tick else "off_tick"
    ):
        price_to_index(
            "1" if is_tick else value, value if is_tick else "1", error="off_tick"
        )


def test_tick_price_output_rejects_unrepresentable_economic_price() -> None:
    with pytest.raises(ValueError, match="order_book_price_index_invalid"):
        index_to_price(10**1000, "1")


def test_snapshot_frontier_is_not_promoted_to_executable_depth(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _futures_run(
        root,
        [
            _futures_delta(
                1,
                90,
                100,
                previous=0,
                transaction_time=10,
                segment_id="s1",
                bids=[["98.0", "4"]],
            )
        ],
    )
    with open_recorder_run(root) as run:
        owned = next(
            iter_reconstructed_depth(
                run, FUTURES_MARKET_TYPE, symbol="BTCUSDT", required_levels=2
            )
        )
        assert owned.bids == ((1000, 2.0), (990, 3.0))
        assert owned.asks == ((1010, 2.0), (1020, 3.0))


def test_insufficient_snapshot_waits_for_next_segment(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "depth",
                "btcusdt@depth@100ms",
                [
                    _futures_delta(
                        2, 90, 100, previous=0, transaction_time=10, segment_id="s1"
                    ),
                    _futures_delta(
                        4, 190, 200, previous=0, transaction_time=20, segment_id="s2"
                    ),
                ],
            ),
            (
                FUTURES_MARKET_TYPE,
                "depth_snapshot",
                "REST:/fapi/v1/depth",
                [
                    _snapshot(
                        1,
                        market_type=FUTURES_MARKET_TYPE,
                        segment_id="s1",
                        last_update_id=100,
                        bids=[["100.0", "2"]],
                        asks=[["101.0", "2"]],
                    ),
                    _snapshot(
                        3,
                        market_type=FUTURES_MARKET_TYPE,
                        segment_id="s2",
                        last_update_id=200,
                    ),
                ],
            ),
        ],
    )
    with open_recorder_run(root) as run:
        events = list(
            iter_reconstructed_depth(
                run, FUTURES_MARKET_TYPE, symbol="BTCUSDT", required_levels=2
            )
        )
        assert isinstance(events[0], BookBoundary)
        assert events[0].reason == "unsupported_coverage"
        assert isinstance(events[1], BookState)
        assert events[1].segment_id == "s2"


def test_futures_zero_transaction_time_is_not_replaced_by_event_time(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _futures_run(
        root,
        [_futures_delta(1, 90, 100, previous=0, transaction_time=0, segment_id="s1")],
    )
    with open_recorder_run(root) as run:
        state = next(
            event
            for event in iter_reconstructed_depth(
                run, FUTURES_MARKET_TYPE, symbol="BTCUSDT", required_levels=1
            )
            if isinstance(event, BookState)
        )
        assert state.transaction_time_ms == 0
        assert state.observed_time_ms > state.transaction_time_ms


def test_spot_uses_snapshot_plus_one_bootstrap(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [
            (
                SPOT_MARKET_TYPE,
                "depth",
                "btcusdt@depth@100ms",
                [
                    _spot_delta(2, 10, 11, event_time=10, segment_id="s1"),
                    _spot_delta(3, 12, 12, event_time=20, segment_id="s1"),
                ],
            ),
            (
                SPOT_MARKET_TYPE,
                "depth_snapshot",
                "REST:/api/v3/depth",
                [
                    _snapshot(
                        1,
                        market_type=SPOT_MARKET_TYPE,
                        segment_id="s1",
                        last_update_id=10,
                    )
                ],
            ),
        ],
    )
    with open_recorder_run(root) as run:
        owned_states = [
            event
            for event in iter_reconstructed_depth(
                run, SPOT_MARKET_TYPE, symbol="BTCUSDT", required_levels=1
            )
            if isinstance(event, BookState)
        ]
        states = [event for event in owned_states]
    assert [state.update_id for state in states] == [11, 12]
    assert all(state.transaction_time_ms is None for state in states)
    assert [state.event_time_ms for state in states] == [10, 20]


def test_gap_emits_boundary_and_recovery_never_reuses_old_book(tmp_path: Path) -> None:
    root = tmp_path / "run"
    depth_rows = [
        _futures_delta(1, 90, 100, previous=0, transaction_time=10, segment_id="s1"),
        _futures_delta(2, 105, 110, previous=99, transaction_time=20, segment_id="s1"),
        _futures_delta(3, 190, 200, previous=0, transaction_time=30, segment_id="s2"),
    ]
    _futures_run(root, depth_rows)
    with open_recorder_run(root) as run:
        events = list(
            iter_reconstructed_depth(
                run, FUTURES_MARKET_TYPE, symbol="BTCUSDT", required_levels=1
            )
        )
        assert [
            type(event) if isinstance(event, BookState) else type(event)
            for event in events
        ] == [BookState, BookBoundary, BookState]
        assert events[1].reason == "sequence_gap"
        assert events[1].transaction_time_ms == 20
        assert events[2].segment_id == "s2"


@pytest.mark.parametrize(
    ("bids", "asks", "reason"),
    (
        ([["100.0", "0"], ["99.0", "0"]], [], "one_sided_book"),
        ([["101.0", "1"]], [], "crossed_book"),
    ),
)
def test_invalid_book_emits_honest_boundary(
    tmp_path: Path, bids: list[list[str]], asks: list[list[str]], reason: str
) -> None:
    root = tmp_path / reason
    _futures_run(
        root,
        [
            _futures_delta(
                1,
                90,
                100,
                previous=0,
                transaction_time=10,
                segment_id="s1",
                bids=bids,
                asks=asks,
            )
        ],
    )
    with open_recorder_run(root) as run:
        events = list(
            iter_reconstructed_depth(
                run, FUTURES_MARKET_TYPE, symbol="BTCUSDT", required_levels=1
            )
        )
    assert len(events) == 1
    assert isinstance(events[0], BookBoundary)
    assert events[0].reason == reason


def test_same_run_replay_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _futures_run(
        root,
        [
            _futures_delta(
                1, 90, 100, previous=0, transaction_time=10, segment_id="s1"
            ),
            _futures_delta(
                2,
                101,
                101,
                previous=100,
                transaction_time=20,
                segment_id="s1",
                bids=[["100.0", "1.5"]],
            ),
        ],
    )
    with open_recorder_run(root) as run:
        first = list(
            iter_reconstructed_depth(
                run, FUTURES_MARKET_TYPE, symbol="BTCUSDT", required_levels=1
            )
        )
        second = list(
            iter_reconstructed_depth(
                run, FUTURES_MARKET_TYPE, symbol="BTCUSDT", required_levels=1
            )
        )
        assert [
            event if isinstance(event, BookState) else event for event in first
        ] == [event if isinstance(event, BookState) else event for event in second]


@pytest.mark.parametrize("mode", ("complete", "cancel", "same_owner_overlap"))
def test_hourly_owner_files_merge_transitive_ranges_with_bounded_handles(
    tmp_path, monkeypatch, mode
):
    from quant_research_platform.policy.lifecycle import closing_iterator

    root = tmp_path / "run"

    def at_hour(record, hour):
        record["receive_time"] = f"2026-08-27T0{hour}:00:00.123456Z"
        record["observed_time_ns"] = pd.Timestamp(record["receive_time"]).value
        return record

    def trade(sequence, hour):
        return at_hour(_agg_trade(sequence), hour)

    def depth(sequence, hour):
        return at_hour(
            _futures_delta(
                sequence,
                sequence,
                sequence,
                previous=sequence - 1,
                transaction_time=sequence,
                segment_id="segment-1",
            ),
            hour,
        )

    manifest = _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [trade(1, 0), trade(5, 0)],
            ),
            (
                FUTURES_MARKET_TYPE,
                "depth",
                "btcusdt@depth@100ms",
                [depth(2, 0), depth(3, 0)],
            ),
        ],
    )
    for kind, stream, rows in (
        (
            "agg_trades",
            "btcusdt@aggTrade",
            [trade(4 if mode == "same_owner_overlap" else 6, 1), trade(8, 1)],
        ),
        ("depth", "btcusdt@depth@100ms", [depth(4, 1), depth(7, 1)]),
    ):
        item = _write_file(
            tmp_path / "next_hour",
            market_type=FUTURES_MARKET_TYPE,
            data_kind=kind,
            stream=stream,
            rows=rows,
        )
        original = tmp_path / "next_hour" / item["path"]
        item["path"] = item["path"].replace("hour=00", "hour=01")
        target = root / item["path"]
        filesystem_mkdir(target.parent, parents=True, exist_ok=True)
        filesystem_write_bytes(target, filesystem_read_bytes(original))
        manifest["files"].append(item)
    sidecar = root / "connection_intervals.jsonl.gz"
    with gzip.open(native_filesystem_path(sidecar), "rt", encoding="utf-8") as handle:
        intervals = [json.loads(line) for line in handle]
    with gzip.open(native_filesystem_path(sidecar), "wt", encoding="utf-8") as handle:
        for row in intervals:
            row["closed_at"] = "2026-08-27T01:00:01.000000Z"
            row["closed_monotonic_ns"] = 3601000000000
            handle.write(json.dumps(row) + "\n")
    raw = filesystem_read_bytes(sidecar)
    manifest["capture"]["connection_intervals"].update(
        compressed_bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
    )
    filesystem_write_text(root / "run.json", json.dumps(manifest), encoding="utf-8")
    real_open = replay.open_filesystem_file
    handles = []
    high_water = 0

    def tracked(path, *args, **kwargs):
        nonlocal high_water
        handle = real_open(path, *args, **kwargs)
        if path.suffix == ".gz" and path.name != "connection_intervals.jsonl.gz":
            handles.append(handle)
            high_water = max(high_water, sum(not item.closed for item in handles))
        return handle

    monkeypatch.setattr(replay, "open_filesystem_file", tracked)
    with replay.open_recorder_run(root) as run:
        if mode == "same_owner_overlap":
            with pytest.raises(
                ValueError, match="binance_replay_file_capture_ranges_overlap"
            ):
                list(run.iter_records())
        else:
            assert [
                row["capture_seq"]
                for row in run.iter_records(data_kinds=("agg_trades",))
            ] == [1, 5, 6, 8]
            assert [
                row["capture_seq"] for row in run.iter_records(data_kinds=("depth",))
            ] == [2, 3, 4, 7]
            if mode == "cancel":
                primary = KeyboardInterrupt("consumer_cancel")
                with pytest.raises(KeyboardInterrupt) as caught:
                    with closing_iterator(run.iter_records()) as records:
                        assert next(records)["capture_seq"] == 1
                        raise primary
                assert caught.value is primary
            else:
                assert [row["capture_seq"] for row in run.iter_records()] == list(
                    range(1, 9)
                )
        assert handles and all(handle.closed for handle in handles)
    assert high_water <= 2


@pytest.mark.parametrize("alias_kind", ("root", "ancestor", "member"))
def test_recorder_run_rejects_aliased_input_paths(tmp_path, alias_kind):
    root = tmp_path / "recordings" / "run"
    manifest = _write_run(
        root, [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])]
    )
    input_root = root
    try:
        if alias_kind == "member":
            member = root / manifest["files"][0]["path"]
            outside = tmp_path / "outside.jsonl.gz"
            filesystem_rename(member, outside)
            Path(native_filesystem_path(member)).symlink_to(
                native_filesystem_path(outside)
            )
        else:
            alias = tmp_path / "alias"
            target = root if alias_kind == "root" else root.parent
            Path(native_filesystem_path(alias)).symlink_to(
                native_filesystem_path(target), target_is_directory=True
            )
            input_root = alias if alias_kind == "root" else alias / root.name
    except OSError as error:
        pytest.skip(f"filesystem symlink unavailable: {error}")
    error = (
        "binance_replay_file_path_invalid"
        if alias_kind == "member"
        else "binance_raw_export_identity_changed"
    )
    with pytest.raises(ValueError, match=f"^{error}$"):
        with open_recorder_run(input_root):
            pytest.fail("aliased recorder input admitted")


@pytest.mark.parametrize("consumer_primary", (False, True))
def test_recorder_run_rejects_replaced_root_and_preserves_primary(
    tmp_path, consumer_primary
):
    root = tmp_path / "run"
    _write_run(
        root, [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])]
    )
    original = filesystem_read_bytes(root / "run.json")
    moved = tmp_path / "moved"
    primary = KeyboardInterrupt("consumer_primary")
    with pytest.raises(KeyboardInterrupt if consumer_primary else ValueError) as raised:
        with open_recorder_run(root):
            filesystem_rename(root, moved)
            filesystem_mkdir(root)
            if consumer_primary:
                raise primary
    if consumer_primary:
        assert raised.value is primary
        assert any("ValueError" in note for note in primary.__notes__)
    else:
        assert str(raised.value) == "binance_raw_export_identity_changed"
    assert filesystem_read_bytes(moved / "run.json") == original


def test_recorder_run_rejects_member_replacement_between_capture_and_open(
    tmp_path, monkeypatch
):
    root = tmp_path / "run"
    manifest = _write_run(
        root, [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])]
    )
    member = root / manifest["files"][0]["path"]
    original = filesystem_read_bytes(member)
    original_open = replay.open_filesystem_file
    mutated = []

    def mutate_before_open(path, *args, **kwargs):
        if path == member and (not mutated):
            before = filesystem_stat(member)
            os.utime(
                native_filesystem_path(member),
                ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000),
            )
            mutated.append(True)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(replay, "open_filesystem_file", mutate_before_open)
    with pytest.raises(ValueError, match="^binance_raw_export_identity_changed$"):
        with open_recorder_run(root):
            pytest.fail("replaced recorder member admitted")
    assert mutated == [True]
    assert filesystem_read_bytes(member) == original


@pytest.mark.parametrize("consumer_primary", (False, True))
def test_recorder_run_checks_consumed_handle_identity_and_preserves_primary(
    tmp_path, consumer_primary
):
    from quant_research_platform.policy.lifecycle import closing_iterator

    root = tmp_path / "run"
    manifest = _write_run(
        root, [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])]
    )
    member = root / manifest["files"][0]["path"]
    original = filesystem_read_bytes(member)
    primary = KeyboardInterrupt("consumer_primary")
    with pytest.raises(KeyboardInterrupt if consumer_primary else ValueError) as raised:
        with open_recorder_run(root) as run:
            with closing_iterator(run.iter_records()) as records:
                assert next(records)["capture_seq"] == 1
                before = filesystem_stat(member)
                os.utime(
                    native_filesystem_path(member),
                    ns=(before.st_atime_ns, before.st_mtime_ns + 2000000000),
                )
                assert filesystem_stat(member).st_mtime_ns != before.st_mtime_ns
                if consumer_primary:
                    raise primary
                list(records)
    if consumer_primary:
        assert raised.value is primary
        assert any("ValueError" in note for note in primary.__notes__)
    else:
        assert str(raised.value) == "binance_raw_export_identity_changed"
    assert filesystem_read_bytes(member) == original


def test_connection_queries_scan_each_interval_once(tmp_path):
    import gzip
    import hashlib
    import json
    from datetime import datetime, timedelta

    root = tmp_path / "run"
    manifest = _write_run(
        root, [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])]
    )
    sidecar = root / "connection_intervals.jsonl.gz"
    with gzip.open(native_filesystem_path(sidecar), "rt", encoding="utf-8") as stream:
        original = json.loads(next(stream))
    start = datetime(2020, 1, 1, tzinfo=UTC)
    with gzip.open(native_filesystem_path(sidecar), "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(original) + "\n")
        for index in range(2000):
            opened = start + timedelta(seconds=2 * index)
            row = dict(
                original,
                connection_id=f"history-{index}",
                opened_at=opened.isoformat(),
                closed_at=(opened + timedelta(seconds=1)).isoformat(),
            )
            stream.write(json.dumps(row) + "\n")
    with open_filesystem_file(sidecar, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest["capture"]["connection_intervals"].update(
        record_count=2001,
        compressed_bytes=filesystem_stat(sidecar).st_size,
        sha256=digest,
    )
    filesystem_write_text(root / "run.json", json.dumps(manifest), encoding="utf-8")
    with open_recorder_run(root) as run:
        coverage = run.observation_coverage(
            market_type=FUTURES_MARKET_TYPE, data_kind="agg_trades", symbol="BTCUSDT"
        )
        first = int(pd.Timestamp(start).value)
        assert coverage.covering(first, first + 999999999) == ("history-0",)
        assert coverage.covering(first, first + 1000000000) == ()
        assert coverage["history-1"] == (first + 2000000000, first + 3000000000)
        assert coverage.get("absent") is None
        with pytest.raises(ValueError, match="binance_coverage_window_non_monotonic"):
            coverage.covering(first - 1, first)
        sweep = run.observation_coverage(
            market_type=FUTURES_MARKET_TYPE, data_kind="agg_trades", symbol="BTCUSDT"
        )

        class CountedRows:
            def __init__(self, rows):
                self.rows = rows
                self.reads = 0

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, position):
                self.reads += 1
                return self.rows[position]

        rows = CountedRows(sweep.rows)
        sweep.rows = rows
        for index in range(100):
            sweep.covering(first + index * 100000000, first + (index + 1) * 100000000)
        assert sweep.position == 5
        assert rows.reads <= 110


def test_live_depth_advance_preserves_unknown_future_coverage_and_validity():
    stats = replay.ReconstructionStats()
    snapshot = dict(
        _raw(
            1,
            {"lastUpdateId": 10, "bids": [["100", "1"]], "asks": [["101", "1"]]},
            segment_id="segment-1",
        ),
        market_type=FUTURES_MARKET_TYPE,
        tick_size="1",
    )
    update = dict(
        _raw(
            2,
            {"U": 10, "u": 10, "pu": 9, "E": 1, "T": 1, "b": [["100", "2"]], "a": []},
            segment_id="segment-1",
        ),
        market_type=FUTURES_MARKET_TYPE,
        tick_size="1",
    )
    book = replay.DepthReconstructor(
        FUTURES_MARKET_TYPE,
        price_unit="1",
        snapshot_limit=1,
        required_levels=1,
        connection_close_ns=None,
        stats=stats,
    )
    book.bootstrap(replay.normalize_recorder_record(snapshot))
    assert book.advance(replay.normalize_recorder_record(update)) is True
    assert book.valid
    assert stats.coverage_segments == {}
    assert stats.emitted_states == 0
    depleted = dict(
        update,
        capture_seq=3,
        payload={
            "U": 11,
            "u": 11,
            "pu": 10,
            "E": 2,
            "T": 2,
            "b": [["100", "0"]],
            "a": [],
        },
    )
    boundary = book.advance(replay.normalize_recorder_record(depleted))
    assert isinstance(boundary, BookBoundary)
    assert boundary.reason == "one_sided_book"
    assert boundary.source_sequence == 3
    assert not book.valid
    assert stats.coverage_segments == {}


def test_default_multisymbol_reader_merges_same_named_rest_streams(tmp_path):
    root = tmp_path / "combined"
    manifests = []
    for symbol, offset in (("BTCUSDT", 0), ("ETHUSDT", 1)):
        source = root if offset == 0 else tmp_path / "eth"
        snapshots, deltas = ([], [])
        for sequence in (1 + offset, 5 + offset):
            segment = f"{symbol}-{sequence}"
            connection = f"connection-{segment}"
            snapshots.append(
                _snapshot(
                    sequence,
                    market_type=FUTURES_MARKET_TYPE,
                    segment_id=segment,
                    last_update_id=100,
                    connection_id=connection,
                )
            )
            delta = _futures_delta(
                sequence + 2,
                100,
                101,
                previous=99,
                transaction_time=101,
                segment_id=segment,
                connection_id=connection,
            )
            delta["payload"]["s"] = symbol
            delta["payload"]["ps"] = symbol
            deltas.append(delta)
        manifest = _write_run(
            source,
            [
                (
                    FUTURES_MARKET_TYPE,
                    "depth_snapshot",
                    "REST:/fapi/v1/depth",
                    snapshots,
                ),
                (FUTURES_MARKET_TYPE, "depth", symbol.lower() + "@depth@100ms", deltas),
            ],
        )
        for item in manifest["files"]:
            item["symbol"] = symbol
            if offset:
                original = source / item["path"]
                item["path"] = "eth/" + item["path"]
                target = root / item["path"]
                filesystem_mkdir(target.parent, parents=True, exist_ok=True)
                filesystem_write_bytes(target, filesystem_read_bytes(original))
        manifests.append(manifest)
    intervals = []
    for source in (root, tmp_path / "eth"):
        with gzip.open(
            native_filesystem_path(source / "connection_intervals.jsonl.gz"), "rt"
        ) as handle:
            intervals.extend(json.loads(line) for line in handle)
    sidecar = root / "connection_intervals.jsonl.gz"
    with gzip.open(native_filesystem_path(sidecar), "wt") as handle:
        for row in intervals:
            handle.write(json.dumps(row) + "\n")
    manifest = manifests[0]
    manifest["files"].extend(manifests[1]["files"])
    payload = filesystem_read_bytes(sidecar)
    manifest["capture"]["connection_intervals"].update(
        record_count=len(intervals),
        compressed_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    filesystem_write_text(root / "run.json", json.dumps(manifest), encoding="utf-8")
    with open_recorder_run(root) as run:
        assert [row["capture_seq"] for row in run.iter_records()] == list(range(1, 9))
        assert [row["capture_seq"] for row in run.iter_records(symbol="BTCUSDT")] == [
            1,
            3,
            5,
            7,
        ]
        assert [row["capture_seq"] for row in run.iter_records(symbol="ETHUSDT")] == [
            2,
            4,
            6,
            8,
        ]


@pytest.mark.parametrize("kind", ["depth", "agg_trades"])
@pytest.mark.parametrize(
    "wrong_stream", ["other_symbol", "other_kind", "other_frequency"]
)
def test_native_connection_must_match_record_symbol_and_stream(
    tmp_path, kind, wrong_stream
):
    root = tmp_path / "run"
    if kind == "depth":
        rows = [
            (
                FUTURES_MARKET_TYPE,
                "depth_snapshot",
                "https://fapi.binance.com/fapi/v1/depth",
                [
                    _snapshot(
                        1,
                        market_type=FUTURES_MARKET_TYPE,
                        segment_id="s",
                        last_update_id=100,
                    )
                ],
            ),
            (
                FUTURES_MARKET_TYPE,
                "depth",
                "btcusdt@depth@100ms",
                [
                    _futures_delta(
                        2, 100, 101, previous=99, transaction_time=100, segment_id="s"
                    )
                ],
            ),
        ]
    else:
        rows = [
            (FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])
        ]
    manifest = _write_run(root, rows)
    with open_recorder_run(root) as run:
        if kind == "depth":
            count = 0
            for event in iter_reconstructed_depth(
                run, FUTURES_MARKET_TYPE, symbol="BTCUSDT", required_levels=2
            ):
                assert isinstance(event, BookState)
                count += 1
            assert count == 1
        else:
            assert (
                len(list(iter_agg_trades(run, FUTURES_MARKET_TYPE, symbol="BTCUSDT")))
                == 1
            )
    sidecar = root / "connection_intervals.jsonl.gz"
    with gzip.open(native_filesystem_path(sidecar), "rt") as handle:
        intervals = [json.loads(line) for line in handle]
    interval = intervals[0]
    interval["stream"] = {
        "other_symbol": interval["stream"].replace("btcusdt", "ethusdt"),
        "other_kind": "btcusdt@aggTrade" if kind == "depth" else "btcusdt@depth@100ms",
        "other_frequency": "btcusdt@depth@0ms",
    }[wrong_stream]
    with gzip.open(native_filesystem_path(sidecar), "wt") as handle:
        for interval in intervals:
            handle.write(json.dumps(interval) + "\n")
    member = manifest["capture"]["connection_intervals"]
    member["compressed_bytes"] = filesystem_stat(sidecar).st_size
    member["sha256"] = hashlib.sha256(filesystem_read_bytes(sidecar)).hexdigest()
    filesystem_write_text(root / "run.json", json.dumps(manifest), encoding="utf-8")
    error = (
        "binance_replay_connection_coverage_invalid"
        if wrong_stream == "other_symbol"
        else "binance_replay_connection_intervals_invalid"
    )
    with pytest.raises(ValueError, match=error):
        with open_recorder_run(root) as run:
            if kind == "depth":
                list(
                    iter_reconstructed_depth(
                        run, FUTURES_MARKET_TYPE, symbol="BTCUSDT", required_levels=2
                    )
                )
            else:
                list(iter_agg_trades(run, FUTURES_MARKET_TYPE, symbol="BTCUSDT"))


@pytest.mark.parametrize(
    "stream", ["ethusdt@depth@100ms", "btcusdt@aggTrade", "btcusdt@depth@0ms"]
)
def test_native_file_stream_matches_symbol_and_kind(tmp_path, stream):
    root = tmp_path / "run"
    manifest = _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "depth",
                "btcusdt@depth@100ms",
                [
                    _futures_delta(
                        1, 100, 101, previous=99, transaction_time=100, segment_id="s"
                    )
                ],
            )
        ],
    )
    manifest["files"][0]["stream"] = stream
    filesystem_write_text(root / "run.json", json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="binance_replay_file_manifest_invalid"):
        with open_recorder_run(root):
            pass


@pytest.mark.parametrize(
    "boundary",
    (
        "root_stat",
        "manifest_stat",
        "manifest_open",
        "coverage_stat",
        "coverage_open",
        "member_open",
        "member_fstat",
        "scandir",
        "raw_open",
    ),
)
def test_recorder_file_boundaries_preserve_serious_io(tmp_path, monkeypatch, boundary):
    import errno

    root = tmp_path / "run"
    manifest = _write_run(
        root, [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])]
    )
    member = root / manifest["files"][0]["path"]
    target = {
        "root_stat": root,
        "manifest_stat": root / "run.json",
        "manifest_open": root / "run.json",
        "coverage_stat": root / "connection_intervals.jsonl.gz",
        "coverage_open": root / "connection_intervals.jsonl.gz",
    }.get(boundary, member)
    primary = OSError(errno.EIO, "controlled_io_failure")
    owner = (
        os
        if boundary == "member_fstat"
        else storage_manifests
        if boundary == "manifest_open"
        else replay
    )
    operation = (
        "fstat"
        if boundary == "member_fstat"
        else "filesystem_scandir"
        if boundary == "scandir"
        else "filesystem_stat"
        if boundary.endswith("stat")
        else "open_filesystem_file"
    )
    original = getattr(owner, operation)

    def fail(candidate, *args, **kwargs):
        selected = (
            Path(os.readlink(f"/proc/self/fd/{candidate}")) == member
            if boundary == "member_fstat"
            else Path(candidate) == (root if boundary == "scandir" else target)
        )
        if selected:
            raise primary
        return original(candidate, *args, **kwargs)

    if boundary == "member_fstat" and (not Path("/proc/self/fd").exists()):
        pytest.skip("POSIX file descriptor path required for targeted fstat fault")
    if boundary == "raw_open":
        with open_recorder_run(root) as run:
            with monkeypatch.context() as patch:
                patch.setattr(owner, operation, fail)
                with pytest.raises(OSError) as caught:
                    next(run.iter_records())
    else:
        with monkeypatch.context() as patch:
            patch.setattr(owner, operation, fail)
            with pytest.raises(OSError) as caught:
                with open_recorder_run(root):
                    pytest.fail("fatal file error was accepted")
    assert caught.value is primary


@pytest.mark.parametrize("replace_parent", (False, True))
def test_raw_identity_rejects_disappearance_after_initial_stat(
    tmp_path, monkeypatch, replace_parent
):
    parent = tmp_path / "parent"
    filesystem_mkdir(parent)
    path = parent / "member"
    filesystem_write_bytes(path, b"synthetic")
    capture = replay.dataset_file_identity

    def remove_before_capture(candidate):
        assert candidate == path
        filesystem_unlink(path)
        if replace_parent:
            os.rmdir(native_filesystem_path(parent))
            filesystem_write_bytes(parent, b"synthetic")
        return capture(candidate)

    monkeypatch.setattr(replay, "dataset_file_identity", remove_before_capture)
    with pytest.raises(ValueError, match="^binance_raw_export_identity_changed$"):
        replay._capture_raw_file_identity(path)


@pytest.mark.parametrize("resend", ["same", "event_clock", "conflict", "unseen"])
def test_native_trade_identity_is_exact_across_official_resends(tmp_path, resend):
    first, duplicate = (_agg_trade(1), _agg_trade(2))
    duplicate["payload"] = dict(first["payload"])
    if resend == "event_clock":
        duplicate["payload"]["E"] += 1
    elif resend == "conflict":
        duplicate["payload"]["q"] = "3"
    elif resend == "unseen":
        duplicate["payload"]["a"] -= 1
    root = tmp_path / "run"
    _write_run(
        root,
        [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [first, duplicate])],
    )
    with open_recorder_run(root) as run:
        if resend in {"conflict", "unseen"}:
            with pytest.raises(ValueError, match="sequence_non_monotonic"):
                list(iter_agg_trades(run, FUTURES_MARKET_TYPE, symbol="BTCUSDT"))
        else:
            result = list(iter_agg_trades(run, FUTURES_MARKET_TYPE, symbol="BTCUSDT"))
            assert [(row.quantity, row.visible_quantity) for row in result] == [
                (2.0, 1.0)
            ]


def _write_coverage_history(root, count, width):
    filesystem_mkdir(root, parents=True, exist_ok=True)
    source = root / "connection_intervals.jsonl.gz"
    with gzip.open(native_filesystem_path(source), "wt", encoding="utf-8") as handle:
        for index in range(count):
            opened = pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(
                seconds=2 * index
            )
            handle.write(
                json.dumps(
                    {
                        "connection_id": f"{index:0{width}x}",
                        "market": FUTURES_MARKET_TYPE,
                        "data_kind": "agg_trades",
                        "stream": "btcusdt@aggTrade",
                        "segment_id": None,
                        "opened_at": opened.isoformat(),
                        "closed_at": (opened + pd.Timedelta(seconds=1)).isoformat(),
                        "opened_monotonic_ns": 2 * index * 10**9,
                        "closed_monotonic_ns": (2 * index + 1) * 10**9,
                        "close_reason": "stop",
                    }
                )
                + "\n"
            )
    return {
        "path": source.name,
        "compressed_bytes": filesystem_stat(source).st_size,
        "record_count": count,
        "sha256": hashlib.sha256(filesystem_read_bytes(source)).hexdigest(),
    }


@pytest.mark.parametrize(
    "bounds,expected",
    [
        ([(0, 0)], []),
        ([(0, 30)], [(0, 30)]),
        ([(0, 30), (10, 10)], [(0, 30)]),
        ([(0, 30), (0, 0), (30, 30)], [(0, 30)]),
        ([(0, 10), (10, 20)], [(0, 10), (10, 20)]),
        ([(0, 10), (10, 10), (10, 20)], [(0, 10), (10, 20)]),
        ([(0, 20), (10, 30)], [(0, 10), (20, 30)]),
        ([(0, 20), (10, 30), (15, 15)], [(0, 10), (20, 30)]),
    ],
)
def test_coverage_empty_connections_keep_facts_without_sweep_contribution(
    tmp_path, bounds, expected
):
    descriptor = _write_coverage_history(tmp_path, 1, 32)
    source = tmp_path / descriptor["path"]
    with gzip.open(native_filesystem_path(source), "rt", encoding="utf-8") as handle:
        original = json.loads(next(handle))
    start = pd.Timestamp("2020-01-01", tz="UTC")
    with gzip.open(native_filesystem_path(source), "wt", encoding="utf-8") as handle:
        for position, (opened, closed) in enumerate(bounds):
            handle.write(
                json.dumps(
                    {
                        **original,
                        "connection_id": f"connection-{position}",
                        "opened_at": (start + pd.Timedelta(seconds=opened)).isoformat(),
                        "closed_at": (start + pd.Timedelta(seconds=closed)).isoformat(),
                        "opened_monotonic_ns": opened * 10**9,
                        "closed_monotonic_ns": closed * 10**9,
                    }
                )
                + "\n"
            )
    raw_hash = hashlib.sha256(filesystem_read_bytes(source)).hexdigest()
    descriptor.update(
        record_count=len(bounds),
        compressed_bytes=filesystem_stat(source).st_size,
        sha256=raw_hash,
    )
    index = replay.ObservationCoverageIndex(
        tmp_path,
        descriptor,
        markets={FUTURES_MARKET_TYPE},
        decode_interval=replay._decode_recorder_interval,
    )
    reader = replay.ObservationCoverage(
        index, FUTURES_MARKET_TYPE, "agg_trades", "BTCUSDT"
    )
    assert list(reader.intervals()) == [
        (start.value + opened * 10**9, start.value + closed * 10**9)
        for opened, closed in expected
    ]
    for position, (opened, closed) in enumerate(bounds):
        assert reader[f"connection-{position}"] == (
            start.value + opened * 10**9,
            start.value + closed * 10**9,
        )
    assert hashlib.sha256(filesystem_read_bytes(source)).hexdigest() == raw_hash


@pytest.mark.parametrize("change", ("grow", "truncate", "replace"))
def test_manifest_race_is_bounded_and_rejected_before_decode(
    tmp_path, monkeypatch, change
):
    from quant_research_platform.sources import open_market_source
    from quant_research_platform.storage.identity import filesystem_replace

    root = tmp_path / "run"
    _write_run(
        root, [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])]
    )
    path = root / "run.json"
    original = filesystem_read_bytes(path)
    original_open = storage_manifests.open_filesystem_file
    reads, handles, decodes = ([], [], [])

    class ObservedReader:
        def __init__(self, handle):
            self.handle = handle

        def fileno(self):
            return self.handle.fileno()

        def read(self, size=-1):
            assert size == len(original) + 1
            filesystem_write_bytes(
                path,
                original + b" " * (2 * 1024**2)
                if change == "grow"
                else original[: len(original) // 2],
            )
            result = self.handle.read(size)
            reads.append((size, len(result)))
            return result

        def close(self):
            self.handle.close()

    def open_changed(candidate, *args, **kwargs):
        assert candidate == path
        if change == "replace":
            replacement = root / "replacement.json"
            filesystem_write_bytes(replacement, original.replace(b"run-1", b"run-2"))
            filesystem_replace(replacement, path)
        reader = ObservedReader(original_open(candidate, *args, **kwargs))
        handles.append(reader.handle)
        return reader

    def forbidden_decode(raw):
        decodes.append(len(raw))
        pytest.fail("changed manifest reached JSON decoding")

    monkeypatch.setattr(storage_manifests, "open_filesystem_file", open_changed)
    monkeypatch.setattr(storage_manifests, "parse_qrp_json_object", forbidden_decode)
    with pytest.raises(ValueError, match="^binance_replay_run_identity_changed$"):
        with open_market_source(root):
            pytest.fail("changed manifest reached the source consumer")
    assert not decodes
    assert handles and all(handle.closed for handle in handles)
    assert reads == (
        []
        if change == "replace"
        else [
            (
                len(original) + 1,
                len(original) + 1 if change == "grow" else len(original) // 2,
            )
        ]
    )


def test_manifest_read_failure_preserves_primary_and_closes_handle(
    tmp_path, monkeypatch
):
    import errno

    root = tmp_path / "run"
    _write_run(
        root, [(FUTURES_MARKET_TYPE, "agg_trades", "btcusdt@aggTrade", [_agg_trade(1)])]
    )
    path = root / "run.json"
    original = filesystem_read_bytes(path)
    original_open = storage_manifests.open_filesystem_file
    primary = OSError(errno.EIO, "manifest_read_primary")
    handles = []

    class FailedReader:
        def __init__(self, handle):
            self.handle = handle

        def fileno(self):
            return self.handle.fileno()

        def read(self, size=-1):
            assert size == len(original) + 1
            filesystem_write_bytes(path, original + b" ")
            raise primary

        def close(self):
            self.handle.close()
            raise OSError("manifest_close_secondary")

    def open_failed(candidate, *args, **kwargs):
        reader = FailedReader(original_open(candidate, *args, **kwargs))
        handles.append(reader.handle)
        return reader

    monkeypatch.setattr(storage_manifests, "open_filesystem_file", open_failed)
    with pytest.raises(OSError) as caught:
        with open_recorder_run(root):
            pytest.fail("failed read reached the consumer")
    assert caught.value is primary
    assert handles and all(handle.closed for handle in handles)
    assert primary.__notes__ == [
        "file identity verification also failed: ValueError",
        "iterator close also failed: OSError",
    ]


def test_manifest_growth_blocks_materialization_and_preserves_old_good(
    tmp_path, monkeypatch
):
    from quant_research_platform.markets.implementations.crypto.ids import (
        HFT_PANEL_DATA_ID,
    )
    from quant_research_platform.profiles.research import (
        CRYPTO_PROFILE_ID,
        research_profile,
    )
    from quant_research_platform.storage.identity import filesystem_walk
    from quant_research_platform.storage.layout import StorageLayout
    from quant_research_platform.workflows import research
    from tests.helpers.binance_hft import write_binance_hft_run

    raw_root = tmp_path / "raw"
    write_binance_hft_run(raw_root)
    layout = StorageLayout(tmp_path / "project")
    profile = research_profile(CRYPTO_PROFILE_ID)
    kwargs = dict(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        data_ids=(HFT_PANEL_DATA_ID,),
        start_date="1970-01-01T00:00:00",
        end_date="1970-01-01T00:00:06",
    )
    result = research.data_materialize(**kwargs)
    assert result["status"] == "COMPLETE"
    spec = profile.processed.get(HFT_PANEL_DATA_ID)
    instances = layout.processed_data_root(spec.market_id, spec.kind, spec.data_id)

    def published_bytes():
        return {
            (parent / name).relative_to(instances): filesystem_read_bytes(parent / name)
            for parent, _directories, files in filesystem_walk(instances)
            for name in files
        }

    old_good = published_bytes()
    assert old_good
    manifest_path = raw_root / "run.json"
    original = filesystem_read_bytes(manifest_path)
    read_manifest = replay.read_bounded_qrp_json
    attempts = []

    def grow_after_admission(path, **options):
        if path == manifest_path:
            attempts.append(options["max_bytes"])
            filesystem_write_bytes(path, filesystem_read_bytes(path) + b" ")
        return read_manifest(path, **options)

    monkeypatch.setattr(replay, "read_bounded_qrp_json", grow_after_admission)
    rejected = research.data_materialize(**kwargs, rebuild=True)
    assert rejected["status"] == "BLOCKED"
    assert rejected["target_results"][0]["reason"] == "source_identity_invalid"
    assert attempts and attempts[0] == len(original)
    assert published_bytes() == old_good


def test_same_utc_book_invalidation_drops_empty_coverage():
    stats = replay.ReconstructionStats()
    snapshot = dict(
        _raw(
            1,
            {"lastUpdateId": 10, "bids": [["100", "1"]], "asks": [["101", "1"]]},
            segment_id="s",
        ),
        market_type=FUTURES_MARKET_TYPE,
        tick_size="1",
    )

    def update(sequence, identifier, bids):
        return replay.normalize_recorder_record(
            dict(
                _raw(
                    sequence,
                    {
                        "U": identifier,
                        "u": identifier,
                        "pu": identifier - 1,
                        "E": 1,
                        "T": 1,
                        "b": bids,
                        "a": [],
                    },
                    segment_id="s",
                ),
                market_type=FUTURES_MARKET_TYPE,
                tick_size="1",
            )
        )

    engine = replay.DepthReconstructor(
        FUTURES_MARKET_TYPE,
        price_unit="1",
        snapshot_limit=1,
        required_levels=1,
        connection_close_ns=snapshot["observed_time_ns"] + 100,
        stats=stats,
    )
    engine.bootstrap(replay.normalize_recorder_record(snapshot))
    first = list(engine.feed(update(2, 10, [["100", "2"]])))
    assert len(first) == 1 and isinstance(first[0], BookState)
    boundary = list(engine.feed(update(3, 11, [["102", "1"]])))
    assert len(boundary) == 1 and isinstance(boundary[0], BookBoundary)
    assert stats.crossed_books == 1
    assert stats.coverage_segments == {}
    decoded = replay.ReconstructionStats.from_payload(
        json.loads(json.dumps(__import__("dataclasses").asdict(stats)))
    )
    assert decoded.coverage_segments == {} and decoded.crossed_books == 1


def test_continuous_book_growth_inside_known_snapshot_range():
    snapshot = {
        "market_type": FUTURES_MARKET_TYPE,
        "segment_id": "segment",
        "observation_id": "observation",
        "payload": {
            "lastUpdateId": 100,
            "bids": [[str(price), "1"] for price in range(20000, 0, -20)],
            "asks": [[str(price), "1"] for price in range(20001, 21001)],
        },
    }
    engine = replay.DepthReconstructor(
        FUTURES_MARKET_TYPE,
        price_unit="1",
        snapshot_limit=1000,
        required_levels=10,
        connection_close_ns=1_800_000_000_000_000_000,
        stats=replay.ReconstructionStats(),
        output_levels=10,
    )
    engine.bootstrap(snapshot)

    def delta(sequence, prices):
        return {
            "market_type": FUTURES_MARKET_TYPE,
            "segment_id": "segment",
            "observation_id": "observation",
            "source_sequence": sequence,
            "receive_time": f"2026-08-01T00:00:0{sequence}+00:00",
            "receive_monotonic_ns": sequence * 10**9,
            "payload": {
                "U": 99 + sequence,
                "u": 100 + sequence,
                "pu": 99 + sequence,
                "E": sequence * 1000,
                "T": sequence * 1000,
                "b": [[str(price), "1"] for price in prices],
                "a": [],
            },
        }

    assert engine.advance(delta(1, [])) is True
    prices = [price for price in range(20, 20001) if price % 20][:10001]
    states = list(engine.feed(delta(2, prices)))
    assert len(states) == 1 and isinstance(states[0], BookState)
    assert engine.valid
    assert len(prices) == 10001
    assert len(engine._bids) == 11001
    assert engine.stats.invalid_segments == 0


@pytest.mark.parametrize("corruption", ["duplicate_id", "count", "hash"])
def test_coverage_index_rejects_invalid_complete_source(tmp_path, corruption):
    descriptor = _write_coverage_history(tmp_path, 1, 32)
    path = tmp_path / descriptor["path"]
    if corruption == "duplicate_id":
        with gzip.open(path, "rb") as handle:
            row = handle.read()
        with gzip.open(path, "wb") as handle:
            handle.write(row + row)
        raw = path.read_bytes()
        descriptor.update(
            record_count=2,
            compressed_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
    elif corruption == "count":
        descriptor["record_count"] = 2
    else:
        descriptor["sha256"] = "0" * 64
    before = path.read_bytes()
    with pytest.raises(
        ValueError, match="^binance_replay_connection_intervals_invalid$"
    ):
        replay.ObservationCoverageIndex(
            tmp_path,
            descriptor,
            markets={FUTURES_MARKET_TYPE},
            decode_interval=replay._decode_recorder_interval,
        )
    assert path.read_bytes() == before
