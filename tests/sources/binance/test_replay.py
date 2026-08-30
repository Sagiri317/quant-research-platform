from __future__ import annotations

import gzip
import hashlib
import io
import json
import tempfile
import weakref
from pathlib import Path

import pandas as pd
import pytest

import quant_research_platform.sources.binance.replay as replay
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeEvent,
    price_to_tick,
    tick_to_price,
)
from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance.ids import (
    FUTURES_HFT_RECORDING_SCOPE,
    FUTURES_MARKET_TYPE,
    SPOT_L2_RECORDING_SCOPE,
    SPOT_MARKET_TYPE,
)
from quant_research_platform.sources.binance.replay import (
    iter_agg_trades,
    iter_reconstructed_depth,
    open_recorder_run,
)
from tests.helpers.runtime_resources import capture_test_policy


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


def test_connection_coverage_cleanup_preserves_active_primary() -> None:
    class FailingDatabase:
        def close(self) -> None:
            raise OSError("database_close")

    class FailingScratch:
        def cleanup(self) -> None:
            raise OSError("scratch_cleanup")

    index = object.__new__(replay._ConnectionCoverageIndex)
    index._database = FailingDatabase()
    index._scratch = FailingScratch()

    with pytest.raises(KeyboardInterrupt, match="body_primary") as raised:
        try:
            raise KeyboardInterrupt("body_primary")
        finally:
            index.close()

    assert raised.value.__notes__ == [
        "Binance connection coverage database close also failed: OSError",
        "Binance connection coverage scratch cleanup also failed: OSError",
    ]


def test_connection_coverage_gzip_constructor_failure_closes_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = KeyboardInterrupt("gzip_constructor_primary")
    raw_handle = io.BytesIO(b"not-consumed")

    class FakePath:
        def open(self, _mode: str):  # noqa: ANN202
            return raw_handle

    def fail_gzip(*_args, **_kwargs):  # noqa: ANN202
        raise primary

    monkeypatch.setattr(replay.gzip, "GzipFile", fail_gzip)
    index = object.__new__(replay._ConnectionCoverageIndex)
    index._path = FakePath()
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    try:
        with runtime_resource_policy_scope(policy):
            with pytest.raises(KeyboardInterrupt) as raised:
                index._load(markets={FUTURES_MARKET_TYPE}, resource_policy=policy)
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

    class FakePath:
        def __init__(self, handle: RawHandle) -> None:
            self.handle = handle

        def open(self, _mode: str) -> RawHandle:
            return self.handle

    class FakeDatabase:
        def commit(self) -> None:
            return None

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
        replay,
        "_verify_raw_file_handle",
        lambda *_args, **_kwargs: None,
    )
    index = object.__new__(replay._ConnectionCoverageIndex)
    index._path = FakePath(raw_handle)
    index.identity = object()
    index._descriptor = {
        "record_count": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }
    index._database = FakeDatabase()
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    try:
        with runtime_resource_policy_scope(policy):
            with pytest.raises(OSError, match="gzip-close-secondary"):
                index._load(
                    markets={FUTURES_MARKET_TYPE},
                    resource_policy=policy,
                )
        assert raw_handle.close_calls == 1
    finally:
        raw_handle.close()


def test_recorder_run_close_failure_preserves_body_primary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1)],
            )
        ],
    )

    real_close = replay._ConnectionCoverageIndex.close

    def fail_close(index: object) -> None:
        real_close(index)
        raise OSError("coverage-close-secondary")

    monkeypatch.setattr(replay._ConnectionCoverageIndex, "close", fail_close)
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy):
        with pytest.raises(KeyboardInterrupt, match="body-primary"):
            with open_recorder_run(root, policy):
                raise KeyboardInterrupt("body-primary")


def test_replay_file_handle_cleanup_preserves_hash_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingHandle:
        def close(self) -> None:
            raise OSError("handle_close")

    class FakePath:
        def open(self, _mode: str) -> FailingHandle:
            return FailingHandle()

    monkeypatch.setattr(
        replay,
        "_verify_raw_file_handle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            KeyboardInterrupt("hash_primary")
        ),
    )
    policy = capture_test_policy(
        available_memory_bytes=32 * 1024**2,
        explicit_cap_bytes=16 * 1024**2,
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(KeyboardInterrupt, match="hash_primary") as raised:
            replay._sha256_file(FakePath(), object(), policy)

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
    capture_seq: int,
    *,
    quantity: str = "2.0",
    visible_quantity: str = "1.0",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    max_record_bytes = 0
    with gzip.open(path, "wb") as handle:
        for row in rows:
            line = (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            max_record_bytes = max(max_record_bytes, len(line))
            handle.write(line)
    raw = path.read_bytes()
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
    root.mkdir(parents=True, exist_ok=True)
    file_manifests = [
        _write_file(
            root,
            market_type=market,
            data_kind=kind,
            stream=stream,
            rows=rows,
        )
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
                    "segment_id": (str(row["segment_id"]) if kind == "depth" else None),
                    "opened_at": "2026-08-27T00:00:00.000000Z",
                    "opened_monotonic_ns": 0,
                    "closed_at": "2026-08-27T00:00:01.000000Z",
                    "closed_monotonic_ns": 1_000_000_000,
                    "close_reason": "stop",
                }
            )
    sidecar = root / "connection_intervals.jsonl.gz"
    with gzip.open(sidecar, "wt", encoding="utf-8") as handle:
        for interval in connection_intervals:
            handle.write(json.dumps(interval, separators=(",", ":")))
            handle.write("\n")
    sidecar_bytes = sidecar.read_bytes()
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
        "status": "PASS",
        "market_data_status": "PASS",
        "venue": "binance",
        "symbol": "BTCUSDT",
        "market_types": market_types,
        "market_metadata": {
            market: {"tick_size": "0.1", "snapshot_limit": 1000} for market, *_ in files
        },
        "capture": {
            "connection_intervals": {
                "path": "connection_intervals.jsonl.gz",
                "record_count": len(connection_intervals),
                "compressed_bytes": len(sidecar_bytes),
                "sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            },
        },
        "files": file_manifests,
    }
    (root / "run.json").write_text(
        json.dumps(manifest, separators=(",", ":")),
        encoding="utf-8",
    )
    return manifest


def test_recorder_run_retains_manifest_lease_through_final_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1)],
            )
        ],
    )
    retained_bound = (
        replay.json_record_working_set_bytes((root / "run.json").stat().st_size)
        + 64 * 1024
    )
    observed: list[int] = []
    capture_run_files = replay._capture_run_files

    def observe_manifest_owner(*args, **kwargs):  # noqa: ANN202
        observed.append(args[2].used_memory_bytes)
        return capture_run_files(*args, **kwargs)

    monkeypatch.setattr(replay, "_capture_run_files", observe_manifest_owner)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=16 * 1024**2,
    )

    with runtime_resource_policy_scope(policy):
        with open_recorder_run(root, policy):
            pass

    assert len(observed) == 2
    assert observed[0] >= retained_bound
    assert observed[1] >= 2 * retained_bound


def test_successful_run_releases_manifests_inside_owner_leases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1)],
            )
        ],
    )
    released: list[int] = []
    parse_json = replay.parse_qrp_json_object
    canonical_hash = replay.canonical_json_hash
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=16 * 1024**2,
    )

    class TrackedManifest(dict):
        pass

    def track_manifest(raw: bytes):
        value = parse_json(raw)
        if value.get("schema_id") != "qrp-binance-recorder":
            return value
        tracked = TrackedManifest(value)
        weakref.finalize(tracked, lambda: released.append(policy.used_memory_bytes))
        return tracked

    monkeypatch.setattr(replay, "parse_qrp_json_object", track_manifest)
    monkeypatch.setattr(
        replay,
        "canonical_json_hash",
        lambda value: canonical_hash(dict(value)),
    )
    with runtime_resource_policy_scope(policy):
        with open_recorder_run(root, policy) as run:
            assert run.manifest["run_id"] == "run-1"
        assert run.manifest == {}

    assert len(released) == 2
    assert all(used > 0 for used in released)


def test_connection_interval_decode_holds_max_record_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1)],
            )
        ],
    )
    expected_bound = (
        replay.json_record_working_set_bytes(
            replay._CONNECTION_INTERVAL_MAX_RECORD_BYTES
        )
        + 64 * 1024
    )
    observed: list[int] = []
    released: list[int] = []
    parse_json = replay.parse_qrp_json_object
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=16 * 1024**2,
    )

    def observe_interval_owner(raw: bytes):
        value = parse_json(raw)
        if b'"opened_at"' not in raw:
            return value
        observed.append(policy.used_memory_bytes)

        class TrackedInterval(dict):
            pass

        tracked = TrackedInterval(value)
        weakref.finalize(tracked, lambda: released.append(policy.used_memory_bytes))
        return tracked

    monkeypatch.setattr(replay, "parse_qrp_json_object", observe_interval_owner)
    with runtime_resource_policy_scope(policy):
        with open_recorder_run(root, policy):
            pass

    assert observed
    assert min(observed) >= expected_bound
    assert released
    assert min(released) >= expected_bound


def test_invalid_connection_interval_releases_decode_inside_owner_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1)],
            )
        ],
    )
    expected_bound = (
        replay.json_record_working_set_bytes(
            replay._CONNECTION_INTERVAL_MAX_RECORD_BYTES
        )
        + 64 * 1024
    )
    released: list[int] = []
    parse_json = replay.parse_qrp_json_object
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=16 * 1024**2,
    )

    class TrackedInterval(dict):
        pass

    def track_interval(raw: bytes):
        value = parse_json(raw)
        if b'"opened_at"' not in raw:
            return value
        tracked = TrackedInterval(value)
        tracked["market"] = "invalid"
        weakref.finalize(tracked, lambda: released.append(policy.used_memory_bytes))
        return tracked

    monkeypatch.setattr(replay, "parse_qrp_json_object", track_interval)

    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            ValueError,
            match="binance_replay_connection_intervals_invalid",
        ):
            with open_recorder_run(root, policy):
                pass

    assert released
    assert min(released) >= expected_bound


def test_invalid_manifest_releases_decode_inside_owner_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1)],
            )
        ],
    )
    expected_bound = (
        replay.json_record_working_set_bytes((root / "run.json").stat().st_size)
        + 64 * 1024
    )
    released: list[int] = []
    parse_json = replay.parse_qrp_json_object
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=16 * 1024**2,
    )

    class TrackedManifest(dict):
        pass

    def track_manifest(raw: bytes):
        value = parse_json(raw)
        if value.get("schema_id") != "qrp-binance-recorder":
            return value
        tracked = TrackedManifest(value)
        tracked["venue"] = "invalid"
        weakref.finalize(tracked, lambda: released.append(policy.used_memory_bytes))
        return tracked

    monkeypatch.setattr(replay, "parse_qrp_json_object", track_manifest)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            ValueError,
            match="binance_replay_run_manifest_identity_invalid",
        ):
            with open_recorder_run(root, policy):
                pass

    assert released
    assert min(released) >= expected_bound


def test_invalid_raw_record_releases_decode_inside_owner_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1)],
            )
        ],
    )
    released: list[int] = []
    parse_json = replay.parse_qrp_json_object
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=16 * 1024**2,
    )

    class TrackedRecord(dict):
        pass

    def track_record(raw: bytes):
        value = parse_json(raw)
        if "capture_seq" not in value:
            return value
        tracked = TrackedRecord(value)
        tracked["capture_seq"] = -1
        weakref.finalize(tracked, lambda: released.append(policy.used_memory_bytes))
        return tracked

    monkeypatch.setattr(replay, "parse_qrp_json_object", track_record)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="binance_replay_capture_seq_missing"):
            with open_recorder_run(root, policy) as run:
                tuple(run.iter_records(market_type=FUTURES_MARKET_TYPE))

    assert released
    assert all(used > 0 for used in released)


def test_connection_coverage_rejects_disk_before_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    manifest = _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1)],
            )
        ],
    )
    monkeypatch.setattr(
        replay,
        "require_free_disk_headroom",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("coverage_disk_headroom")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        replay.tempfile,
        "TemporaryDirectory",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("coverage_scratch_created_before_disk_preflight")
        ),
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy):
        with pytest.raises(OSError, match="coverage_disk_headroom"):
            replay._ConnectionCoverageIndex(
                root,
                manifest["capture"]["connection_intervals"],
                markets={FUTURES_MARKET_TYPE},
                resource_policy=policy,
            )


def test_connection_coverage_rechecks_disk_and_removes_scratch_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    manifest = _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1)],
            )
        ],
    )
    calls: list[int] = []

    def gate(_path: Path, *, reserve_bytes: int) -> int:
        calls.append(reserve_bytes)
        if len(calls) == 2:
            raise OSError("coverage_disk_headroom")
        return reserve_bytes

    scratch = tempfile.TemporaryDirectory(prefix="qrp-coverage-test-")
    scratch_path = Path(scratch.name)
    monkeypatch.setattr(
        replay,
        "require_free_disk_headroom",
        gate,
        raising=False,
    )
    monkeypatch.setattr(
        replay.tempfile,
        "TemporaryDirectory",
        lambda **_kwargs: scratch,
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    index = None
    try:
        with runtime_resource_policy_scope(policy):
            with pytest.raises(OSError, match="coverage_disk_headroom"):
                index = replay._ConnectionCoverageIndex(
                    root,
                    manifest["capture"]["connection_intervals"],
                    markets={FUTURES_MARKET_TYPE},
                    resource_policy=policy,
                )
    finally:
        if index is not None:
            index.close()

    assert len(calls) == 2
    assert not scratch_path.exists()


@pytest.mark.parametrize(
    ("status", "market_data_status"),
    (("UNKNOWN", "PASS"), ("PASS", "UNKNOWN"), ("PASS", "FAIL")),
)
def test_run_manifest_requires_equal_known_statuses(
    tmp_path: Path,
    status: str,
    market_data_status: str,
) -> None:
    manifest = _write_run(
        tmp_path / "run",
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1)],
            )
        ],
    )
    manifest["status"] = status
    manifest["market_data_status"] = market_data_status

    with pytest.raises(ValueError, match="binance_replay_run_manifest_invalid"):
        replay._validate_run_manifest(manifest)


@pytest.mark.parametrize("status", ("PASS", "PARTIAL", "FAIL"))
def test_run_manifest_accepts_equal_known_statuses(
    tmp_path: Path,
    status: str,
) -> None:
    manifest = _write_run(
        tmp_path / "run",
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1)],
            )
        ],
    )
    manifest["status"] = status
    manifest["market_data_status"] = status

    replay._validate_run_manifest(manifest)


def test_run_manifest_binds_recording_scope_to_exact_market_set(
    tmp_path: Path,
) -> None:
    manifest = _write_run(
        tmp_path / "run",
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1)],
            )
        ],
    )
    manifest["recording_scope"] = SPOT_L2_RECORDING_SCOPE

    with pytest.raises(ValueError, match="binance_replay_run_manifest_invalid"):
        replay._validate_run_manifest(manifest)


def _close_owned_events(events: list[object]) -> None:
    for event in events:
        if isinstance(event, OwnedValue):
            event.lease.close()


def test_agg_trades_normalize_all_clocks_and_visible_quantity(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1)],
            )
        ],
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        trades = list(iter_agg_trades(run, FUTURES_MARKET_TYPE))

    assert len(trades) == 1
    trade = trades[0]
    assert isinstance(trade, TradeEvent)
    assert trade.event_time_ms == 101
    assert trade.transaction_time_ms == 100
    assert trade.observed_time_ms == 1787788800123
    assert trade.quantity == 2.0
    assert trade.visible_quantity == 1.0
    assert trade.price_tick == 1000


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
            (
                FUTURES_MARKET_TYPE,
                "depth",
                "btcusdt@depth@100ms",
                ordered_depth_rows,
            ),
            (
                FUTURES_MARKET_TYPE,
                "depth_snapshot",
                "REST:/fapi/v1/depth",
                snapshots,
            ),
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
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
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
    tmp_path: Path,
    run_id: str,
    valid: bool,
) -> None:
    root = tmp_path / "run"
    _write_run(
        root,
        [
            (
                FUTURES_MARKET_TYPE,
                "agg_trades",
                "btcusdt@aggTrade",
                [_agg_trade(1)],
            )
        ],
        run_id=run_id,
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy):
        if valid:
            with open_recorder_run(root, policy):
                pass
        else:
            with pytest.raises(ValueError, match="run_manifest_invalid"):
                with open_recorder_run(root, policy):
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
    with gzip.open(extra, "wt", encoding="utf-8") as handle:
        handle.write("{}\n")
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="file_manifest_mismatch"):
            with open_recorder_run(root, policy):
                pass

    extra.unlink()
    declared = next(
        path
        for path in root.rglob("*.jsonl.gz")
        if path.name == "usd_s_m_futures_depth.jsonl.gz"
    )
    declared.write_bytes(declared.read_bytes() + b"tampered")
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    with runtime_resource_policy_scope(policy):
        with pytest.raises(ValueError, match="file_identity_mismatch"):
            with open_recorder_run(root, policy):
                pass


def test_futures_bootstrap_keeps_all_clocks_and_exact_tick_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _futures_run(
        root,
        [_futures_delta(1, 90, 100, previous=0, transaction_time=10, segment_id="s1")],
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        events = list(
            iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=1)
        )
        state = events[0].value
        assert isinstance(state, BookState)
        assert state.transaction_time_ms == 10
        assert state.event_time_ms == 12
        assert state.receive_time == "2026-08-27T00:00:00.123456Z"
        assert state.receive_monotonic_ns == 10
        assert state.capture_seq == 2
        assert state.best_bid == (1000, 2.0)
        _close_owned_events(events)
    assert price_to_tick("100.0", "0.1", error="bad") == 1000
    assert tick_to_price(1000, "0.1") == 100.0
    with pytest.raises(ValueError, match="bad"):
        price_to_tick("100.05", "0.1", error="bad")


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
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        owned = next(
            iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=2)
        )
        assert owned.value.bids == ((1000, 2.0), (990, 3.0))
        assert owned.value.asks == ((1010, 2.0), (1020, 3.0))
        assert owned.lease.active
        owned.lease.close()


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
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        events = list(
            iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=2)
        )
        assert isinstance(events[0], BookBoundary)
        assert events[0].reason == "unsupported_coverage"
        assert isinstance(events[1], OwnedValue)
        assert events[1].value.segment_id == "s2"
        events[1].lease.close()


def test_advancing_replay_keeps_previous_book_state_lease_active(
    tmp_path: Path,
) -> None:
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
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        events = iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=1)
        first = next(events)
        second = next(events)
        assert first.lease.active
        assert second.lease.active
        first.lease.close()
        second.lease.close()
        events.close()


def test_futures_zero_transaction_time_is_not_replaced_by_event_time(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _futures_run(
        root,
        [
            _futures_delta(
                1,
                90,
                100,
                previous=0,
                transaction_time=0,
                segment_id="s1",
            )
        ],
    )
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        state = next(
            event
            for event in iter_reconstructed_depth(
                run,
                FUTURES_MARKET_TYPE,
                required_levels=1,
            )
            if isinstance(event, OwnedValue)
        )
        assert state.value.observed_time_ms > state.value.transaction_time_ms
        state.lease.close()


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
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        owned_states = [
            event
            for event in iter_reconstructed_depth(
                run, SPOT_MARKET_TYPE, required_levels=1
            )
            if isinstance(event, OwnedValue)
        ]
        states = [event.value for event in owned_states]
        for event in owned_states:
            event.lease.close()

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
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        events = list(
            iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=1)
        )
        assert [
            type(event.value) if isinstance(event, OwnedValue) else type(event)
            for event in events
        ] == [BookState, BookBoundary, BookState]
        assert events[1].reason == "sequence_gap"
        assert events[1].transaction_time_ms == 20
        assert events[2].value.segment_id == "s2"
        _close_owned_events(events)


@pytest.mark.parametrize(
    ("bids", "asks", "reason"),
    (
        ([["100.0", "0"], ["99.0", "0"]], [], "one_sided_book"),
        ([["101.0", "1"]], [], "crossed_book"),
    ),
)
def test_invalid_book_emits_honest_boundary(
    tmp_path: Path,
    bids: list[list[str]],
    asks: list[list[str]],
    reason: str,
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
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        events = list(
            iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=1)
        )

    assert len(events) == 1
    assert isinstance(events[0], BookBoundary)
    assert events[0].reason == reason


def test_raw_decode_is_admitted_before_json_allocation(tmp_path: Path) -> None:
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
                    )
                ],
            )
        ],
    )
    manifest["files"][0]["max_record_bytes"] = 1_000_000
    (root / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    policy = capture_test_policy(available_memory_bytes=256 * 1024)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            next(run.iter_records())


def test_replay_bootstraps_snapshot_without_locator_rescan(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    _futures_run(
        root,
        [_futures_delta(1, 90, 100, previous=0, transaction_time=10, segment_id="s1")],
    )

    assert not hasattr(replay, "_read_snapshot_at_locator")
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        events = list(
            iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=1)
        )
        _close_owned_events(events)


def test_snapshot_decode_and_bootstrap_share_an_active_transient_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    _futures_run(
        root,
        [_futures_delta(1, 90, 100, previous=0, transaction_time=10, segment_id="s1")],
    )
    observed_owners: list[set[str]] = []
    real_bootstrap = replay.DepthReconstructor.bootstrap

    def probe_bootstrap(self, record):  # noqa: ANN001
        observed_owners.append(
            {lease.owner for lease in self.policy._leases if lease.active}
        )
        return real_bootstrap(self, record)

    monkeypatch.setattr(replay.DepthReconstructor, "bootstrap", probe_bootstrap)
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        events = list(
            iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=1)
        )
        _close_owned_events(events)

    assert any("binance.replay.snapshot" in owners for owners in observed_owners)
    assert policy.used_memory_bytes == 0


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
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    with runtime_resource_policy_scope(policy), open_recorder_run(root, policy) as run:
        first = list(
            iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=1)
        )
        second = list(
            iter_reconstructed_depth(run, FUTURES_MARKET_TYPE, required_levels=1)
        )
        assert [
            event.value if isinstance(event, OwnedValue) else event for event in first
        ] == [
            event.value if isinstance(event, OwnedValue) else event for event in second
        ]
        _close_owned_events(first)
        _close_owned_events(second)
