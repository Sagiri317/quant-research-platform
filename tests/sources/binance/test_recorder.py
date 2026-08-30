from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import quant_research_platform.sources.binance.recorder as recorder
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    SPOT_MARKET_TYPE,
)
from quant_research_platform.sources.binance.recorder import (
    RECORDER_STREAMS_BY_SCOPE,
    RecorderConfig,
)
from quant_research_platform.sources.binance.validation import validate_market_payload
from tests.helpers.runtime_resources import capture_test_policy

FUTURES_STREAMS = RECORDER_STREAMS_BY_SCOPE["futures-hft"]
SPOT_STREAMS = RECORDER_STREAMS_BY_SCOPE["spot-l2"]


def _depth_payload(market_type: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "e": "depthUpdate",
        "E": 1_700_000_000_001,
        "s": "BTCUSDT",
        "U": 10,
        "u": 12,
        "b": [["100.0", "1.0"]],
        "a": [["100.1", "2.0"]],
    }
    if market_type == FUTURES_MARKET_TYPE:
        payload.update(
            {
                "T": 1_700_000_000_000,
                "pu": 9,
                "ps": "BTCUSDT",
                "st": 1,
            }
        )
    return payload


def _validate_stream(stream, payload):  # noqa: ANN001, ANN202
    return validate_market_payload(
        stream.market_type,
        stream.data_kind,
        "BTCUSDT",
        payload,
    )


def test_recorder_scopes_bind_exact_public_streams() -> None:
    scopes = recorder.RECORDER_STREAMS_BY_SCOPE

    assert set(scopes) == {"futures-hft", "spot-l2"}
    assert [(item.market_type, item.data_kind) for item in scopes["futures-hft"]] == [
        (FUTURES_MARKET_TYPE, "depth"),
        (FUTURES_MARKET_TYPE, "agg_trades"),
    ]
    assert [(item.market_type, item.data_kind) for item in scopes["spot-l2"]] == [
        (SPOT_MARKET_TYPE, "depth")
    ]
    assert scopes["futures-hft"][0].url == (
        "wss://fstream.binance.com/public/ws/btcusdt@depth@100ms"
    )
    assert scopes["futures-hft"][1].url == (
        "wss://fstream.binance.com/market/ws/btcusdt@aggTrade"
    )
    assert scopes["spot-l2"][0].url == (
        "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"
    )


def test_run_writer_handle_cleanup_preserves_active_primary(tmp_path: Path) -> None:
    class FailingClose:
        def __init__(self, message: str) -> None:
            self.message = message
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise OSError(self.message)

    raw_handle = FailingClose("raw_close")
    gzip_handle = FailingClose("gzip_close")
    writer = recorder._RunWriter(tmp_path)
    key = (FUTURES_MARKET_TYPE, "depth", "stream", "connection")
    writer._handles[key] = recorder._OpenFile(
        raw_handle=raw_handle,
        gzip_handle=gzip_handle,
        path=tmp_path / "raw.jsonl.gz",
    )

    with pytest.raises(KeyboardInterrupt, match="body_primary") as raised:
        try:
            raise KeyboardInterrupt("body_primary")
        finally:
            writer._close_key(key)

    assert gzip_handle.close_calls == 1
    assert raw_handle.close_calls == 1
    assert raised.value.__notes__ == [
        "Binance gzip handle close also failed: OSError",
        "Binance raw handle close also failed: OSError",
    ]


def test_stream_gzip_constructor_failure_closes_raw_and_preserves_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = KeyboardInterrupt("gzip_constructor_primary")

    class RawHandle:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise OSError("raw_close_secondary")

    raw_handle = RawHandle()
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: raw_handle)

    def fail_gzip(*_args, **_kwargs):  # noqa: ANN202
        raise primary

    monkeypatch.setattr(recorder.gzip, "GzipFile", fail_gzip)
    writer = recorder._RunWriter(tmp_path)

    with pytest.raises(KeyboardInterrupt) as raised:
        writer._open_file(
            ("hour", FUTURES_MARKET_TYPE, "depth", "stream"),
            market_type=FUTURES_MARKET_TYPE,
            data_kind="depth",
            stream="stream",
            receive_time="2026-08-29T00:00:00Z",
        )

    assert raised.value is primary
    assert raw_handle.close_calls == 1
    assert raised.value.__notes__ == ["Binance raw handle close also failed: OSError"]
    assert writer._handles == {}


def test_interval_gzip_constructor_failure_closes_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = KeyboardInterrupt("gzip_constructor_primary")
    raw_handle = (tmp_path / "tracked.bin").open("wb")
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: raw_handle)

    def fail_gzip(*_args, **_kwargs):  # noqa: ANN202
        raise primary

    monkeypatch.setattr(recorder.gzip, "GzipFile", fail_gzip)
    writer = recorder._RunWriter(tmp_path)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            writer._open_interval_file()
        assert raised.value is primary
        assert raw_handle.closed
        assert writer._interval_raw_handle is None
        assert writer._interval_gzip_handle is None
    finally:
        raw_handle.close()


def test_run_writer_attempts_all_stream_closes_after_first_failure(
    tmp_path: Path,
) -> None:
    class Handle:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.fail:
                raise OSError("first_gzip_close")

    first_gzip = Handle(fail=True)
    first_raw = Handle()
    second_gzip = Handle()
    second_raw = Handle()
    writer = recorder._RunWriter(tmp_path)
    writer._handles = {
        ("00", "market", "depth", "first"): recorder._OpenFile(
            first_raw,
            first_gzip,
            tmp_path / "first.gz",
        ),
        ("00", "market", "depth", "second"): recorder._OpenFile(
            second_raw,
            second_gzip,
            tmp_path / "second.gz",
        ),
    }

    with pytest.raises(OSError, match="first_gzip_close"):
        writer.close()

    assert (first_gzip.close_calls, first_raw.close_calls) == (1, 1)
    assert (second_gzip.close_calls, second_raw.close_calls) == (1, 1)
    assert writer._handles == {}
    assert writer._closed is True


def test_run_writer_rejects_disk_before_first_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "run"
    writer = recorder._RunWriter(root)

    def reject(path: Path, *, reserve_bytes: int) -> int:
        assert path == root
        assert reserve_bytes > 0
        assert not root.exists()
        raise OSError("recorder_disk_headroom")

    monkeypatch.setattr(
        recorder,
        "require_free_disk_headroom",
        reject,
        raising=False,
    )
    try:
        with pytest.raises(OSError, match="recorder_disk_headroom"):
            writer.write(
                market_type=FUTURES_MARKET_TYPE,
                data_kind="depth",
                stream="btcusdt@depth@100ms",
                payload={},
                receive_time="2026-08-27T00:00:00.000000Z",
                receive_monotonic_ns=1,
            )
        assert writer._capture_seq == 0
        assert not root.exists()
    finally:
        writer.abort()


def test_run_writer_rechecks_cumulative_disk_and_closes_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def gate(_path: Path, *, reserve_bytes: int) -> int:
        calls.append(reserve_bytes)
        if len(calls) == 2:
            raise OSError("recorder_disk_headroom")
        return reserve_bytes

    monkeypatch.setattr(
        recorder,
        "require_free_disk_headroom",
        gate,
        raising=False,
    )
    monkeypatch.setattr(recorder, "_RECORDER_DISK_CHECK_BYTES", 1, raising=False)
    writer = recorder._RunWriter(tmp_path / "run")
    write_args = {
        "market_type": FUTURES_MARKET_TYPE,
        "data_kind": "depth",
        "stream": "btcusdt@depth@100ms",
        "payload": {},
        "receive_time": "2026-08-27T00:00:00.000000Z",
    }
    try:
        writer.write(**write_args, receive_monotonic_ns=1)
        with pytest.raises(OSError, match="recorder_disk_headroom"):
            writer.write(**write_args, receive_monotonic_ns=2)
        assert writer._capture_seq == 1
        assert next(iter(writer._files.values()))["record_count"] == 1
    finally:
        writer.abort()

    assert len(calls) == 2
    assert writer._closed is True
    assert writer._handles == {}


def test_recorder_validates_spot_and_futures_depth_semantics() -> None:
    spot, futures = SPOT_STREAMS[0], FUTURES_STREAMS[0]

    assert _validate_stream(spot, _depth_payload(SPOT_MARKET_TYPE))["u"] == 12
    assert _validate_stream(futures, _depth_payload(FUTURES_MARKET_TYPE))["pu"] == 9

    with pytest.raises(ValueError, match="market_payload_identity_invalid"):
        _validate_stream(
            futures,
            {**_depth_payload(FUTURES_MARKET_TYPE), "st": 2},
        )


def test_recorder_validates_futures_aggregate_trade_semantics() -> None:
    aggregate_trade = FUTURES_STREAMS[1]
    payload = {
        "e": "aggTrade",
        "E": 1_700_000_000_001,
        "T": 1_700_000_000_000,
        "s": "BTCUSDT",
        "st": 1,
        "a": 10,
        "f": 20,
        "l": 22,
        "p": "100.1",
        "q": "2.5",
        "nq": "2.0",
        "m": False,
    }

    assert _validate_stream(aggregate_trade, payload) is payload

    with pytest.raises(ValueError, match="trade_quantity_invalid"):
        _validate_stream(aggregate_trade, {**payload, "q": "0"})

    with pytest.raises(ValueError, match="visible_quantity_invalid"):
        _validate_stream(aggregate_trade, {**payload, "nq": "3.0"})

    missing_visible = dict(payload)
    missing_visible.pop("nq")
    with pytest.raises(ValueError, match="visible_quantity_invalid"):
        _validate_stream(aggregate_trade, missing_visible)


def test_recorder_config_rejects_existing_or_unbounded_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duration_invalid"):
        RecorderConfig(output_root=tmp_path / "run", duration_seconds=0).validate()
    with pytest.raises(ValueError, match="scope_invalid"):
        RecorderConfig(
            output_root=tmp_path / "run",
            duration_seconds=1,
            scope="mixed",
        ).validate()

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError, match="output_exists"):
        RecorderConfig(output_root=empty, duration_seconds=1).validate()

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output_exists"):
        RecorderConfig(output_root=occupied, duration_seconds=1).validate()


def test_recorder_help_is_gbk_safe() -> None:
    environment = {
        **os.environ,
        "PYTHONIOENCODING": "gbk",
        "PYTHONPATH": str(Path("src").resolve()),
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_research_platform.sources.binance.recorder",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode("gbk", errors="replace")


def test_recorder_rejects_aliased_output_root_before_access(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory_symlink_not_available: {exc}")

    with pytest.raises(ValueError, match="invalid_storage_write_path"):
        RecorderConfig(alias, duration_seconds=1).validate()


def test_depth_reader_waits_for_same_generation_snapshot_ack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = FUTURES_STREAMS[0]
    payload = _depth_payload(FUTURES_MARKET_TYPE)
    payload.update({"U": 1, "u": 1, "pu": 0})
    writes: list[dict[str, object]] = []

    class Writer:
        def write(self, **kwargs: object) -> int:
            writes.append(kwargs)
            return len(writes)

    class Websocket:
        def __init__(self) -> None:
            self.sent = False
            self.release = threading.Event()

        def recv(self, timeout: float) -> str:
            del timeout
            if not self.sent:
                self.sent = True
                return json.dumps(payload)
            self.release.wait(1)
            raise TimeoutError

        def close(self) -> None:
            self.release.set()

    websocket = Websocket()
    monkeypatch.setattr(
        recorder,
        "_connect_websocket",
        lambda *_args, **_kwargs: websocket,
    )
    reader = recorder._StreamReader(
        stream=stream,
        writer=Writer(),
        stats=recorder._CaptureStats(),
        stop=threading.Event(),
        config=RecorderConfig(Path("unused"), 1),
    )
    reader.start()
    assert reader.ready.wait(1)
    try:
        assert writes == []
        generation, _connection_id, _segment_id = reader.current_identity()
        assert reader.acknowledge_snapshot(generation)
        assert reader.stop.wait(0.05) is False
    finally:
        reader.stop.set()
        websocket.release.set()
        reader.join(1)
    assert len(writes) == 1


def test_connection_interval_freezes_before_websocket_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    order: list[str] = []

    class Websocket:
        def recv(self, timeout: float) -> None:
            del timeout
            raise OSError("connection lost")

        def close(self) -> None:
            order.append("websocket")

    stats = recorder._CaptureStats()
    real_closed = stats.closed

    def record_closed(*args: object, **kwargs: object) -> None:
        order.append("interval")
        real_closed(*args, **kwargs)

    monkeypatch.setattr(stats, "closed", record_closed)
    monkeypatch.setattr(
        recorder,
        "_connect_websocket",
        lambda *_args, **_kwargs: Websocket(),
    )
    reader = recorder._StreamReader(
        stream=FUTURES_STREAMS[1],
        writer=recorder._RunWriter(tmp_path),
        stats=stats,
        stop=threading.Event(),
        config=RecorderConfig(tmp_path / "run", 1),
    )
    with pytest.raises(recorder._ConnectionProblem):
        reader._read_connection(reconnect=False)

    assert order == ["interval", "websocket"]


def test_connection_stats_close_failure_still_closes_websocket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Websocket:
        closed = False

        def close(self) -> None:
            self.closed = True

    websocket = Websocket()
    stats = recorder._CaptureStats()

    def fail_closed(*_args: object, **_kwargs: object) -> None:
        raise OSError("stats-close-primary")

    monkeypatch.setattr(stats, "closed", fail_closed)
    monkeypatch.setattr(
        recorder,
        "_connect_websocket",
        lambda *_args, **_kwargs: websocket,
    )
    reader = recorder._StreamReader(
        stream=FUTURES_STREAMS[1],
        writer=recorder._RunWriter(tmp_path),
        stats=stats,
        stop=threading.Event(),
        config=RecorderConfig(tmp_path / "run", 1),
    )
    reader.stop.set()

    with pytest.raises(OSError, match="stats-close-primary"):
        reader._read_connection(reconnect=False)

    assert websocket.closed


def test_connection_close_time_is_frozen_at_error_detection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    phase = {"value": "open"}
    closed: list[str] = []

    def utc_now() -> str:
        return phase["value"]

    class LateConnectionProblem(Exception):
        def __init__(self) -> None:
            phase["value"] = "after"

    class Websocket:
        def recv(self, timeout: float) -> None:
            del timeout
            phase["value"] = "detected"
            raise OSError("connection lost")

        def close(self) -> None:
            return None

    stats = recorder._CaptureStats()

    def record_closed(*args: object, **kwargs: object) -> None:
        del args
        closed.append(str(kwargs["closed_at"]))

    monkeypatch.setattr(recorder, "_utc_now", utc_now)
    monkeypatch.setattr(recorder, "_ConnectionProblem", LateConnectionProblem)
    monkeypatch.setattr(stats, "closed", record_closed)
    monkeypatch.setattr(
        recorder,
        "_connect_websocket",
        lambda *_args, **_kwargs: Websocket(),
    )
    reader = recorder._StreamReader(
        stream=FUTURES_STREAMS[1],
        writer=recorder._RunWriter(tmp_path),
        stats=stats,
        stop=threading.Event(),
        config=RecorderConfig(tmp_path / "run", 1),
    )

    with pytest.raises(LateConnectionProblem):
        reader._read_connection(reconnect=False)

    assert closed == ["detected"]


def test_run_writer_rejects_late_writes_after_close(tmp_path: Path) -> None:
    writer = recorder._RunWriter(tmp_path)
    writer.write(
        market_type=FUTURES_MARKET_TYPE,
        data_kind="agg_trades",
        stream="btcusdt@aggTrade",
        payload={},
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=1,
    )
    writer.close()
    path = next(tmp_path.rglob("*.jsonl.gz"))
    before = path.read_bytes()

    with pytest.raises(RuntimeError, match="writer_closed"):
        writer.write(
            market_type=FUTURES_MARKET_TYPE,
            data_kind="agg_trades",
            stream="btcusdt@aggTrade",
            payload={},
            receive_time="2026-08-27T00:00:00.000000Z",
            receive_monotonic_ns=2,
        )

    assert path.read_bytes() == before


def test_capture_stats_streams_finalized_intervals_to_sidecar(
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    writer = recorder._RunWriter(tmp_path)

    with runtime_resource_policy_scope(policy):
        stats = recorder._CaptureStats(policy, writer)
        for index in range(10_000):
            connection_id = f"trade-{index}"
            stats.opened(
                FUTURES_STREAMS[1],
                reconnect=index > 0,
                connection_id=connection_id,
                segment_id=None,
                opened_at="1970-01-01T00:00:00.000000000Z",
                opened_monotonic_ns=index,
            )
            stats.closed(
                connection_id,
                closed_at="1970-01-01T00:00:01.000000000Z",
                closed_monotonic_ns=index + 1,
                close_reason="stop",
            )

        assert stats.connection_intervals == {}
        writer.close()
        descriptor = writer.connection_interval_manifest(policy)
        stats.set_connection_interval_manifest(descriptor)
        stats.close()
        payload = stats.payload()

    sidecar = tmp_path / descriptor["path"]
    assert payload["connection_intervals"] == {
        "path": "connection_intervals.jsonl.gz",
        "record_count": 10_000,
        "compressed_bytes": sidecar.stat().st_size,
        "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    }
    with gzip.open(sidecar, "rt", encoding="utf-8") as handle:
        assert sum(1 for _ in handle) == 10_000


def test_recorder_does_not_publish_when_reader_join_times_out(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(available_memory_bytes=512 * 1024**2)
    output_root = tmp_path / "run"
    stream = FUTURES_STREAMS[1]

    class Reader:
        def __init__(self, **kwargs: object) -> None:
            self.name = "fake-reader"
            self.stream = kwargs["stream"]
            self.ready = threading.Event()
            self.failure = None

        def start(self) -> None:
            self.ready.set()

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float) -> None:
            del timeout

    monkeypatch.setitem(
        recorder.RECORDER_STREAMS_BY_SCOPE,
        "futures-hft",
        (stream,),
    )
    monkeypatch.setattr(
        recorder,
        "runtime_resource_policy_scope",
        lambda *_args, **_kwargs: runtime_resource_policy_scope(policy),
    )
    monkeypatch.setattr(
        recorder,
        "_record_market_metadata",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(recorder, "_StreamReader", Reader)

    with pytest.raises(RuntimeError, match="quiescence_timeout"):
        recorder.record(RecorderConfig(output_root, duration_seconds=0.001))

    assert not output_root.exists()


def test_recorder_abort_failure_preserves_reader_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy = capture_test_policy(available_memory_bytes=512 * 1024**2)
    output_root = tmp_path / "run"
    stream = FUTURES_STREAMS[1]

    class Reader:
        def __init__(self, **kwargs: object) -> None:
            self.name = "fake-reader"
            self.stream = kwargs["stream"]
            self.ready = threading.Event()
            self.failure = "reader-failure"

        def start(self) -> None:
            self.ready.set()

        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float) -> None:
            del timeout

    def abort(_writer: object) -> None:
        raise OSError("abort-secondary")

    monkeypatch.setitem(
        recorder.RECORDER_STREAMS_BY_SCOPE,
        "futures-hft",
        (stream,),
    )
    monkeypatch.setattr(
        recorder,
        "runtime_resource_policy_scope",
        lambda *_args, **_kwargs: runtime_resource_policy_scope(policy),
    )
    monkeypatch.setattr(
        recorder,
        "_record_market_metadata",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(recorder, "_StreamReader", Reader)
    monkeypatch.setattr(recorder._RunWriter, "abort", abort)

    with pytest.raises(RuntimeError, match="reader_failure"):
        recorder.record(RecorderConfig(output_root, duration_seconds=0.001))


def test_connection_interval_retention_is_admitted_and_released() -> None:
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    with runtime_resource_policy_scope(policy):
        stats = recorder._CaptureStats(policy)
        stats.opened(
            FUTURES_STREAMS[1],
            reconnect=False,
            connection_id="trade-1",
            segment_id=None,
            opened_at="1970-01-01T00:00:00.000000000Z",
            opened_monotonic_ns=0,
        )
        assert policy.used_memory_bytes >= 1024
        stats.closed(
            "trade-1",
            closed_at="1970-01-01T00:00:01.000000000Z",
            closed_monotonic_ns=1_000_000_000,
            close_reason="stop",
        )
        stats.close()
        assert policy.used_memory_bytes == 0
