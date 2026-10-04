from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import threading
import weakref
from pathlib import Path

import pytest

import quant_research_platform.runtime.resources as resources
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


def _snapshot_payload(update_id, mid_tick=1000):
    return {
        "lastUpdateId": update_id,
        "E": 1_700_000_000_001,
        "T": 1_700_000_000_000,
        "bids": [[f"{(mid_tick - i) / 10:.1f}", "1.0"] for i in range(12)],
        "asks": [[f"{(mid_tick + 1 + i) / 10:.1f}", "2.0"] for i in range(12)],
    }


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
    assert [
        (item.symbol, item.market_type, item.data_kind)
        for item in scopes["futures-hft"]
    ] == [
        ("BTCUSDT", FUTURES_MARKET_TYPE, "depth"),
        ("BTCUSDT", FUTURES_MARKET_TYPE, "agg_trades"),
        ("ETHUSDT", FUTURES_MARKET_TYPE, "depth"),
        ("ETHUSDT", FUTURES_MARKET_TYPE, "agg_trades"),
    ]
    assert [
        (item.symbol, item.market_type, item.data_kind) for item in scopes["spot-l2"]
    ] == [
        ("BTCUSDT", SPOT_MARKET_TYPE, "depth"),
        ("ETHUSDT", SPOT_MARKET_TYPE, "depth"),
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


def test_websocket_reservation_covers_queue_and_current_decode() -> None:
    assert recorder._WS_STREAM_WORKING_SET_BYTES == (
        recorder.MAX_WS_QUEUE * recorder.MAX_WS_MESSAGE_BYTES * 4
        + recorder.json_record_working_set_bytes(recorder.MAX_WS_MESSAGE_BYTES)
        + 64 * 1024
    )


def test_rest_metadata_releases_previous_payload_before_next_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[int] = []
    requests = []
    writes = []
    observations: list[bool] = []
    policy = capture_test_policy(
        available_memory_bytes=2 * 1024**3,
        explicit_cap_bytes=1024**3,
    )

    class Payload(dict):
        pass

    def get_json(base_url, endpoint, *, params):  # noqa: ANN202
        requests.append((base_url, endpoint, params))
        if observations:
            observations.append(bool(released))
        else:
            observations.append(False)
        payload = Payload(
            {
                "symbols": [
                    {
                        "symbol": "BTCUSDT",
                        "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.1"}],
                    },
                    {
                        "symbol": "ETHUSDT",
                        "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"}],
                    },
                ]
            }
        )
        weakref.finalize(payload, released.append, policy.used_memory_bytes)
        return payload, recorder._utc_now(), len(requests)

    class Writer:
        def write(self, **kwargs) -> None:
            writes.append(
                (
                    kwargs["symbol"],
                    kwargs["market_type"],
                    kwargs["receive_time"],
                    kwargs["receive_monotonic_ns"],
                )
            )

    monkeypatch.setattr(recorder, "_http_get_json", get_json)
    with runtime_resource_policy_scope(policy):
        recorder._record_market_metadata(
            Writer(),
            policy,
            market_types=(SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE),
            symbols=recorder.RESEARCH_SYMBOLS,
        )

    assert observations == [False, True, True]
    assert [params for _base, _endpoint, params in requests] == [
        {"symbol": "BTCUSDT"},
        {"symbol": "ETHUSDT"},
        None,
    ]
    futures_writes = [row for row in writes if row[1] == FUTURES_MARKET_TYPE]
    assert [row[0] for row in futures_writes] == list(recorder.RESEARCH_SYMBOLS)
    assert futures_writes[0][2:] == futures_writes[1][2:]
    assert released
    assert all(used > 0 for used in released)


def test_depth_validation_maps_large_json_integer_to_stable_owner_error() -> None:
    payload = _depth_payload(FUTURES_MARKET_TYPE)
    payload["b"] = [[10**4000, "1.0"]]

    with pytest.raises(ValueError, match="binance_recorder_depth_bids_invalid"):
        _validate_stream(FUTURES_STREAMS[0], payload)


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
    monkeypatch.setattr(
        recorder, "open_filesystem_file", lambda *_args, **_kwargs: raw_handle
    )

    def fail_gzip(*_args, **_kwargs):  # noqa: ANN202
        raise primary

    monkeypatch.setattr(recorder.gzip, "GzipFile", fail_gzip)
    writer = recorder._RunWriter(tmp_path)

    with pytest.raises(KeyboardInterrupt) as raised:
        writer._open_file(
            ("hour", "BTCUSDT", FUTURES_MARKET_TYPE, "depth", "stream"),
            symbol="BTCUSDT",
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
    monkeypatch.setattr(
        recorder, "open_filesystem_file", lambda *_args, **_kwargs: raw_handle
    )

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
                symbol="BTCUSDT",
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
        "symbol": "BTCUSDT",
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


def test_depth_reader_waits_for_snapshot_before_receiving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=2 * 1024**3)
    requested = threading.Event()
    release_snapshot = threading.Event()
    written = threading.Event()
    stop = threading.Event()
    writes = []
    payload = _depth_payload(FUTURES_MARKET_TYPE)
    payload.update({"U": 1, "u": 1, "pu": 0})

    class Writer:
        def write(self, **kwargs):
            writes.append(kwargs)
            if kwargs["data_kind"] == "depth":
                written.set()
            return len(writes)

    class Websocket:
        received = False

        def recv(self, timeout):
            if not self.received:
                self.received = True
                return json.dumps(payload)
            stop.wait(timeout)
            raise TimeoutError

        def close(self):
            stop.set()

    def snapshot(*_args, **_kwargs):
        requested.set()
        assert release_snapshot.wait(2)
        return _snapshot_payload(1), recorder._utc_now(), recorder.time.monotonic_ns()

    websocket = Websocket()
    monkeypatch.setattr(recorder, "_connect_websocket", lambda *_a, **_k: websocket)
    monkeypatch.setattr(recorder, "_http_get_json", snapshot)
    reader = recorder._StreamReader(
        stream=FUTURES_STREAMS[0],
        writer=Writer(),
        stats=recorder._CaptureStats(),
        stop=stop,
        config=RecorderConfig(Path("unused"), 1),
        resource_policy=policy,
        tick_size="0.1",
    )
    with runtime_resource_policy_scope(policy):
        reader.start()
        try:
            assert requested.wait(2)
            assert not reader.ready.is_set()
            assert not websocket.received
            assert writes == []
            release_snapshot.set()
            assert reader.ready.wait(2)
            assert written.wait(2)
        finally:
            release_snapshot.set()
            stop.set()
            reader.join(3)
        assert not reader.is_alive()
        assert reader.failure is None
        assert policy.used_memory_bytes == 0
    assert [row["data_kind"] for row in writes] == ["depth_snapshot", "depth"]
    assert writes[0]["connection_id"] == writes[1]["connection_id"]
    assert writes[0]["segment_id"] == writes[1]["segment_id"]


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
        resource_policy=capture_test_policy(available_memory_bytes=64 * 1024**2),
        tick_size="0.1",
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
        resource_policy=capture_test_policy(available_memory_bytes=64 * 1024**2),
        tick_size="0.1",
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
        resource_policy=capture_test_policy(available_memory_bytes=64 * 1024**2),
        tick_size="0.1",
    )

    with pytest.raises(LateConnectionProblem):
        reader._read_connection(reconnect=False)

    assert closed == ["detected"]


def test_run_writer_rejects_late_writes_after_close(tmp_path: Path) -> None:
    writer = recorder._RunWriter(tmp_path)
    writer.write(
        symbol="BTCUSDT",
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
            symbol="BTCUSDT",
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


@pytest.mark.parametrize(
    "join_completes,late_failure", [(False, False), (True, False), (True, True)]
)
def test_recorder_publishes_only_after_reader_and_writer_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    join_completes: bool,
    late_failure: bool,
) -> None:
    # The fake policy has no OS lock; restore only its expected process poison.
    monkeypatch.setattr(resources, "_PROCESS_POISONED", resources._PROCESS_POISONED)
    policy = capture_test_policy(available_memory_bytes=512 * 1024**2)
    output_root = tmp_path / "run"
    stream = FUTURES_STREAMS[1]

    class Reader:
        def __init__(self, **kwargs: object) -> None:
            self.name = "fake-reader"
            self.stream = kwargs["stream"]
            self.ready = threading.Event()
            self.failure = None
            self.alive = True

        def start(self) -> None:
            self.ready.set()

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float) -> None:
            del timeout
            self.alive = not join_completes
            if late_failure:
                self.failure = OSError("final_write_failed")

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
        lambda *_args, **_kwargs: {
            stream.symbol: {stream.market_type: {"tick_size": "0.1"}}
        },
    )
    monkeypatch.setattr(recorder, "_StreamReader", Reader)
    original_json = recorder.atomic_write_json
    original_directory = recorder.atomic_write_directory
    observed = []

    def write_json(*args, **kwargs):
        assert policy.state == "ACTIVE"
        assert any(
            lease.owner == "binance.recorder.manifest" for lease in policy._leases
        )
        observed.append("json")
        return original_json(*args, **kwargs)

    def publish_directory(path, write):
        def staged(root):
            result = write(root)
            assert policy.state == "ACTIVE"
            observed.append("publish")
            return result

        result = original_directory(path, staged)
        assert policy.state == "ACTIVE"
        return result

    monkeypatch.setattr(recorder, "atomic_write_json", write_json)
    monkeypatch.setattr(recorder, "atomic_write_directory", publish_directory)

    if late_failure:
        with pytest.raises(RuntimeError, match="binance_recorder_reader_failure"):
            recorder.record(RecorderConfig(output_root, duration_seconds=0.001))
        assert not output_root.exists()
    elif join_completes:
        result = recorder.record(RecorderConfig(output_root, duration_seconds=0.001))
        assert result == json.loads(
            (output_root / "run.json").read_text(encoding="utf-8")
        )
        assert result["status"] == "PARTIAL"
        assert observed == ["json", "publish"]
        descriptor = result["capture"]["connection_intervals"]
        sidecar = output_root / descriptor["path"]
        assert descriptor["sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
        assert descriptor["record_count"] == 0
        with gzip.open(sidecar, "rb") as handle:
            assert handle.read() == b""
    else:
        with pytest.raises(RuntimeError, match="quiescence_timeout"):
            recorder.record(RecorderConfig(output_root, duration_seconds=0.001))
        assert not output_root.exists()
    if join_completes:
        assert policy.used_memory_bytes == 0
    else:
        assert policy.used_memory_bytes == recorder._WS_STREAM_WORKING_SET_BYTES
        assert policy.state == "POISONED"


def test_recorder_abort_failure_preserves_reader_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # The fake policy has no OS lock; restore only its expected process poison.
    monkeypatch.setattr(resources, "_PROCESS_POISONED", resources._PROCESS_POISONED)
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
        lambda *_args, **_kwargs: {
            stream.symbol: {stream.market_type: {"tick_size": "0.1"}}
        },
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


@pytest.mark.parametrize("failure_type", [TimeoutError, OSError])
def test_snapshot_network_failure_preserves_cause_and_closes_connection(
    tmp_path, monkeypatch, failure_type
):
    policy = capture_test_policy(available_memory_bytes=2 * 1024**3)
    primary = failure_type("snapshot network failure")

    class Websocket:
        closed = False

        def close(self):
            self.closed = True
            raise OSError("close secondary")

    def fail_snapshot(*_args, **_kwargs):
        raise primary

    websocket = Websocket()
    stats = recorder._CaptureStats()
    monkeypatch.setattr(recorder, "_connect_websocket", lambda *_a, **_k: websocket)
    monkeypatch.setattr(recorder, "_http_get_json", fail_snapshot)
    reader = recorder._StreamReader(
        stream=FUTURES_STREAMS[0],
        writer=recorder._RunWriter(tmp_path),
        stats=stats,
        stop=threading.Event(),
        config=RecorderConfig(tmp_path / "run", 1),
        resource_policy=policy,
        tick_size="0.1",
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(recorder._ConnectionProblem) as raised:
            reader._read_connection(reconnect=False)
        assert raised.value.__cause__ is primary
        assert "Binance websocket close also failed: OSError" in raised.value.__notes__
        assert websocket.closed
        assert not reader.ready.is_set()
        assert stats.connection_intervals == {}
        assert policy.used_memory_bytes == 0


def test_recorder_recovers_exhausted_known_depth_with_new_connection_and_snapshot(
    tmp_path, monkeypatch
):
    from collections import Counter
    from types import SimpleNamespace

    from quant_research_platform.contracts.order_book import BookBoundary
    from quant_research_platform.sources.binance.replay import (
        ReconstructionStats,
        iter_reconstructed_depth,
        open_recorder_run,
    )

    policy = capture_test_policy(available_memory_bytes=4 * 1024**3)
    connections = Counter()
    snapshots = Counter()
    sockets = []
    recovered_recorded = threading.Event()
    clock = 0.0
    real_time = recorder.time
    recovery_deadline = real_time.monotonic() + 10
    real_write = recorder._RunWriter.write

    def monotonic():
        nonlocal clock
        if threading.current_thread() is not threading.main_thread():
            return real_time.monotonic()
        if recovered_recorded.is_set():
            clock += 1.0
        elif real_time.monotonic() >= recovery_deadline:
            raise AssertionError("recorder_recovery_did_not_complete")
        return clock

    def write(self, **kwargs):
        result = real_write(self, **kwargs)
        if (
            kwargs["symbol"] == "BTCUSDT"
            and kwargs["data_kind"] == "depth"
            and kwargs["payload"]["u"] == 102
        ):
            recovered_recorded.set()
        return result

    monkeypatch.setattr(
        recorder,
        "time",
        SimpleNamespace(
            monotonic=monotonic,
            monotonic_ns=real_time.monotonic_ns,
            process_time=real_time.process_time,
            time_ns=real_time.time_ns,
        ),
    )
    monkeypatch.setattr(recorder._RunWriter, "write", write)

    def http(_base, endpoint, *, params=None):
        if endpoint.endswith("exchangeInfo"):
            payload = {
                "symbols": [
                    {
                        "symbol": symbol,
                        "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.1"}],
                    }
                    for symbol in recorder.RESEARCH_SYMBOLS
                ]
            }
        else:
            symbol = params["symbol"]
            snapshots[symbol] += 1
            recovered = symbol == "BTCUSDT" and snapshots[symbol] > 1
            payload = _snapshot_payload(
                102 if recovered else 100, 900 if recovered else 1000
            )
        return payload, recorder._utc_now(), recorder.time.monotonic_ns()

    class Websocket:
        def __init__(self, messages):
            self.messages = iter(messages)
            self.closed = threading.Event()

        def recv(self, timeout):
            try:
                return json.dumps(next(self.messages))
            except StopIteration:
                self.closed.wait(timeout)
                raise TimeoutError from None

        def close(self):
            self.closed.set()

    def connect(url, **_kwargs):
        symbol = "BTCUSDT" if "btcusdt" in url else "ETHUSDT"
        if "depth" in url:
            connections[symbol] += 1
            recovered = symbol == "BTCUSDT" and connections[symbol] > 1
            initial = _depth_payload(FUTURES_MARKET_TYPE)
            initial.update(
                s=symbol,
                ps=symbol,
                U=102 if recovered else 100,
                u=102 if recovered else 100,
                pu=101 if recovered else 99,
            )
            if recovered:
                initial.update(b=[["90.0", "1.0"]], a=[["90.1", "2.0"]])
            messages = [initial]
            if symbol == "BTCUSDT" and not recovered:
                old = _snapshot_payload(100)
                fresh = _snapshot_payload(102, 900)
                depleted = dict(initial, U=101, u=101, pu=100)
                depleted["b"] = [[p, "0"] for p, _q in old["bids"]] + fresh["bids"]
                depleted["a"] = [[p, "0"] for p, _q in old["asks"]] + fresh["asks"]
                messages.append(depleted)
        else:
            messages = [
                {
                    "e": "aggTrade",
                    "E": 1_700_000_000_001,
                    "T": 1_700_000_000_000,
                    "s": symbol,
                    "st": 1,
                    "a": 10,
                    "f": 20,
                    "l": 22,
                    "p": "100.1",
                    "q": "2.5",
                    "nq": "2.0",
                    "m": False,
                }
            ]
        socket = Websocket(messages)
        sockets.append(socket)
        return socket

    monkeypatch.setattr(recorder, "_http_get_json", http)
    monkeypatch.setattr(recorder, "_connect_websocket", connect)
    monkeypatch.setattr(
        recorder,
        "runtime_resource_policy_scope",
        lambda *_a, **_k: runtime_resource_policy_scope(policy),
    )
    root = tmp_path / "recording"
    result = recorder.record(RecorderConfig(root, duration_seconds=0.2))
    assert recovered_recorded.is_set()
    assert result["status"] == "PARTIAL"
    assert result["capture"]["sequence_gaps"] == 0
    assert result["capture"]["cleanup_errors"] == []
    assert connections == {"BTCUSDT": 2, "ETHUSDT": 1}
    assert snapshots == connections
    assert all(socket.closed.is_set() for socket in sockets)
    assert policy.used_memory_bytes == 0
    stats = ReconstructionStats()
    states = []
    boundaries = []
    with open_recorder_run(root) as run:
        records = list(
            run.iter_records(
                symbol="BTCUSDT",
                market_type=FUTURES_MARKET_TYPE,
                data_kinds=("depth_snapshot", "depth"),
            )
        )
        assert [r["payload"]["u"] for r in records if r["data_kind"] == "depth"] == [
            100,
            101,
            102,
        ]
        snapshot_records = [r for r in records if r["data_kind"] == "depth_snapshot"]
        assert len({r["connection_id"] for r in snapshot_records}) == 2
        assert len({r["segment_id"] for r in snapshot_records}) == 2
        for event in iter_reconstructed_depth(
            run, FUTURES_MARKET_TYPE, symbol="BTCUSDT", stats=stats
        ):
            if isinstance(event, BookBoundary):
                boundaries.append(event.reason)
            else:
                states.append(event.value.update_id)
                event.release(owner="test.recorder.recovery")
    assert states == [100, 102]
    assert boundaries == ["one_sided_book"]
    assert stats.valid_segments == 2
    assert stats.recovered_segments == 1
    assert stats.one_sided_books == 1
    assert len(stats.coverage_segments) == 2


def test_recorder_unquiesced_real_threads_keep_budget_and_machine_lock(tmp_path):
    script = r'''
import contextlib, json, sys, threading, time
from pathlib import Path
from quant_research_platform.sources.binance import recorder
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from tests.sources.binance.test_recorder import _snapshot_payload
import subprocess

release = threading.Event()
sockets, policies = [], []
class Socket:
    def __init__(self):
        self.buffer = bytearray(1024**2)
    def recv(self, timeout):
        time.sleep(.01)
        raise TimeoutError
    def close(self):
        release.wait(10)
        self.buffer = None

def connect(*args, **kwargs):
    socket = Socket()
    sockets.append(socket)
    return socket

def http(_base, endpoint, *, params=None):
    payload = ({"symbols": [{"symbol": s, "filters": [
        {"filterType": "PRICE_FILTER", "tickSize": "0.1"}
    ]} for s in recorder.RESEARCH_SYMBOLS]}
        if endpoint.endswith("exchangeInfo") else _snapshot_payload(100))
    return payload, recorder._utc_now(), time.monotonic_ns()

@contextlib.contextmanager
def scope(*args, **kwargs):
    with runtime_resource_policy_scope(*args, **kwargs) as policy:
        policies.append(policy)
        yield policy

join = recorder._StreamReader.join
recorder._connect_websocket = connect
recorder._http_get_json = http
recorder.runtime_resource_policy_scope = scope
recorder._StreamReader.join = lambda self, timeout=None: join(self, .03)
root = Path(sys.argv[1])
try:
    try:
        recorder.record(recorder.RecorderConfig(root, duration_seconds=.02))
    except RuntimeError as error:
        assert str(error) == "binance_recorder_reader_quiescence_timeout"
    else:
        raise AssertionError("unquiesced_recording_published")
    alive = [t for t in threading.enumerate() if t.name.startswith("binance-")]
    assert len(alive) == 4 and not root.exists()
    assert sum(len(s.buffer) for s in sockets) == 4 * 1024**2
    assert policies[0].used_memory_bytes == 4 * recorder._WS_STREAM_WORKING_SET_BYTES
    assert policies[0].state == "POISONED"
    try:
        with runtime_resource_policy_scope():
            raise AssertionError("poisoned_root_reused")
    except RuntimeError as error:
        assert str(error) == "runtime_data_scale_process_poisoned"
    competitor = subprocess.run([sys.executable, "-c", """
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
try:
    with runtime_resource_policy_scope(): print("ACQUIRED")
except RuntimeError as error: print(str(error))
"""], text=True, capture_output=True, timeout=10)
    assert competitor.returncode == 0
    assert competitor.stdout.strip() == "runtime_data_scale_lock_busy"
finally:
    release.set()
    for thread in threading.enumerate():
        if thread.name.startswith("binance-"):
            join(thread, 3)
            assert not thread.is_alive()
print(json.dumps({"live_threads_at_failure": 4, "new_root": "REJECTED"}))
'''
    # The worker is a fresh process; an intentionally poisoned root cannot leak
    # into pytest. Only network boundaries and join duration are controlled.
    repository = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "recording")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(repository / "src"), str(repository))),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "live_threads_at_failure": 4,
        "new_root": "REJECTED",
    }


def test_invalid_payload_coverage_stops_at_captured_receive_clock(
    tmp_path, monkeypatch
):
    phase = {"value": "open"}
    stamps = {
        "open": "1970-01-01T00:00:00.000000Z",
        "receive": "1970-01-01T00:00:00.250000Z",
        "detect": "1970-01-01T00:00:00.350000Z",
    }
    monotonic = {"open": 1, "receive": 250_000_000, "detect": 350_000_000}

    class Websocket:
        def recv(self, timeout):
            phase["value"] = "receive"
            return b"{}"

        def close(self):
            pass

    def invalid(*args):
        phase["value"] = "detect"
        raise ValueError("synthetic_invalid_payload")

    monkeypatch.setattr(recorder, "_utc_now", lambda: stamps[phase["value"]])
    monkeypatch.setattr(
        recorder.time, "monotonic_ns", lambda: monotonic[phase["value"]]
    )
    monkeypatch.setattr(
        recorder, "_connect_websocket", lambda *args, **kwargs: Websocket()
    )
    monkeypatch.setattr(recorder, "validate_market_payload", invalid)
    with runtime_resource_policy_scope() as policy:
        writer = recorder._RunWriter(tmp_path)
        stats = recorder._CaptureStats(policy, writer)
        reader = recorder._StreamReader(
            stream=FUTURES_STREAMS[1],
            writer=writer,
            stats=stats,
            stop=threading.Event(),
            config=RecorderConfig(tmp_path / "unused", 1),
            resource_policy=policy,
            tick_size="0.1",
        )
        with pytest.raises(recorder._InvalidPayload):
            reader._read_connection(reconnect=False)
        writer.close()
        descriptor = writer.connection_interval_manifest(policy)
        stats.set_connection_interval_manifest(descriptor)
        stats.close()
        with gzip.open(tmp_path / descriptor["path"], "rt") as stream:
            interval = json.loads(next(stream))
        assert interval["closed_at"] == stamps["receive"]
        assert interval["closed_monotonic_ns"] == monotonic["receive"]
        assert phase["value"] == "detect"
        assert stats.invalid_messages == 1
        assert policy.used_memory_bytes == 0


@pytest.mark.parametrize("failure_at", ["close", "hash", "cancel"])
@pytest.mark.parametrize("body_fails", [False, True])
def test_writer_lifecycle_preserves_first_cleanup_or_body_failure(
    tmp_path, monkeypatch, failure_at, body_fails
):
    primary = ValueError("synthetic_body")
    secondary = (
        KeyboardInterrupt() if failure_at == "cancel" else OSError("synthetic_cleanup")
    )
    with runtime_resource_policy_scope() as policy:
        writer = recorder._RunWriter(tmp_path)
        stats = recorder._CaptureStats(policy, writer)
        original = writer.close

        def fail_close():
            original()
            raise secondary

        def fail_hash(*args):
            raise secondary

        if failure_at == "close":
            monkeypatch.setattr(writer, "close", fail_close)
        else:
            monkeypatch.setattr(recorder, "_sha256_path", fail_hash)
        with pytest.raises(BaseException) as caught:
            with recorder._writer_lifecycle(writer, policy, [], capture_stats=stats):
                if body_fails:
                    raise primary
        assert caught.value is (primary if body_fails else secondary)
        assert writer._closed
        assert policy.used_memory_bytes == 0
        if body_fails:
            assert any(type(secondary).__name__ in note for note in primary.__notes__)
