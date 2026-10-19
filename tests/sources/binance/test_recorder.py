from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

import quant_research_platform.sources.binance.recorder as recorder
from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    SPOT_MARKET_TYPE,
)
from quant_research_platform.sources.binance.recorder import (
    RECORDER_STREAMS_BY_SCOPE,
    RecorderConfig,
)
from quant_research_platform.sources.binance.validation import validate_market_payload

FUTURES_STREAMS = RECORDER_STREAMS_BY_SCOPE["futures-hft"]
SPOT_STREAMS = RECORDER_STREAMS_BY_SCOPE["spot-l2"]


def _depth_payload(market_type: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "e": "depthUpdate",
        "E": 1700000000001,
        "s": "BTCUSDT",
        "U": 10,
        "u": 12,
        "b": [["100.0", "1.0"]],
        "a": [["100.1", "2.0"]],
    }
    if market_type == FUTURES_MARKET_TYPE:
        payload.update({"T": 1700000000000, "pu": 9, "ps": "BTCUSDT", "st": 1})
    return payload


def _snapshot_payload(update_id, mid_tick=1000):
    return {
        "lastUpdateId": update_id,
        "E": 1700000000001,
        "T": 1700000000000,
        "bids": [[f"{(mid_tick - i) / 10:.1f}", "1.0"] for i in range(12)],
        "asks": [[f"{(mid_tick + 1 + i) / 10:.1f}", "2.0"] for i in range(12)],
    }


def _validate_stream(stream, payload):
    return validate_market_payload(
        stream.market_type, stream.data_kind, "BTCUSDT", payload
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
    assert (
        scopes["futures-hft"][0].url
        == "wss://fstream.binance.com/public/ws/btcusdt@depth@100ms"
    )
    assert (
        scopes["futures-hft"][1].url
        == "wss://fstream.binance.com/market/ws/btcusdt@aggTrade"
    )
    assert (
        scopes["spot-l2"][0].url
        == "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"
    )


def test_depth_validation_maps_large_json_integer_to_stable_owner_error() -> None:
    payload = _depth_payload(FUTURES_MARKET_TYPE)
    payload["b"] = [[10**4000, "1.0"]]
    with pytest.raises(ValueError, match="binance_recorder_depth_bids_invalid"):
        _validate_stream(FUTURES_STREAMS[0], payload)


def test_compressed_handle_cleanup_preserves_explicit_primary() -> None:
    class FailingClose:
        def __init__(self, message: str) -> None:
            self.message = message
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise OSError(self.message)

    raw_handle = FailingClose("raw_close")
    gzip_handle = FailingClose("gzip_close")
    primary = KeyboardInterrupt("body_primary")
    with pytest.raises(KeyboardInterrupt) as raised:
        try:
            raise primary
        except KeyboardInterrupt as body_error:
            recorder._close_compressed_handles(
                gzip_handle, raw_handle, primary=body_error
            )
            raise
    assert raised.value is primary
    assert gzip_handle.close_calls == 1
    assert raw_handle.close_calls == 1
    assert primary.__notes__ == [
        "Binance gzip handle close also failed: OSError",
        "Binance raw handle close also failed: OSError",
    ]


def test_stream_gzip_constructor_failure_closes_raw_and_preserves_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    def fail_gzip(*_args, **_kwargs):
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = KeyboardInterrupt("gzip_constructor_primary")
    raw_handle = (tmp_path / "tracked.bin").open("wb")
    monkeypatch.setattr(
        recorder, "open_filesystem_file", lambda *_args, **_kwargs: raw_handle
    )

    def fail_gzip(*_args, **_kwargs):
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
            first_raw, first_gzip, tmp_path / "first.gz"
        ),
        ("00", "market", "depth", "second"): recorder._OpenFile(
            second_raw, second_gzip, tmp_path / "second.gz"
        ),
    }
    with pytest.raises(OSError, match="first_gzip_close"):
        writer.close()
    assert (first_gzip.close_calls, first_raw.close_calls) == (1, 1)
    assert (second_gzip.close_calls, second_raw.close_calls) == (1, 1)
    assert writer._handles == {}
    assert writer._closed is True


def test_run_writer_does_not_query_free_disk(tmp_path, monkeypatch):
    import shutil

    def unexpected_query(*args, **kwargs):
        pytest.fail("disk capacity must not gate recording")

    monkeypatch.setattr(shutil, "disk_usage", unexpected_query)
    writer = recorder._RunWriter(tmp_path / "run")
    try:
        writer.write(
            symbol="BTCUSDT",
            market_type=FUTURES_MARKET_TYPE,
            data_kind="depth",
            stream="btcusdt@depth@100ms",
            payload={},
            receive_time="2026-08-27T00:00:00.000000Z",
            receive_monotonic_ns=1,
        )
        assert writer._capture_seq == 1
        assert next(iter(writer._files.values()))["record_count"] == 1
    finally:
        writer.abort()
    assert writer._closed
    assert writer._handles == {}


def test_recorder_validates_spot_and_futures_depth_semantics() -> None:
    spot, futures = (SPOT_STREAMS[0], FUTURES_STREAMS[0])
    assert _validate_stream(spot, _depth_payload(SPOT_MARKET_TYPE))["u"] == 12
    assert _validate_stream(futures, _depth_payload(FUTURES_MARKET_TYPE))["pu"] == 9
    with pytest.raises(ValueError, match="market_payload_identity_invalid"):
        _validate_stream(futures, {**_depth_payload(FUTURES_MARKET_TYPE), "st": 2})


def test_recorder_validates_futures_aggregate_trade_semantics() -> None:
    aggregate_trade = FUTURES_STREAMS[1]
    payload = {
        "e": "aggTrade",
        "E": 1700000000001,
        "T": 1700000000000,
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
            output_root=tmp_path / "run", duration_seconds=1, scope="mixed"
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
        return (_snapshot_payload(1), recorder._utc_now(), recorder.time.monotonic_ns())

    websocket = Websocket()
    monkeypatch.setattr(recorder, "_connect_websocket", lambda *_a, **_k: websocket)
    monkeypatch.setattr(recorder, "_http_get_json", snapshot)
    reader = recorder._StreamReader(
        stream=FUTURES_STREAMS[0],
        writer=Writer(),
        stats=recorder._CaptureStats(),
        stop=stop,
        config=RecorderConfig(Path("unused"), 1),
        tick_size="0.1",
    )
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
    assert [row["data_kind"] for row in writes] == ["depth_snapshot", "depth"]
    assert writes[0]["connection_id"] == writes[1]["connection_id"]
    assert writes[0]["segment_id"] == writes[1]["segment_id"]


def test_connection_interval_freezes_before_websocket_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
        recorder, "_connect_websocket", lambda *_args, **_kwargs: Websocket()
    )
    reader = recorder._StreamReader(
        stream=FUTURES_STREAMS[1],
        writer=recorder._RunWriter(tmp_path),
        stats=stats,
        stop=threading.Event(),
        config=RecorderConfig(tmp_path / "run", 1),
        tick_size="0.1",
    )
    with pytest.raises(recorder._ConnectionProblem):
        reader._read_connection(reconnect=False)
    assert order == ["interval", "websocket"]


def test_connection_stats_close_failure_still_closes_websocket(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:

    class Websocket:
        closed = False

        def recv(self, timeout):
            reader.stop.set()
            raise TimeoutError

        def close(self) -> None:
            self.closed = True

    websocket = Websocket()
    stats = recorder._CaptureStats()

    def fail_closed(*_args: object, **_kwargs: object) -> None:
        raise OSError("stats-close-primary")

    monkeypatch.setattr(stats, "closed", fail_closed)
    monkeypatch.setattr(
        recorder, "_connect_websocket", lambda *_args, **_kwargs: websocket
    )
    reader = recorder._StreamReader(
        stream=FUTURES_STREAMS[1],
        writer=recorder._RunWriter(tmp_path),
        stats=stats,
        stop=threading.Event(),
        config=RecorderConfig(tmp_path / "run", 1),
        tick_size="0.1",
    )
    with pytest.raises(OSError, match="stats-close-primary"):
        reader._read_connection(reconnect=False)
    assert websocket.closed


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


def test_capture_stats_streams_finalized_intervals_to_sidecar(tmp_path: Path) -> None:
    writer = recorder._RunWriter(tmp_path)
    stats = recorder._CaptureStats(writer)
    for index in range(10000):
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
    descriptor = writer.connection_interval_manifest()
    stats.set_connection_interval_manifest(descriptor)
    payload = stats.payload()
    sidecar = tmp_path / descriptor["path"]
    assert payload["connection_intervals"] == {
        "path": "connection_intervals.jsonl.gz",
        "record_count": 10000,
        "compressed_bytes": sidecar.stat().st_size,
        "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    }
    with gzip.open(sidecar, "rt", encoding="utf-8") as handle:
        assert sum(1 for _ in handle) == 10000


@pytest.mark.parametrize(
    "phase,interrupt_join,cleanup_failure",
    (
        ("reconnect_handshake", False, None),
        ("snapshot_headers", False, None),
        ("snapshot_body", False, None),
        ("snapshot_headers", True, None),
        ("snapshot_body", False, "websocket_close"),
        ("snapshot_body", False, "keepalive_join"),
    ),
)
def test_real_recorder_stops_owned_network_and_joins_before_return(
    tmp_path, monkeypatch, phase, interrupt_join, cleanup_failure
):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from websockets.sync.client import ClientConnection
    from websockets.sync.server import serve

    release = threading.Event()
    entered = threading.Event()
    server_threads = []
    readers = []
    clients = []
    connections = []
    handshakes = []
    original_reader = recorder._StreamReader
    original_join = original_reader.join
    original_start = original_reader.start
    join_interruptions = []
    keepalive_paused = threading.Event()
    release_keepalive = threading.Event()
    keepalive_joins = []
    cleanup_failures = []
    original_keepalive = ClientConnection.keepalive
    original_thread_join = threading.Thread.join

    def keepalive(client):
        original_keepalive(client)
        keepalive_paused.set()
        assert release_keepalive.wait(80)

    def join_thread(thread, timeout=None):
        if any(thread is client.keepalive_thread for client in clients):
            keepalive_joins.append(True)
            if cleanup_failure == "keepalive_join" and not cleanup_failures:
                cleanup_failures.append(True)
                raise KeyboardInterrupt("keepalive join interrupted")
            assert keepalive_paused.wait(3)
            release_keepalive.set()
        return original_thread_join(thread, timeout)

    def fail_websocket_close(client):
        del client
        cleanup_failures.append(True)
        raise OSError("websocket close failed")

    monkeypatch.setattr(ClientConnection, "keepalive", keepalive)
    monkeypatch.setattr(threading.Thread, "join", join_thread)
    if cleanup_failure == "websocket_close":
        monkeypatch.setattr(ClientConnection, "close", fail_websocket_close)

    def join_reader(reader, timeout=None):
        if interrupt_join and timeout is None and not join_interruptions:
            join_interruptions.append(True)
            raise KeyboardInterrupt("join interrupted")
        return original_join(reader, timeout)

    monkeypatch.setattr(original_reader, "join", join_reader)

    def start_reader(reader):
        original_start(reader)
        # Start recorder deadlines only after the target I/O phase is active.
        assert entered.wait(3)

    monkeypatch.setattr(original_reader, "start", start_reader)

    def make_reader(**kwargs):
        if phase != "reconnect_handshake":
            # Separate fixture handshake setup from the recorder readiness deadline.
            kwargs["config"] = replace(kwargs["config"], connect_timeout_seconds=3)
        reader = original_reader(**kwargs)
        readers.append(reader)
        return reader

    def make_connection(*args, **kwargs):
        client = ClientConnection(*args, **kwargs)
        clients.append(client)
        return client

    def request_gate(connection, request):
        del connection, request
        server_threads.append(threading.current_thread())
        handshakes.append(True)
        if phase == "reconnect_handshake" and len(handshakes) == 2:
            entered.set()
            assert release.wait(80)

    def websocket_handler(connection):
        connections.append(connection)
        if phase == "reconnect_handshake":
            connection.close()
        else:
            release.wait(80)

    class SnapshotHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            server_threads.append(threading.current_thread())
            try:
                if phase == "snapshot_headers":
                    # No response bytes: cancellation must reach the active socket.
                    entered.set()
                    release.wait(80)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", "1000000")
                    self.end_headers()
                    while not release.wait(0.02):
                        self.wfile.write(b" ")
                        self.wfile.flush()
                        entered.set()
            except (BrokenPipeError, ConnectionResetError):
                pass

    http_server = ThreadingHTTPServer(("127.0.0.1", 0), SnapshotHandler)
    http_thread = threading.Thread(target=http_server.serve_forever)
    http_thread.start()
    with serve(
        websocket_handler, "127.0.0.1", 0, process_request=request_gate, open_timeout=60
    ) as websocket_server:
        websocket_thread = threading.Thread(target=websocket_server.serve_forever)
        websocket_thread.start()
        original_stream = FUTURES_STREAMS[1 if phase == "reconnect_handshake" else 0]
        stream = replace(
            original_stream,
            url=f"ws://127.0.0.1:{websocket_server.socket.getsockname()[1]}/stream",
        )
        monkeypatch.setitem(
            recorder.RECORDER_STREAMS_BY_SCOPE, "futures-hft", (stream,)
        )
        monkeypatch.setattr(recorder, "_StreamReader", make_reader)
        monkeypatch.setattr(
            recorder, "ClientConnection", make_connection, raising=False
        )
        monkeypatch.setattr(
            recorder,
            "FUTURES_REST_BASE_URL",
            f"http://127.0.0.1:{http_server.server_port}",
        )
        monkeypatch.setattr(
            recorder,
            "_record_market_metadata",
            lambda *_a, **_k: {
                stream.symbol: {stream.market_type: {"tick_size": "0.1"}}
            },
        )
        output = tmp_path / "run"
        started = time.monotonic()
        returned_readers = []
        try:
            configuration = RecorderConfig(
                output,
                duration_seconds=1.3,
                connect_timeout_seconds=60 if phase == "reconnect_handshake" else 0.5,
            )
            if phase == "reconnect_handshake":
                result = recorder.record(configuration)
                assert result["status"] == "PARTIAL"
                assert (
                    json.loads((output / "run.json").read_text(encoding="utf-8"))
                    == result
                )
            else:
                with pytest.raises(
                    TimeoutError,
                    match="^binance_recorder_stream_connect_timeout(?:\n|$)",
                ) as raised:
                    recorder.record(configuration)
                assert str(raised.value) == "binance_recorder_stream_connect_timeout"
                if interrupt_join:
                    assert raised.value.__notes__ == [
                        "Binance reader join also failed: KeyboardInterrupt"
                    ]
                assert not output.exists()
            returned_readers = [reader for reader in readers if reader.is_alive()]
            assert entered.is_set()
            assert not returned_readers
            assert time.monotonic() - started < 5
            assert all(reader._active_websocket is None for reader in readers)
            assert len(clients) >= (2 if phase == "reconnect_handshake" else 1)
            assert all(not client.recv_events_thread.is_alive() for client in clients)
            assert keepalive_paused.is_set()
            assert keepalive_joins
            if cleanup_failure is not None:
                assert cleanup_failures == [True]
            if cleanup_failure == "keepalive_join":
                assert len(keepalive_joins) == 2
            assert all(
                client.keepalive_thread is None
                or not client.keepalive_thread.is_alive()
                for client in clients
            )
        finally:
            release.set()
            release_keepalive.set()
            for client in clients:
                if client.keepalive_thread is not None:
                    original_thread_join(client.keepalive_thread, 3)
            for connection in connections:
                connection.close()
            websocket_server.shutdown()
            websocket_thread.join(3)
            http_server.shutdown()
            http_thread.join(3)
            http_server.server_close()
            for thread in server_threads:
                thread.join(3)
            for reader in readers:
                reader.stop.set()
                reader.join(3)
            assert not websocket_thread.is_alive() and not http_thread.is_alive()
            assert all(not thread.is_alive() for thread in server_threads)
            assert all(not reader.is_alive() for reader in readers)


def test_connection_start_failure_closes_owner_and_skips_unstarted_keepalive(
    tmp_path, monkeypatch
):
    import socket

    from websockets.client import ClientProtocol
    from websockets.uri import parse_uri

    primary = RuntimeError("keepalive start failed")
    clients = []
    original_start = threading.Thread.start

    def start_thread(thread):
        if clients and thread is clients[0].keepalive_thread:
            raise primary
        return original_start(thread)

    client_socket, peer_socket = socket.socketpair()

    def connect(uri, **kwargs):
        client = kwargs["create_connection"](
            client_socket, ClientProtocol(parse_uri(uri))
        )
        clients.append(client)
        client.start_keepalive()
        raise AssertionError("keepalive start must fail")

    monkeypatch.setattr(threading.Thread, "start", start_thread)
    monkeypatch.setattr(recorder, "_connect_websocket", connect)
    reader = recorder._StreamReader(
        stream=FUTURES_STREAMS[1],
        writer=recorder._RunWriter(tmp_path),
        stats=recorder._CaptureStats(),
        stop=threading.Event(),
        config=RecorderConfig(tmp_path / "run", 1),
        tick_size="0.1",
    )
    try:
        with pytest.raises(RuntimeError, match="^keepalive start failed$") as raised:
            reader._read_connection(reconnect=False)
        assert raised.value is primary
        assert reader._active_websocket is None
        assert client_socket.fileno() == -1
        assert not clients[0].recv_events_thread.is_alive()
        assert clients[0].keepalive_thread.ident is None
        assert not clients[0].keepalive_thread.is_alive()
    finally:
        for client in clients:
            client.close_socket()
            client.recv_events_thread.join(3)
        client_socket.close()
        peer_socket.close()


def test_recorder_start_failure_joins_only_started_native_threads(
    tmp_path, monkeypatch
):
    readers = []
    primary = OSError("thread start failed")

    class Reader(recorder._StreamReader):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            readers.append(self)

        def start(self):
            if self is readers[1]:
                raise primary
            super().start()

        def _run_stream(self):
            self.ready.set()
            self.stop.wait()

    streams = RECORDER_STREAMS_BY_SCOPE["futures-hft"][:2]
    monkeypatch.setitem(recorder.RECORDER_STREAMS_BY_SCOPE, "futures-hft", streams)
    monkeypatch.setattr(recorder, "_StreamReader", Reader)
    monkeypatch.setattr(
        recorder,
        "_record_market_metadata",
        lambda *_a, **_k: {
            stream.symbol: {stream.market_type: {"tick_size": "0.1"}}
            for stream in streams
        },
    )
    output = tmp_path / "run"
    with pytest.raises(OSError, match="^thread start failed$") as raised:
        recorder.record(RecorderConfig(output, duration_seconds=0.001))
    assert raised.value is primary
    assert readers[0].finished.is_set() and not readers[0].is_alive()
    assert readers[1].ident is None and not readers[1].is_alive()
    assert not output.exists()


@pytest.mark.parametrize("late_failure", (False, True))
def test_recorder_publishes_only_after_reader_and_writer_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    late_failure: bool,
) -> None:
    output_root = tmp_path / "run"
    stream = FUTURES_STREAMS[1]

    class Reader:
        def __init__(self, **kwargs: object) -> None:
            self.name = "fake-reader"
            self.stream = kwargs["stream"]
            self.ready = threading.Event()
            self.finished = threading.Event()
            self.failure = None
            self.alive = True

        def start(self) -> None:
            self.ready.set()

        def cancel(self):
            self.finished.set()

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float) -> None:
            del timeout
            self.alive = False
            if late_failure:
                self.failure = OSError("final_write_failed")

    monkeypatch.setitem(recorder.RECORDER_STREAMS_BY_SCOPE, "futures-hft", (stream,))
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
        observed.append("json")
        return original_json(*args, **kwargs)

    def publish_directory(path, write):

        def staged(root):
            result = write(root)
            observed.append("publish")
            return result

        result = original_directory(path, staged)
        return result

    monkeypatch.setattr(recorder, "atomic_write_json", write_json)
    monkeypatch.setattr(recorder, "atomic_write_directory", publish_directory)
    if late_failure:
        with pytest.raises(OSError, match="final_write_failed"):
            recorder.record(RecorderConfig(output_root, duration_seconds=0.001))
        assert not output_root.exists()
    else:
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


def test_recorder_abort_failure_preserves_reader_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_root = tmp_path / "run"
    stream = FUTURES_STREAMS[1]

    class Reader:
        def __init__(self, **kwargs: object) -> None:
            self.name = "fake-reader"
            self.stream = kwargs["stream"]
            self.ready = threading.Event()
            self.finished = threading.Event()
            self.failure = RuntimeError("reader_failure")
            self.alive = True

        def start(self) -> None:
            self.ready.set()

        def cancel(self):
            self.finished.set()

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float) -> None:
            del timeout
            self.alive = False
            raise OSError("join-secondary")

    def abort(_writer: object) -> None:
        raise OSError("abort-secondary")

    monkeypatch.setitem(recorder.RECORDER_STREAMS_BY_SCOPE, "futures-hft", (stream,))
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


@pytest.mark.parametrize("failure_type", [TimeoutError, OSError])
def test_snapshot_network_failure_preserves_cause_and_closes_connection(
    tmp_path, monkeypatch, failure_type
):
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
        tick_size="0.1",
    )
    with pytest.raises(recorder._ConnectionProblem) as raised:
        reader._read_connection(reconnect=False)
    assert raised.value.__cause__ is primary
    assert "Binance websocket close also failed: OSError" in raised.value.__notes__
    assert websocket.closed
    assert not reader.ready.is_set()
    assert stats.connection_intervals == {}


@pytest.mark.parametrize(
    "failure", ["structure", "identity", "clock", "disk", "unknown"]
)
def test_snapshot_nonlocal_failures_remain_fatal(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace

    attempts = []
    payload = _snapshot_payload(100)
    received = recorder._utc_now()
    if failure == "structure":
        payload["asks"] = [["100.1"]]
    elif failure == "identity":
        payload["lastUpdateId"] = -1
    elif failure == "clock":
        received = "not-a-clock"
    writer = recorder._RunWriter(tmp_path)
    primary = (
        OSError("synthetic_disk_error")
        if failure == "disk"
        else ValueError("binance_replay_snapshot_unrecognized_failure")
    )

    def fail(*_args, **_kwargs):
        raise primary

    def connect(*_args, **_kwargs):
        attempts.append(1)
        return SimpleNamespace(close=lambda: None)

    if failure == "disk":
        monkeypatch.setattr(writer, "write", fail)
    elif failure == "unknown":
        monkeypatch.setattr(recorder.DepthReconstructor, "bootstrap", fail)
    monkeypatch.setattr(recorder, "_connect_websocket", connect)
    monkeypatch.setattr(
        recorder,
        "_http_get_json",
        lambda *_a, **_k: (payload, received, recorder.time.monotonic_ns()),
    )
    stats = recorder._CaptureStats()
    reader = recorder._StreamReader(
        stream=FUTURES_STREAMS[0],
        writer=writer,
        stats=stats,
        stop=threading.Event(),
        config=RecorderConfig(tmp_path / "run", 1),
        tick_size="0.1",
    )
    try:
        reader._run_stream()
    finally:
        writer.close()
    assert len(attempts) == 1
    assert isinstance(reader.failure, (ValueError, OSError))
    if failure in {"disk", "unknown"}:
        assert reader.failure is primary
    assert stats.connection_intervals == {}
    assert stats.connection_errors == 0


def test_persistent_bad_snapshot_uses_bounded_backoff(tmp_path, monkeypatch):
    from types import SimpleNamespace

    class Stop:
        def __init__(self):
            self.waits = []

        def is_set(self):
            return len(self.waits) == 4

        def wait(self, seconds):
            self.waits.append(seconds)

    stop = Stop()
    payload = _snapshot_payload(100)
    payload["asks"][-1][1] = "-1"
    monkeypatch.setattr(
        recorder,
        "_connect_websocket",
        lambda *_a, **_k: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        recorder,
        "_http_get_json",
        lambda *_a, **_k: (payload, recorder._utc_now(), recorder.time.monotonic_ns()),
    )
    writer = recorder._RunWriter(tmp_path)
    stats = recorder._CaptureStats(writer=writer)
    reader = recorder._StreamReader(
        stream=FUTURES_STREAMS[0],
        writer=writer,
        stats=stats,
        stop=stop,
        config=RecorderConfig(tmp_path / "run", 1, reconnect_max_seconds=3),
        tick_size="0.1",
    )
    try:
        reader._run_stream()
    finally:
        writer.close()
    assert reader.failure is None
    assert stop.waits == [1, 2, 3, 3]
    assert stats.connection_errors == 4
    assert stats.connection_intervals == {}
    assert writer.connection_interval_manifest()["record_count"] == 4
    assert sum(item["record_count"] for item in writer.file_manifests()) == 4


@pytest.mark.parametrize(
    "bad_snapshot",
    [None, "invalid_numeric", "one_sided_book", "crossed_book", "unsupported_coverage"],
)
def test_recorder_recovers_exhausted_known_depth_with_new_connection_and_snapshot(
    tmp_path, monkeypatch, bad_snapshot
):
    from collections import Counter
    from types import SimpleNamespace

    from quant_research_platform.contracts.order_book import BookBoundary
    from quant_research_platform.sources.binance.replay import (
        ReconstructionStats,
        iter_reconstructed_depth,
        open_recorder_run,
    )

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
            and (kwargs["payload"]["u"] == 102)
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

    def http(_base, endpoint, *, params=None, **_kwargs):
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
            if symbol == "BTCUSDT" and snapshots[symbol] == 2:
                if bad_snapshot == "invalid_numeric":
                    payload["asks"][-1][1] = "-1"
                elif bad_snapshot == "one_sided_book":
                    payload["asks"] = []
                elif bad_snapshot == "crossed_book":
                    payload["asks"][0][0] = "89.0"
                elif bad_snapshot == "unsupported_coverage":
                    payload["asks"] = payload["asks"][:1]
        return (payload, recorder._utc_now(), recorder.time.monotonic_ns())

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
            if symbol == "BTCUSDT" and (not recovered):
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
                    "E": 1700000000001,
                    "T": 1700000000000,
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
    root = tmp_path / "recording"
    result = recorder.record(
        RecorderConfig(root, duration_seconds=0.2, reconnect_max_seconds=0.001)
    )
    assert recovered_recorded.is_set()
    assert result["status"] == "PARTIAL"
    assert result["capture"]["sequence_gaps"] == 0
    expected_segments = 3 if bad_snapshot else 2
    assert connections == {"BTCUSDT": expected_segments, "ETHUSDT": 1}
    assert result["capture"]["connection_errors"] == int(bad_snapshot is not None)
    assert snapshots == connections
    assert all(socket.closed.is_set() for socket in sockets)
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
        assert len({r["connection_id"] for r in snapshot_records}) == expected_segments
        assert len({r["segment_id"] for r in snapshot_records}) == expected_segments
        if bad_snapshot == "invalid_numeric":
            assert snapshot_records[1]["payload"]["asks"][-1][1] == "-1"
        for symbol in ("BTCUSDT", "ETHUSDT"):
            assert (
                len(
                    list(
                        run.iter_records(
                            symbol=symbol,
                            market_type=FUTURES_MARKET_TYPE,
                            data_kinds=("agg_trades",),
                        )
                    )
                )
                == 1
            )
        assert (
            len(
                list(
                    run.iter_records(
                        symbol="ETHUSDT",
                        market_type=FUTURES_MARKET_TYPE,
                        data_kinds=("depth_snapshot", "depth"),
                    )
                )
            )
            == 2
        )
        for event in iter_reconstructed_depth(
            run, FUTURES_MARKET_TYPE, symbol="BTCUSDT", stats=stats
        ):
            if isinstance(event, BookBoundary):
                boundaries.append(event.reason)
            else:
                states.append(event.update_id)
    assert states == [100, 102]
    assert boundaries == ["one_sided_book", "connection_closed"]
    assert stats.valid_segments == 2
    assert stats.recovered_segments == 1
    assert stats.one_sided_books == 1 + (bad_snapshot == "one_sided_book")
    assert len(stats.coverage_segments) == 2


def test_invalid_payload_coverage_stops_at_captured_receive_clock(
    tmp_path, monkeypatch
):
    phase = {"value": "open"}
    stamps = {
        "open": "1970-01-01T00:00:00.000000Z",
        "receive": "1970-01-01T00:00:00.250000Z",
        "detect": "1970-01-01T00:00:00.350000Z",
    }
    monotonic = {"open": 1, "receive": 250000000, "detect": 350000000}

    class Websocket:
        def recv(self, timeout):
            phase["value"] = "receive"
            return b"{}"

        def close(self):
            pass

    def invalid(*args, validate_values):
        assert validate_values is False
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
    writer = recorder._RunWriter(tmp_path)
    stats = recorder._CaptureStats(writer)
    reader = recorder._StreamReader(
        stream=FUTURES_STREAMS[1],
        writer=writer,
        stats=stats,
        stop=threading.Event(),
        config=RecorderConfig(tmp_path / "unused", 1),
        tick_size="0.1",
    )
    with pytest.raises(recorder._InvalidPayload):
        reader._read_connection(reconnect=False)
    writer.close()
    descriptor = writer.connection_interval_manifest()
    stats.set_connection_interval_manifest(descriptor)
    with gzip.open(tmp_path / descriptor["path"], "rt") as stream:
        interval = json.loads(next(stream))
    assert interval["closed_at"] == stamps["receive"]
    assert interval["closed_monotonic_ns"] == monotonic["receive"]
    assert phase["value"] == "detect"
    assert stats.invalid_messages == 1


@pytest.mark.parametrize("failure_at", ["close", "hash"])
@pytest.mark.parametrize("body_fails", [False, True])
def test_writer_lifecycle_preserves_first_cleanup_or_body_failure(
    tmp_path, monkeypatch, failure_at, body_fails
):
    primary = ValueError("synthetic_body")
    secondary = (
        KeyboardInterrupt() if failure_at == "hash" else OSError("synthetic_cleanup")
    )
    writer = recorder._RunWriter(tmp_path)
    stats = recorder._CaptureStats(writer)
    original = writer.close

    def fail_close():
        original()
        raise secondary

    def fail_hash(*args):
        raise secondary

    if failure_at == "close":
        monkeypatch.setattr(writer, "close", fail_close)
    else:
        monkeypatch.setattr(recorder, "_hash_file_content", fail_hash)
    with pytest.raises(BaseException) as caught:
        with recorder._writer_lifecycle(writer, [], capture_stats=stats):
            if body_fails:
                raise primary
    assert caught.value is (primary if body_fails else secondary)
    assert writer._closed
    if body_fails:
        assert any(type(secondary).__name__ in note for note in primary.__notes__)


@pytest.mark.parametrize("set_ready", [False, True])
def test_recorder_preserves_reader_business_exception(tmp_path, monkeypatch, set_ready):
    primary = KeyboardInterrupt("reader-primary")
    readers = []
    stream = FUTURES_STREAMS[1]

    class FailingReader(recorder._StreamReader):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            readers.append(self)

        def _read_connection(self, *, reconnect):
            if set_ready:
                self.ready.set()
            raise primary

    monkeypatch.setitem(recorder.RECORDER_STREAMS_BY_SCOPE, "futures-hft", (stream,))
    monkeypatch.setattr(recorder, "_StreamReader", FailingReader)
    monkeypatch.setattr(
        recorder,
        "_record_market_metadata",
        lambda *_a, **_k: {stream.symbol: {stream.market_type: {"tick_size": "0.1"}}},
    )
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt) as raised:
        recorder.record(RecorderConfig(output, duration_seconds=0.01))
    assert raised.value is primary
    assert readers and all(not reader.is_alive() for reader in readers)
    assert all(reader.failure is primary for reader in readers)
    assert not output.exists()


@pytest.mark.parametrize("abort_fails", [False, True])
def test_join_primary_survives_all_reader_cleanup_and_abort(
    tmp_path, monkeypatch, abort_fails
):
    primary = KeyboardInterrupt("join-primary")
    readers = []
    joined = []
    aborted = []
    streams = RECORDER_STREAMS_BY_SCOPE["futures-hft"]

    class Reader:
        def __init__(self, **kwargs):
            self.name = f"reader-{len(readers)}"
            self.stream = kwargs["stream"]
            self.stop = kwargs["stop"]
            self.ready = threading.Event()
            self.finished = threading.Event()
            self.failure = None
            self.alive = True
            readers.append(self)

        def start(self):
            self.ready.set()

        def cancel(self):
            self.finished.set()

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            assert timeout is None
            assert self.stop.is_set()
            joined.append(self)
            self.alive = False
            if self is readers[0]:
                raise primary
            if self is readers[-1]:
                self.failure = ValueError("late-reader-secondary")
            raise OSError("join-secondary")

    original_abort = recorder._RunWriter.abort

    def abort(writer):
        aborted.append(True)
        original_abort(writer)
        if abort_fails:
            raise OSError("abort-secondary")

    monkeypatch.setattr(recorder, "_StreamReader", Reader)
    monkeypatch.setattr(recorder._RunWriter, "abort", abort)
    monkeypatch.setattr(
        recorder,
        "_record_market_metadata",
        lambda *_a, **_k: {
            stream.symbol: {stream.market_type: {"tick_size": "0.1"}}
            for stream in streams
        },
    )
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt) as raised:
        recorder.record(RecorderConfig(output, duration_seconds=0.001))
    assert raised.value is primary
    assert len(readers) == len(streams) > 1
    assert joined == readers
    assert aborted == [True]
    assert primary.__notes__ == [
        *["Binance reader join also failed: OSError"] * (len(readers) - 1),
        "Binance reader also failed: ValueError",
        *(["Binance run writer abort also failed: OSError"] if abort_fails else []),
    ]
    assert not output.exists()


def test_compressed_close_surfaces_failure_inside_handled_outer():
    closed = []

    class Handle:
        def close(self):
            closed.append(True)
            raise OSError("close_failure")

    try:
        raise RuntimeError("handled_outer")
    except RuntimeError as outer:
        with pytest.raises(OSError, match="close_failure") as caught:
            recorder._close_compressed_handles(Handle(), Handle())
        assert not getattr(outer, "__notes__", [])
    assert closed == [True, True]
    assert caught.value.__notes__ == ["Binance raw handle close also failed: OSError"]


@pytest.mark.parametrize("failure", ("none", "read", "close", "both"))
def test_hash_path_preserves_read_failure_and_closes_native_handle(
    tmp_path, monkeypatch, failure
):
    data = b"synthetic-digest\x00" * 8192
    path = tmp_path / "member.bin"
    path.write_bytes(data)
    from quant_research_platform.storage import identity

    original = identity.open_filesystem_file
    primary = OSError("synthetic_read")
    secondary = RuntimeError("synthetic_close")
    opened = []

    class Handle:
        def __init__(self, native):
            self.native = native
            self.close_calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def read(self, size):
            if failure in {"read", "both"}:
                raise primary
            return self.native.read(size)

        def close(self):
            self.close_calls += 1
            self.native.close()
            if failure in {"close", "both"}:
                raise secondary

    def observed_open(*args, **kwargs):
        handle = Handle(original(*args, **kwargs))
        opened.append(handle)
        return handle

    monkeypatch.setattr(identity, "open_filesystem_file", observed_open)
    if failure == "none":
        assert recorder._hash_file_content(path) == hashlib.sha256(data).hexdigest()
    else:
        with pytest.raises(BaseException) as caught:
            recorder._hash_file_content(path)
        assert caught.value is (primary if failure in {"read", "both"} else secondary)
        if failure == "both":
            assert primary.__notes__ == ["iterator close also failed: RuntimeError"]
    assert len(opened) == 1
    assert opened[0].close_calls == 1
    assert opened[0].native.closed
    assert path.read_bytes() == data


def test_recorder_hash_read_and_close_failure_aborts_native_atomic_publish(
    tmp_path, monkeypatch
):
    from quant_research_platform.storage import identity

    original = identity.open_filesystem_file
    primary = OSError("synthetic_hash_read")
    secondary = RuntimeError("synthetic_hash_close")
    handles = []

    class Handle:
        def __init__(self, native):
            self.native = native

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def read(self, _size):
            raise primary

        def close(self):
            self.native.close()
            raise secondary

    def observed_open(path, mode="rb", **kwargs):
        native = original(path, mode, **kwargs)
        if mode != "rb":
            return native
        handle = Handle(native)
        handles.append(handle)
        return handle

    def write(staging):
        writer = recorder._RunWriter(staging)
        stats = recorder._CaptureStats(writer)
        with recorder._writer_lifecycle(writer, [], capture_stats=stats):
            pass

    monkeypatch.setattr(identity, "open_filesystem_file", observed_open)
    with pytest.raises(OSError) as caught:
        recorder.atomic_write_directory(tmp_path / "capture", write)
    assert caught.value is primary
    assert primary.__notes__ == ["iterator close also failed: RuntimeError"]
    assert handles and all(handle.native.closed for handle in handles)
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("padding", [0, 4 * 1024 * 1024])
def test_metadata_rest_size_does_not_change_semantics(monkeypatch, padding):
    from types import SimpleNamespace

    import httpx

    payload = {
        "symbols": [
            {
                "symbol": symbol,
                "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.1"}],
            }
            for symbol in ("BTCUSDT", "ETHUSDT")
        ]
    }
    raw = b" " * padding + json.dumps(payload).encode()
    client = httpx.Client
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, stream=httpx.ByteStream(raw))
    )
    monkeypatch.setattr(recorder.httpx, "Client", lambda: client(transport=transport))
    writes = []
    result = recorder._record_market_metadata(
        SimpleNamespace(write=lambda **row: writes.append(row)),
        market_types=(FUTURES_MARKET_TYPE,),
        symbols=("BTCUSDT", "ETHUSDT"),
    )
    assert len(writes) == 2
    assert result == {
        symbol: {FUTURES_MARKET_TYPE: {"tick_size": "0.1", "snapshot_limit": 1000}}
        for symbol in ("BTCUSDT", "ETHUSDT")
    }


def test_websocket_large_legal_message_and_stop_close():
    from types import SimpleNamespace

    from websockets.exceptions import ConnectionClosed
    from websockets.sync.server import serve

    stop = threading.Event()
    closed = threading.Event()
    writes = []
    payload = {
        "e": "aggTrade",
        "E": 1700000000001,
        "T": 1700000000000,
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

    def receive(**row):
        writes.append(row)
        stop.set()
        return len(writes)

    def handler(connection):
        connection.send(" " * (4 * 1024 * 1024) + json.dumps(payload))
        try:
            connection.recv()
        except ConnectionClosed:
            closed.set()

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        stream = replace(
            FUTURES_STREAMS[1],
            url=f"ws://127.0.0.1:{server.socket.getsockname()[1]}/stream",
        )
        reader = recorder._StreamReader(
            stream=stream,
            writer=SimpleNamespace(write=receive),
            stats=recorder._CaptureStats(),
            stop=stop,
            config=RecorderConfig(Path("unused"), 1),
            tick_size="0.1",
        )
        try:
            reader._read_connection(reconnect=False)
            assert len(writes) == 1
            assert writes[0]["payload"] == payload
            assert reader._active_websocket is None
            assert closed.wait(3)
        finally:
            stop.set()
            reader.cancel()
            server.shutdown()
            thread.join(3)
        assert not thread.is_alive()
