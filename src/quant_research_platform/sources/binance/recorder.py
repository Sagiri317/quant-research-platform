"""Read-only Binance BTCUSDT Spot and USDⓈ-M recorder."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import psutil
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect as _connect_websocket

from quant_research_platform.contracts.json_codec import (
    parse_qrp_json_object,
)
from quant_research_platform.contracts.order_book import BookBoundary
from quant_research_platform.contracts.validation import (
    strict_finite_float,
)
from quant_research_platform.runtime.resources import (
    require_free_disk_headroom,
)
from quant_research_platform.sources.binance.ids import (
    FUTURES_HFT_RECORDING_SCOPE,
    FUTURES_MARKET_TYPE,
    RECORDING_SCOPE_MARKET_TYPES,
    RESEARCH_SYMBOLS,
    SPOT_L2_RECORDING_SCOPE,
    SPOT_MARKET_TYPE,
    WS_STREAM_SUFFIXES,
)
from quant_research_platform.sources.binance.replay import (
    RECONSTRUCTION_MAX_BOOK_LEVELS,
    RECONSTRUCTION_REQUIRED_LEVELS,
    DepthReconstructor,
    ReconstructionStats,
    normalize_recorder_record,
)
from quant_research_platform.sources.binance.validation import (
    strict_iso_timestamp,
    strict_non_negative_integer,
    strict_positive_float,
    validate_depth_update_levels,
    validate_market_payload,
)
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_mkdir,
    filesystem_stat,
    open_filesystem_file,
    require_unaliased_path,
)

SPOT_REST_BASE_URL = "https://data-api.binance.vision"
FUTURES_REST_BASE_URL = "https://fapi.binance.com"
MAX_WS_MESSAGE_BYTES = 4 * 1024 * 1024
MAX_WS_QUEUE = 4
MAX_REST_RESPONSE_BYTES = 4 * 1024 * 1024
REST_REQUEST_TIMEOUT_SECONDS = 15.0
SNAPSHOT_LIMITS = {SPOT_MARKET_TYPE: 5000, FUTURES_MARKET_TYPE: 1000}
_RECORDER_DISK_CHECK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class RecorderStream:
    symbol: str
    market_type: str
    data_kind: str
    stream: str
    url: str


RECORDER_STREAMS_BY_SCOPE = {
    FUTURES_HFT_RECORDING_SCOPE: tuple(
        RecorderStream(
            symbol,
            FUTURES_MARKET_TYPE,
            data_kind,
            f"{symbol.lower()}@{WS_STREAM_SUFFIXES[data_kind]}",
            f"wss://fstream.binance.com/{channel}/ws/{symbol.lower()}@{WS_STREAM_SUFFIXES[data_kind]}",
        )
        for symbol in RESEARCH_SYMBOLS
        for data_kind, channel in (
            ("depth", "public"),
            ("agg_trades", "market"),
        )
    ),
    SPOT_L2_RECORDING_SCOPE: tuple(
        RecorderStream(
            symbol,
            SPOT_MARKET_TYPE,
            "depth",
            f"{symbol.lower()}@{WS_STREAM_SUFFIXES['depth']}",
            f"wss://stream.binance.com:9443/ws/{symbol.lower()}@{WS_STREAM_SUFFIXES['depth']}",
        )
        for symbol in RESEARCH_SYMBOLS
    ),
}


@dataclass(frozen=True)
class RecorderConfig:
    output_root: Path
    duration_seconds: float
    connect_timeout_seconds: float = 20.0
    reconnect_max_seconds: float = 30.0
    connection_idle_timeout_seconds: float = 30.0
    scope: str = FUTURES_HFT_RECORDING_SCOPE
    symbols: tuple[str, ...] = RESEARCH_SYMBOLS

    def validate(self) -> None:
        if type(self.scope) is not str or self.scope not in RECORDER_STREAMS_BY_SCOPE:
            raise ValueError("binance_recorder_scope_invalid")
        if self.symbols != RESEARCH_SYMBOLS:
            raise ValueError("binance_recorder_symbols_invalid")
        _positive_number(
            self.duration_seconds,
            error="binance_recorder_duration_invalid",
        )
        for value in (
            self.connect_timeout_seconds,
            self.reconnect_max_seconds,
            self.connection_idle_timeout_seconds,
        ):
            _positive_number(value, error="binance_recorder_timeout_invalid")
        root = Path(self.output_root)
        require_unaliased_path(root, error="invalid_storage_write_path")
        if filesystem_exists(root):
            raise FileExistsError("binance_recorder_output_exists")


def _positive_number(value: object, *, error: str) -> float:
    number = strict_finite_float(value, error=error)
    if number <= 0:
        raise ValueError(error)
    return number


def _utc_now() -> str:
    epoch_ns = time.time_ns()
    seconds, nanoseconds = divmod(epoch_ns, 1_000_000_000)
    stamp = dt.datetime.fromtimestamp(seconds, dt.UTC)
    return f"{stamp:%Y-%m-%dT%H:%M:%S}.{nanoseconds:09d}Z"


@dataclass
class _OpenFile:
    raw_handle: Any
    gzip_handle: Any
    path: Path


def _close_compressed_handles(gzip_handle: Any, raw_handle: Any) -> None:
    primary = sys.exception()
    failure: BaseException | None = None
    for handle, label in (
        (gzip_handle, "Binance gzip handle close also failed"),
        (raw_handle, "Binance raw handle close also failed"),
    ):
        if handle is None:
            continue
        try:
            handle.close()
        except BaseException as close_error:
            note = f"{label}: {type(close_error).__name__}"
            if primary is not None:
                primary.add_note(note)
            elif failure is None:
                failure = close_error
            else:
                failure.add_note(note)
    if failure is not None:
        raise failure


@dataclass
class _RunWriter:
    root: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _closed: bool = field(default=False, init=False)
    _aborted: bool = field(default=False, init=False)
    _capture_seq: int = field(default=0, init=False)
    _handles: dict[tuple[str, str, str, str, str], _OpenFile] = field(
        default_factory=dict,
        init=False,
    )
    _files: dict[Path, dict[str, Any]] = field(default_factory=dict, init=False)
    _event_counts: Counter[str] = field(default_factory=Counter, init=False)
    _interval_path: Path | None = field(default=None, init=False, repr=False)
    _interval_raw_handle: Any | None = field(default=None, init=False, repr=False)
    _interval_gzip_handle: Any | None = field(default=None, init=False, repr=False)
    _interval_record_count: int = field(default=0, init=False, repr=False)
    _disk_bytes_remaining: int = field(default=0, init=False, repr=False)

    def write(
        self,
        *,
        symbol: str,
        market_type: str,
        data_kind: str,
        stream: str,
        payload: object,
        receive_time: str,
        receive_monotonic_ns: int,
        connection_id: str | None = None,
        segment_id: str | None = None,
    ) -> int:
        observed_time_ns = strict_iso_timestamp(
            receive_time,
            error="binance_recorder_receive_time_invalid",
        ).value
        strict_non_negative_integer(
            receive_monotonic_ns,
            error="binance_recorder_receive_monotonic_invalid",
        )
        with self._lock:
            self._require_open()
            capture_seq = self._capture_seq + 1
            record: dict[str, Any] = {
                "capture_seq": capture_seq,
                "receive_time": receive_time,
                "observed_time_ns": observed_time_ns,
                "receive_monotonic_ns": receive_monotonic_ns,
                "payload": payload,
            }
            if connection_id is not None:
                record["connection_id"] = connection_id
            if segment_id is not None:
                record["segment_id"] = segment_id
            line = (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            self._admit_disk_growth(len(line))
            self._capture_seq = capture_seq
            hour = receive_time[:13]
            for key in tuple(self._handles):
                if key[0] != hour:
                    self._close_key(key)
            key = (hour, symbol, market_type, data_kind, stream)
            opened = self._handles.get(key)
            if opened is None:
                opened = self._open_file(
                    key,
                    symbol=symbol,
                    market_type=market_type,
                    data_kind=data_kind,
                    stream=stream,
                    receive_time=receive_time,
                )
            opened.gzip_handle.write(line)
            metadata = self._files[opened.path]
            metadata["record_count"] += 1
            metadata["capture_seq_min"] = (
                capture_seq
                if metadata["capture_seq_min"] is None
                else min(metadata["capture_seq_min"], capture_seq)
            )
            metadata["capture_seq_max"] = capture_seq
            metadata["max_record_bytes"] = max(
                metadata["max_record_bytes"],
                len(line),
            )
            self._event_counts[f"{symbol}:{market_type}:{data_kind}"] += 1
            if metadata["record_count"] % 256 == 0:
                opened.gzip_handle.flush()
                opened.raw_handle.flush()
            return capture_seq

    def _open_file(
        self,
        key: tuple[str, str, str, str, str],
        *,
        symbol: str,
        market_type: str,
        data_kind: str,
        stream: str,
        receive_time: str,
    ) -> _OpenFile:
        directory = (
            self.root / f"date={receive_time[:10]}" / f"hour={receive_time[11:13]}"
        )
        filesystem_mkdir(directory, parents=True, exist_ok=True)
        path = directory / f"{symbol}_{market_type}_{data_kind}.jsonl.gz"
        raw_handle = open_filesystem_file(path, "ab")
        try:
            gzip_handle = gzip.GzipFile(
                fileobj=raw_handle,
                mode="wb",
                compresslevel=6,
            )
        except BaseException:
            _close_compressed_handles(None, raw_handle)
            raise
        opened = _OpenFile(raw_handle, gzip_handle, path)
        self._handles[key] = opened
        self._files.setdefault(
            path,
            {
                "path": path.relative_to(self.root).as_posix(),
                "venue": "binance",
                "symbol": symbol,
                "market": market_type,
                "stream": stream,
                "data_kind": data_kind,
                "record_count": 0,
                "capture_seq_min": None,
                "capture_seq_max": None,
                "max_record_bytes": 0,
            },
        )
        return opened

    def write_connection_interval(self, interval: dict[str, Any]) -> None:
        with self._lock:
            self._require_open()
            line = (
                json.dumps(
                    interval,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            self._admit_disk_growth(len(line))
            self._open_interval_file()
            assert self._interval_gzip_handle is not None
            self._interval_gzip_handle.write(line)
            self._interval_record_count += 1

    def _open_interval_file(self) -> None:
        if self._interval_gzip_handle is not None:
            return
        path = self.root / "connection_intervals.jsonl.gz"
        raw_handle = open_filesystem_file(path, "wb")
        try:
            gzip_handle = gzip.GzipFile(
                fileobj=raw_handle,
                mode="wb",
                compresslevel=6,
            )
        except BaseException:
            _close_compressed_handles(None, raw_handle)
            raise
        self._interval_raw_handle = raw_handle
        self._interval_gzip_handle = gzip_handle
        self._interval_path = path

    def _admit_disk_growth(self, logical_bytes: int) -> None:
        if logical_bytes <= self._disk_bytes_remaining:
            self._disk_bytes_remaining -= logical_bytes
            return
        admitted = max(_RECORDER_DISK_CHECK_BYTES, logical_bytes)
        require_free_disk_headroom(self.root, reserve_bytes=2 * admitted)
        self._disk_bytes_remaining = admitted - logical_bytes

    def _close_key(self, key: tuple[str, str, str, str, str]) -> None:
        opened = self._handles.pop(key, None)
        if opened is None:
            return
        _close_compressed_handles(opened.gzip_handle, opened.raw_handle)

    def _close_stream_handles(self) -> BaseException | None:
        failure: BaseException | None = None
        for key in tuple(self._handles):
            try:
                self._close_key(key)
            except BaseException as close_error:
                if failure is None:
                    failure = close_error
                else:
                    failure.add_note(
                        "additional Binance stream close also failed: "
                        f"{type(close_error).__name__}"
                    )
        return failure

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            failure = self._close_stream_handles()
            try:
                if self._interval_gzip_handle is None:
                    self._admit_disk_growth(1)
                self._open_interval_file()
                assert self._interval_gzip_handle is not None
                assert self._interval_raw_handle is not None
                _close_compressed_handles(
                    self._interval_gzip_handle,
                    self._interval_raw_handle,
                )
            except BaseException as close_error:
                if failure is None:
                    failure = close_error
                else:
                    failure.add_note(
                        "Binance interval handle close also failed: "
                        f"{type(close_error).__name__}"
                    )
            self._closed = True
            if failure is not None:
                raise failure

    def abort(self) -> None:
        with self._lock:
            if self._closed:
                self._aborted = True
                return
            failure = self._close_stream_handles()
            try:
                if self._interval_gzip_handle is not None:
                    assert self._interval_raw_handle is not None
                    _close_compressed_handles(
                        self._interval_gzip_handle,
                        self._interval_raw_handle,
                    )
            except BaseException as close_error:
                if failure is None:
                    failure = close_error
                else:
                    failure.add_note(
                        "Binance interval handle close also failed: "
                        f"{type(close_error).__name__}"
                    )
            self._closed = True
            self._aborted = True
            if failure is not None:
                raise failure

    def connection_interval_manifest(self) -> dict[str, Any]:
        with self._lock:
            if not self._closed:
                raise RuntimeError("binance_recorder_writer_not_closed")
            assert self._interval_path is not None
            path = self._interval_path
            digest = _sha256_path(path)
            return {
                "path": path.relative_to(self.root).as_posix(),
                "record_count": self._interval_record_count,
                "compressed_bytes": filesystem_stat(path).st_size,
                "sha256": digest,
            }

    def file_manifests(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self._closed or self._handles:
                raise RuntimeError("binance_recorder_writer_not_closed")
            return [
                {
                    **self._files[path],
                    "compressed_bytes": filesystem_stat(path).st_size,
                    "sha256": _sha256_path(path),
                }
                for path in sorted(self._files)
            ]

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("binance_recorder_writer_closed")

    def event_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(sorted(self._event_counts.items()))


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    buffer = bytearray(64 * 1024)
    with open_filesystem_file(path, "rb") as handle:
        while count := handle.readinto(buffer):
            digest.update(memoryview(buffer)[:count])
    return digest.hexdigest()


@contextmanager
def _writer_lifecycle(
    writer: _RunWriter,
    files: list[dict[str, Any]],
    capture_stats: _CaptureStats | None = None,
) -> Iterator[None]:
    try:
        yield
    finally:
        primary = sys.exception()
        failure = None
        try:
            if not writer._aborted:  # noqa: SLF001
                writer.close()
                files.extend(writer.file_manifests())
                if capture_stats is not None:
                    capture_stats.set_connection_interval_manifest(
                        writer.connection_interval_manifest()
                    )
        except BaseException as cleanup_error:
            if primary is not None:
                primary.add_note(
                    "Binance run writer cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            else:
                failure = cleanup_error
        if failure is not None:
            raise failure


@dataclass
class _CaptureStats:
    writer: _RunWriter | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    connection_opens: Counter[str] = field(default_factory=Counter)
    reconnects: Counter[str] = field(default_factory=Counter)
    connection_intervals: dict[str, dict[str, Any]] = field(default_factory=dict)
    _connection_interval_manifest: dict[str, Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    connection_errors: int = 0
    sequence_gaps: int = 0
    stale_depth_events: int = 0
    invalid_messages: int = 0

    def opened(
        self,
        stream: RecorderStream,
        *,
        reconnect: bool,
        connection_id: str,
        segment_id: str | None,
        opened_at: str,
        opened_monotonic_ns: int,
    ) -> None:
        with self._lock:
            key = f"{stream.market_type}:{stream.stream}"
            self.connection_opens[key] += 1
            if reconnect:
                self.reconnects[key] += 1
            if connection_id in self.connection_intervals:
                raise RuntimeError("binance_recorder_connection_interval_duplicate")
            self.connection_intervals[connection_id] = {
                "market": stream.market_type,
                "data_kind": stream.data_kind,
                "stream": stream.stream,
                "connection_id": connection_id,
                "segment_id": segment_id,
                "opened_at": opened_at,
                "opened_monotonic_ns": opened_monotonic_ns,
                "closed_at": None,
                "closed_monotonic_ns": None,
                "close_reason": None,
            }

    def closed(
        self,
        connection_id: str,
        *,
        closed_at: str,
        closed_monotonic_ns: int,
        close_reason: str,
    ) -> None:
        with self._lock:
            interval = self.connection_intervals.get(connection_id)
            if interval is None:
                raise RuntimeError("binance_recorder_connection_interval_missing")
            interval["closed_at"] = closed_at
            interval["closed_monotonic_ns"] = closed_monotonic_ns
            interval["close_reason"] = close_reason
            if self.writer is not None:
                self.writer.write_connection_interval(interval)
            del self.connection_intervals[connection_id]

    def connection_error(self) -> None:
        with self._lock:
            self.connection_errors += 1

    def sequence_gap(self) -> None:
        with self._lock:
            self.sequence_gaps += 1

    def stale(self) -> None:
        with self._lock:
            self.stale_depth_events += 1

    def invalid(self) -> None:
        with self._lock:
            self.invalid_messages += 1

    def set_connection_interval_manifest(self, value: dict[str, Any]) -> None:
        with self._lock:
            if self._connection_interval_manifest is not None:
                raise RuntimeError("binance_recorder_interval_manifest_duplicate")
            self._connection_interval_manifest = dict(value)

    def payload(self) -> dict[str, Any]:
        with self._lock:
            if self.connection_intervals:
                raise RuntimeError("binance_recorder_connection_interval_active")
            if self._connection_interval_manifest is None:
                raise RuntimeError("binance_recorder_interval_manifest_missing")
            return {
                "connection_opens": dict(sorted(self.connection_opens.items())),
                "reconnects": dict(sorted(self.reconnects.items())),
                "connection_intervals": dict(self._connection_interval_manifest),
                "connection_errors": self.connection_errors,
                "sequence_gaps": self.sequence_gaps,
                "stale_depth_events": self.stale_depth_events,
                "invalid_messages": self.invalid_messages,
            }


class _ConnectionProblem(Exception):
    pass


class _DepthResyncRequired(Exception):
    pass


class _InvalidPayload(Exception):
    pass


class _ReaderQuiescenceFailure(RuntimeError):
    pass


class _StreamReader(threading.Thread):
    def __init__(
        self,
        *,
        stream: RecorderStream,
        writer: _RunWriter,
        stats: _CaptureStats,
        stop: threading.Event,
        config: RecorderConfig,
        tick_size: str,
    ) -> None:
        super().__init__(
            name=f"binance-{stream.market_type}-{stream.data_kind}",
            daemon=True,
        )
        self.stream = stream
        self.writer = writer
        self.stats = stats
        self.stop = stop
        self.config = config
        self.tick_size = tick_size
        self.ready = threading.Event()
        self.failure: str | None = None

    def run(self) -> None:
        backoff = 1.0
        reconnect = False
        while not self.stop.is_set():
            try:
                self._read_connection(reconnect=reconnect)
            except _DepthResyncRequired:
                if self.stop.is_set():
                    break
                reconnect = True
                continue
            except (_ConnectionProblem, _InvalidPayload):
                if self.stop.is_set():
                    break
                self.stats.connection_error()
                self.stop.wait(min(backoff, self.config.reconnect_max_seconds))
                backoff = min(self.config.reconnect_max_seconds, backoff * 2)
                reconnect = True
                continue
            except BaseException as error:
                self.failure = type(error).__name__
                break
            break

    def _read_connection(self, *, reconnect: bool) -> None:
        connection_id = uuid.uuid4().hex
        segment_id = uuid.uuid4().hex if self.stream.data_kind == "depth" else None
        websocket: Any | None = None
        opened = False
        close_reason = "stop"
        closed_at: str | None = None
        closed_monotonic_ns: int | None = None
        depth_resources = ExitStack()
        depth_book: DepthReconstructor | None = None

        def freeze_close() -> None:
            nonlocal closed_at, closed_monotonic_ns
            if closed_at is None:
                closed_at = _utc_now()
                closed_monotonic_ns = time.monotonic_ns()

        try:
            try:
                websocket = _connect_websocket(
                    self.stream.url,
                    open_timeout=self.config.connect_timeout_seconds,
                    close_timeout=2,
                    max_size=MAX_WS_MESSAGE_BYTES,
                    max_queue=MAX_WS_QUEUE,
                    user_agent_header="qrp-binance-recorder/0.1",
                )
            except (OSError, TimeoutError, WebSocketException) as exc:
                raise _ConnectionProblem() from exc
            self.stats.opened(
                self.stream,
                reconnect=reconnect,
                connection_id=connection_id,
                segment_id=segment_id,
                opened_at=_utc_now(),
                opened_monotonic_ns=time.monotonic_ns(),
            )
            opened = True
            if self.stream.data_kind == "depth":
                depth_book = DepthReconstructor(
                    self.stream.market_type,
                    price_unit=self.tick_size,
                    snapshot_limit=SNAPSHOT_LIMITS[self.stream.market_type],
                    required_levels=RECONSTRUCTION_REQUIRED_LEVELS,
                    max_book_levels=RECONSTRUCTION_MAX_BOOK_LEVELS,
                    connection_close_ns=None,
                    stats=ReconstructionStats(),
                )
                depth_resources.callback(depth_book.close)
                assert segment_id is not None
                _record_depth_snapshot(
                    self.writer,
                    self.stream,
                    connection_id=connection_id,
                    segment_id=segment_id,
                    reconstructor=depth_book,
                )
            self.ready.set()
            last_message_at = time.monotonic()
            while True:
                if self.stop.is_set():
                    freeze_close()
                    break
                try:
                    raw = websocket.recv(timeout=1)
                except TimeoutError:
                    if (
                        time.monotonic() - last_message_at
                        >= self.config.connection_idle_timeout_seconds
                    ):
                        freeze_close()
                        raise _ConnectionProblem() from None
                    continue
                except (OSError, WebSocketException) as exc:
                    freeze_close()
                    raise _ConnectionProblem() from exc
                if raw is None:
                    freeze_close()
                    raise _ConnectionProblem()
                last_message_at = time.monotonic()
                receive_time = _utc_now()
                receive_monotonic_ns = time.monotonic_ns()
                try:
                    if isinstance(raw, str):
                        raw = raw.encode("utf-8")
                    if not isinstance(raw, bytes):
                        raise ValueError("binance_recorder_payload_bytes_invalid")
                    payload = validate_market_payload(
                        self.stream.market_type,
                        self.stream.data_kind,
                        self.stream.symbol,
                        parse_qrp_json_object(raw),
                    )
                except (UnicodeError, ValueError):
                    self.stats.invalid()
                    closed_at = receive_time
                    closed_monotonic_ns = receive_monotonic_ns
                    raise _InvalidPayload() from None
                capture_seq = self.writer.write(
                    symbol=self.stream.symbol,
                    market_type=self.stream.market_type,
                    data_kind=self.stream.data_kind,
                    stream=self.stream.stream,
                    payload=payload,
                    receive_time=receive_time,
                    receive_monotonic_ns=receive_monotonic_ns,
                    connection_id=connection_id,
                    segment_id=segment_id,
                )
                if depth_book is not None:
                    result = depth_book.advance(
                        normalize_recorder_record(
                            {
                                "capture_seq": capture_seq,
                                "tick_size": self.tick_size,
                                "market_type": self.stream.market_type,
                                "payload": payload,
                                "receive_time": receive_time,
                                "receive_monotonic_ns": receive_monotonic_ns,
                                "connection_id": connection_id,
                                "segment_id": segment_id,
                            }
                        )
                    )
                    if isinstance(result, BookBoundary):
                        if result.reason == "sequence_gap":
                            self.stats.sequence_gap()
                        freeze_close()
                        raise _DepthResyncRequired()
                    if not result:
                        self.stats.stale()
        except _DepthResyncRequired:
            close_reason = "depth_resync"
            freeze_close()
            raise
        except _InvalidPayload:
            close_reason = "failure"
            freeze_close()
            raise
        except _ConnectionProblem:
            close_reason = "connection_problem"
            freeze_close()
            raise
        except BaseException:
            close_reason = "failure"
            freeze_close()
            raise
        finally:
            primary = sys.exception()
            first_failure: BaseException | None = None
            cleanups = []
            if opened:
                assert closed_at is not None
                assert closed_monotonic_ns is not None
                cleanups.append(
                    (
                        "connection interval",
                        lambda: self.stats.closed(
                            connection_id,
                            closed_at=closed_at,
                            closed_monotonic_ns=closed_monotonic_ns,
                            close_reason=close_reason,
                        ),
                    )
                )
            cleanups.append(("depth validation", depth_resources.close))
            if websocket is not None:
                cleanups.append(("websocket", websocket.close))
            for name, close in cleanups:
                try:
                    close()
                except BaseException as cleanup_error:
                    if primary is not None:
                        primary.add_note(
                            f"Binance {name} close also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                    elif first_failure is None:
                        first_failure = cleanup_error
                    else:
                        first_failure.add_note(
                            f"Binance {name} close also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
            if first_failure is not None:
                raise first_failure


def _http_get_json(
    base_url: str,
    path: str,
    *,
    params: dict[str, str] | None = None,
) -> tuple[Any, str, int]:
    try:
        with httpx.stream(
            "GET",
            base_url + path,
            params=params,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "qrp-binance-recorder/0.1",
            },
            timeout=REST_REQUEST_TIMEOUT_SECONDS,
        ) as response:
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise OSError("binance_recorder_rest_status_error") from exc
            content_encoding = response.headers.get("Content-Encoding", "identity")
            if content_encoding.lower() not in {"", "identity"}:
                raise ValueError("binance_recorder_rest_content_encoding_invalid")
            raw = bytearray()
            for chunk in response.iter_raw(chunk_size=64 * 1024):
                if len(chunk) > MAX_REST_RESPONSE_BYTES - len(raw):
                    raise ValueError("binance_recorder_rest_response_too_large")
                raw.extend(chunk)
    except httpx.TimeoutException as exc:
        raise TimeoutError("binance_recorder_rest_timeout") from exc
    except httpx.HTTPError as exc:
        raise OSError("binance_recorder_rest_request_failed") from exc
    try:
        payload = parse_qrp_json_object(bytes(raw))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("binance_recorder_rest_json_invalid") from exc
    return payload, _utc_now(), time.monotonic_ns()


def _selected_symbol(payload: object, symbol_id: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise ValueError("binance_recorder_exchange_info_invalid")
    symbol = next(
        (
            item
            for item in payload["symbols"]
            if isinstance(item, dict) and item.get("symbol") == symbol_id
        ),
        None,
    )
    if symbol is None:
        raise ValueError("binance_recorder_symbol_metadata_missing")
    return symbol


def _record_market_metadata(
    writer: _RunWriter, *, market_types: tuple[str, ...], symbols: tuple[str, ...]
) -> dict[str, dict[str, dict[str, Any]]]:
    requests = {
        SPOT_MARKET_TYPE: (SPOT_REST_BASE_URL, "/api/v3/exchangeInfo"),
        FUTURES_MARKET_TYPE: (FUTURES_REST_BASE_URL, "/fapi/v1/exchangeInfo"),
    }
    result: dict[str, dict[str, dict[str, Any]]] = {symbol: {} for symbol in symbols}
    for market_type in market_types:
        base_url, endpoint = requests[market_type]
        symbol_groups = (
            (symbols,)
            if market_type == FUTURES_MARKET_TYPE
            else tuple((symbol,) for symbol in symbols)
        )
        for requested_symbols in symbol_groups:
            params = (
                {"symbol": requested_symbols[0]}
                if market_type == SPOT_MARKET_TYPE
                else None
            )
            payload = symbol = filters = tick_size = received = monotonic_ns = None
            try:
                payload, received, monotonic_ns = _http_get_json(
                    base_url,
                    endpoint,
                    params=params,
                )
                for symbol_id in requested_symbols:
                    symbol = _selected_symbol(payload, symbol_id)
                    filters = symbol.get("filters")
                    if not isinstance(filters, list):
                        raise ValueError("binance_recorder_symbol_metadata_invalid")
                    tick_size = next(
                        (
                            item.get("tickSize")
                            for item in filters
                            if isinstance(item, dict)
                            and item.get("filterType") == "PRICE_FILTER"
                        ),
                        None,
                    )
                    strict_positive_float(
                        tick_size,
                        error="binance_recorder_tick_size_invalid",
                    )
                    result[symbol_id][market_type] = {
                        "tick_size": tick_size,
                        "snapshot_limit": SNAPSHOT_LIMITS[market_type],
                    }
                    writer.write(
                        symbol=symbol_id,
                        market_type=market_type,
                        data_kind="metadata",
                        stream=f"REST:{endpoint}",
                        payload=symbol,
                        receive_time=received,
                        receive_monotonic_ns=monotonic_ns,
                    )
            finally:
                payload = symbol = filters = tick_size = received = monotonic_ns = None
    return result


def _validate_snapshot(payload: object, *, market_type: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("binance_recorder_snapshot_object_required")
    strict_non_negative_integer(
        payload.get("lastUpdateId"),
        error="binance_recorder_snapshot_id_invalid",
    )
    if market_type == FUTURES_MARKET_TYPE:
        for key in ("E", "T"):
            strict_non_negative_integer(
                payload.get(key),
                error="binance_recorder_snapshot_time_invalid",
            )
    validate_depth_update_levels(
        payload.get("bids"), error="binance_recorder_snapshot_bids_invalid"
    )
    validate_depth_update_levels(
        payload.get("asks"), error="binance_recorder_snapshot_asks_invalid"
    )
    return payload


def _record_depth_snapshot(
    writer: _RunWriter,
    stream: RecorderStream,
    *,
    connection_id: str,
    segment_id: str,
    reconstructor: DepthReconstructor,
) -> None:
    market_type = stream.market_type
    base_url, endpoint = (
        (SPOT_REST_BASE_URL, "/api/v3/depth")
        if market_type == SPOT_MARKET_TYPE
        else (FUTURES_REST_BASE_URL, "/fapi/v1/depth")
    )
    payload = snapshot = received = monotonic_ns = None
    try:
        try:
            payload, received, monotonic_ns = _http_get_json(
                base_url,
                endpoint,
                params={
                    "symbol": stream.symbol,
                    "limit": str(SNAPSHOT_LIMITS[market_type]),
                },
            )
        except (TimeoutError, OSError) as error:
            raise _ConnectionProblem() from error
        snapshot = _validate_snapshot(payload, market_type=market_type)
        capture_seq = writer.write(
            symbol=stream.symbol,
            market_type=market_type,
            data_kind="depth_snapshot",
            stream=f"REST:{endpoint}",
            payload=snapshot,
            receive_time=received,
            receive_monotonic_ns=monotonic_ns,
            connection_id=connection_id,
            segment_id=segment_id,
        )
        reconstructor.bootstrap(
            normalize_recorder_record(
                {
                    "capture_seq": capture_seq,
                    "tick_size": reconstructor.price_unit,
                    "market_type": market_type,
                    "payload": snapshot,
                    "receive_time": received,
                    "receive_monotonic_ns": monotonic_ns,
                    "connection_id": connection_id,
                    "segment_id": segment_id,
                }
            )
        )
    finally:
        payload = snapshot = received = monotonic_ns = None


def record(config: RecorderConfig) -> dict[str, Any]:
    """Record one bounded public-data run and publish its exact manifest."""

    config.validate()
    return atomic_write_directory(
        Path(config.output_root),
        lambda root: _record_staged(config, root),
    )


def _record_staged(config: RecorderConfig, root: Path) -> dict[str, Any]:
    streams = RECORDER_STREAMS_BY_SCOPE[config.scope]
    market_types = RECORDING_SCOPE_MARKET_TYPES[config.scope]
    writer = _RunWriter(root)
    stats = _CaptureStats(writer=writer)
    stop = threading.Event()
    readers: list[_StreamReader] = []
    started_at = _utc_now()
    started_clock = time.monotonic()
    cpu_start = time.process_time()
    process = psutil.Process(os.getpid())
    market_metadata: dict[str, dict[str, dict[str, Any]]] = {}
    duration_reached = False
    cleanup_errors: list[str] = []
    files: list[dict[str, Any]] = []
    with _writer_lifecycle(writer, files, capture_stats=stats):
        market_metadata = _record_market_metadata(
            writer, market_types=market_types, symbols=config.symbols
        )
        try:
            readers = [
                _StreamReader(
                    stream=stream,
                    writer=writer,
                    stats=stats,
                    stop=stop,
                    config=config,
                    tick_size=market_metadata[stream.symbol][stream.market_type][
                        "tick_size"
                    ],
                )
                for stream in streams
            ]
            for reader in readers:
                reader.start()
            deadline = time.monotonic() + config.connect_timeout_seconds
            for reader in readers:
                remaining = max(0.0, deadline - time.monotonic())
                if not reader.ready.wait(remaining):
                    raise TimeoutError("binance_recorder_stream_connect_timeout")
            started_at = _utc_now()
            started_clock = time.monotonic()
            cpu_start = time.process_time()
            while True:
                elapsed = time.monotonic() - started_clock
                remaining = config.duration_seconds - elapsed
                if remaining <= 0:
                    duration_reached = True
                    break
                stop.wait(min(0.25, remaining))
                for reader in readers:
                    if reader.failure is not None:
                        raise RuntimeError("binance_recorder_reader_failure")
                    if not reader.is_alive():
                        raise RuntimeError("binance_recorder_reader_thread_died")
        finally:
            active_primary = sys.exception()
            stop.set()
            join_failed = False
            for reader in readers:
                try:
                    reader.join(timeout=20)
                    if reader.is_alive():
                        join_failed = True
                        cleanup_errors.append(f"{reader.name}:join_timeout")
                except BaseException as cleanup_error:
                    join_failed = True
                    if active_primary is not None:
                        active_primary.add_note(
                            "Binance reader join also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                    else:
                        cleanup_errors.append(type(cleanup_error).__name__)
            if join_failed:
                try:
                    writer.abort()
                except BaseException as cleanup_error:
                    if active_primary is None:
                        raise
                    active_primary.add_note(
                        "Binance run writer abort also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
                if active_primary is None:
                    raise _ReaderQuiescenceFailure(
                        "binance_recorder_reader_quiescence_timeout"
                    )
            if active_primary is None and any(
                reader.failure is not None for reader in readers
            ):
                raise RuntimeError("binance_recorder_reader_failure")
    captured = stats.payload()

    ended_at = _utc_now()
    duration_seconds = max(0.001, time.monotonic() - started_clock)
    missing_required_streams = [
        f"{stream.market_type}:{stream.stream}"
        for stream in streams
        if not any(
            item["market"] == stream.market_type
            and item["stream"] == stream.stream
            and item["record_count"] > 0
            for item in files
        )
    ]
    status = (
        "FAIL"
        if cleanup_errors
        else "PARTIAL"
        if captured["connection_errors"]
        or any(captured["reconnects"].values())
        or captured["sequence_gaps"]
        or captured["invalid_messages"]
        or missing_required_streams
        else "PASS"
        if duration_reached
        else "PARTIAL"
    )
    manifest = {
        "schema_id": "qrp-binance-recorder",
        "run_id": uuid.uuid4().hex,
        "recording_scope": config.scope,
        "status": status,
        "market_data_status": status,
        "venue": "binance",
        "symbols": list(config.symbols),
        "market_types": list(market_types),
        "market_metadata": market_metadata,
        "files": files,
        "capture": {
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": duration_seconds,
            "duration_target_seconds": config.duration_seconds,
            "event_counts": writer.event_counts(),
            "missing_required_streams": missing_required_streams,
            **captured,
            "process_cpu_seconds": max(0.0, time.process_time() - cpu_start),
            "threads": len(readers),
            "descendants": len(process.children(recursive=True)),
            "cleanup_errors": cleanup_errors,
        },
    }
    atomic_write_json(root / "run.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unauthenticated Binance public-market-data recorder."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--connect-timeout-seconds", type=float, default=20.0)
    parser.add_argument(
        "--scope",
        choices=tuple(RECORDER_STREAMS_BY_SCOPE),
        default=FUTURES_HFT_RECORDING_SCOPE,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = record(
            RecorderConfig(
                output_root=args.output_root,
                duration_seconds=args.duration_seconds,
                connect_timeout_seconds=args.connect_timeout_seconds,
                scope=args.scope,
            )
        )
    except BaseException as error:
        print(
            f"binance recorder failed: {type(error).__name__}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_id": manifest["run_id"],
                "event_counts": manifest["capture"]["event_counts"],
            },
            indent=2,
        )
    )
    return 0 if manifest["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
