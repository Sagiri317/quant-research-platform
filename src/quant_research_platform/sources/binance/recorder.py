"""Read-only Binance BTCUSDT Spot and USDⓈ-M recorder."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
import os
import sys
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import psutil
from websockets.exceptions import WebSocketException
from websockets.sync.client import connect as _connect_websocket

from quant_research_platform.contracts._json import parse_qrp_json_object
from quant_research_platform.runtime.resources import (
    MemoryLease,
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    SPOT_MARKET_TYPE,
)
from quant_research_platform.sources.binance.raw_exports import (
    _json_record_working_set_bytes,
)
from quant_research_platform.sources.binance.validation import (
    strict_boolean,
    strict_iso_timestamp,
    strict_non_negative_float,
    strict_non_negative_integer,
    strict_positive_float,
    validate_depth_update_levels,
)
from quant_research_platform.storage.atomic import atomic_write_json

SYMBOL = "BTCUSDT"
SPOT_REST_BASE_URL = "https://data-api.binance.vision"
FUTURES_REST_BASE_URL = "https://fapi.binance.com"
MAX_WS_MESSAGE_BYTES = 4 * 1024 * 1024
MAX_WS_QUEUE = 4
MAX_REST_RESPONSE_BYTES = 4 * 1024 * 1024
REST_REQUEST_TIMEOUT_SECONDS = 15.0
SNAPSHOT_LIMITS = {SPOT_MARKET_TYPE: 5000, FUTURES_MARKET_TYPE: 1000}
_REST_WORKING_SET_BYTES = (
    _json_record_working_set_bytes(MAX_REST_RESPONSE_BYTES) + 64 * 1024
)


@dataclass(frozen=True)
class RecorderStream:
    market_type: str
    data_kind: str
    stream: str
    url: str


RECORDER_STREAMS = (
    RecorderStream(
        SPOT_MARKET_TYPE,
        "depth",
        "btcusdt@depth@100ms",
        "wss://stream.binance.com:9443/ws/btcusdt@depth@100ms",
    ),
    RecorderStream(
        FUTURES_MARKET_TYPE,
        "depth",
        "btcusdt@depth@100ms",
        "wss://fstream.binance.com/public/ws/btcusdt@depth@100ms",
    ),
    RecorderStream(
        FUTURES_MARKET_TYPE,
        "agg_trades",
        "btcusdt@aggTrade",
        "wss://fstream.binance.com/market/ws/btcusdt@aggTrade",
    ),
)


def validate_stream_payload(
    stream: RecorderStream,
    payload: object,
) -> dict[str, Any]:
    """Validate one raw public payload while preserving its exchange shape."""

    if not isinstance(payload, dict):
        raise ValueError("binance_recorder_payload_object_required")
    expected_event = "depthUpdate" if stream.data_kind == "depth" else "aggTrade"
    if payload.get("e") != expected_event or payload.get("s") != SYMBOL:
        raise ValueError("binance_recorder_market_payload_identity_invalid")
    strict_non_negative_integer(
        payload.get("E"), error="binance_recorder_event_time_invalid"
    )
    if stream.market_type == FUTURES_MARKET_TYPE:
        if payload.get("st") != 1 or (
            stream.data_kind == "depth" and payload.get("ps") != SYMBOL
        ):
            raise ValueError("binance_recorder_market_payload_identity_invalid")
    if stream.data_kind == "depth":
        for key in ("U", "u"):
            strict_non_negative_integer(
                payload.get(key), error="binance_recorder_depth_id_invalid"
            )
        if payload["U"] > payload["u"]:
            raise ValueError("binance_recorder_depth_id_range_invalid")
        if stream.market_type == FUTURES_MARKET_TYPE:
            for key in ("T", "pu"):
                strict_non_negative_integer(
                    payload.get(key), error="binance_recorder_depth_id_invalid"
                )
        validate_depth_update_levels(
            payload.get("b"), error="binance_recorder_depth_bids_invalid"
        )
        validate_depth_update_levels(
            payload.get("a"), error="binance_recorder_depth_asks_invalid"
        )
        return payload
    for key in ("a", "f", "l", "T"):
        strict_non_negative_integer(
            payload.get(key), error="binance_recorder_trade_id_invalid"
        )
    if payload["l"] < payload["f"]:
        raise ValueError("binance_recorder_trade_id_range_invalid")
    strict_positive_float(
        payload.get("p"), error="binance_recorder_trade_price_invalid"
    )
    quantity = strict_positive_float(
        payload.get("q"), error="binance_recorder_trade_quantity_invalid"
    )
    if stream.market_type == FUTURES_MARKET_TYPE:
        visible_quantity = strict_non_negative_float(
            payload.get("nq"),
            error="binance_recorder_trade_visible_quantity_invalid",
        )
        if visible_quantity > quantity:
            raise ValueError("binance_recorder_trade_visible_quantity_invalid")
    strict_boolean(payload.get("m"), error="binance_recorder_trade_side_invalid")
    return payload


@dataclass(frozen=True)
class RecorderConfig:
    output_root: Path
    duration_seconds: float
    connect_timeout_seconds: float = 20.0
    reconnect_max_seconds: float = 30.0
    connection_idle_timeout_seconds: float = 30.0

    def validate(self) -> None:
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
        if (root / "run.json").exists():
            raise FileExistsError("binance_recorder_output_exists")
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("binance_recorder_output_not_empty")


def _positive_number(value: object, *, error: str) -> float:
    if isinstance(value, bool):
        raise ValueError(error)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not math.isfinite(number) or number <= 0:
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


@dataclass
class _RunWriter:
    root: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _capture_seq: int = field(default=0, init=False)
    _handles: dict[tuple[str, str, str, str], _OpenFile] = field(
        default_factory=dict,
        init=False,
    )
    _files: dict[Path, dict[str, Any]] = field(default_factory=dict, init=False)
    _event_counts: Counter[str] = field(default_factory=Counter, init=False)

    def write(
        self,
        *,
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
            self._capture_seq += 1
            capture_seq = self._capture_seq
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
            hour = receive_time[:13]
            for key in tuple(self._handles):
                if key[0] != hour:
                    self._close_key(key)
            key = (hour, market_type, data_kind, stream)
            opened = self._handles.get(key)
            if opened is None:
                opened = self._open_file(
                    key,
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
            self._event_counts[data_kind] += 1
            if metadata["record_count"] % 256 == 0:
                opened.gzip_handle.flush()
                opened.raw_handle.flush()
            return capture_seq

    def _open_file(
        self,
        key: tuple[str, str, str, str],
        *,
        market_type: str,
        data_kind: str,
        stream: str,
        receive_time: str,
    ) -> _OpenFile:
        directory = (
            self.root / f"date={receive_time[:10]}" / f"hour={receive_time[11:13]}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{market_type}_{data_kind}.jsonl.gz"
        raw_handle = path.open("ab")
        gzip_handle = gzip.GzipFile(fileobj=raw_handle, mode="wb", compresslevel=6)
        opened = _OpenFile(raw_handle, gzip_handle, path)
        self._handles[key] = opened
        self._files.setdefault(
            path,
            {
                "path": path.relative_to(self.root).as_posix(),
                "venue": "binance",
                "symbol": SYMBOL,
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

    def _close_key(self, key: tuple[str, str, str, str]) -> None:
        opened = self._handles.pop(key, None)
        if opened is None:
            return
        try:
            opened.gzip_handle.close()
        finally:
            opened.raw_handle.close()

    def close(self) -> None:
        with self._lock:
            for key in tuple(self._handles):
                self._close_key(key)

    def file_manifests(
        self,
        resource_policy: RuntimeResourcePolicy,
    ) -> list[dict[str, Any]]:
        with self._lock:
            if self._handles:
                raise RuntimeError("binance_recorder_writer_not_closed")
            result = []
            for path in sorted(self._files):
                with resource_policy.lease(
                    "binance.recorder.file_hash",
                    128 * 1024,
                ):
                    digest = hashlib.sha256()
                    buffer = bytearray(64 * 1024)
                    with path.open("rb") as handle:
                        while count := handle.readinto(buffer):
                            digest.update(memoryview(buffer)[:count])
                    result.append(
                        {
                            **self._files[path],
                            "compressed_bytes": path.stat().st_size,
                            "sha256": digest.hexdigest(),
                        }
                    )
            return result

    def event_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(sorted(self._event_counts.items()))


@contextmanager
def _writer_lifecycle(
    writer: _RunWriter,
    policy: RuntimeResourcePolicy,
    files: list[dict[str, Any]],
    cleanup_errors: list[str],
    capture_stats: _CaptureStats | None = None,
) -> Iterator[None]:
    try:
        yield
    finally:
        primary = sys.exception()
        try:
            writer.close()
            files.extend(writer.file_manifests(policy))
        except BaseException as cleanup_error:
            if primary is not None:
                primary.add_note(
                    "Binance run writer cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            else:
                cleanup_errors.append(type(cleanup_error).__name__)
        finally:
            if capture_stats is not None:
                try:
                    capture_stats.close()
                except BaseException as cleanup_error:
                    if primary is not None:
                        primary.add_note(
                            "Binance capture stats cleanup also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                    else:
                        cleanup_errors.append(type(cleanup_error).__name__)


@dataclass
class _CaptureStats:
    resource_policy: RuntimeResourcePolicy | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _interval_lease: MemoryLease | None = field(default=None, init=False, repr=False)
    connection_opens: Counter[str] = field(default_factory=Counter)
    reconnects: Counter[str] = field(default_factory=Counter)
    connection_intervals: list[dict[str, Any]] = field(default_factory=list)
    connection_errors: int = 0
    sequence_gaps: int = 0
    stale_depth_events: int = 0
    invalid_messages: int = 0

    def __post_init__(self) -> None:
        if self.resource_policy is not None:
            self._interval_lease = self.resource_policy.lease(
                "binance.recorder.connection_intervals",
                0,
            )

    def bind(self, resource_policy: RuntimeResourcePolicy) -> None:
        if self.resource_policy is not None or self._interval_lease is not None:
            raise RuntimeError("binance_recorder_capture_stats_already_bound")
        self.resource_policy = resource_policy
        self._interval_lease = resource_policy.lease(
            "binance.recorder.connection_intervals",
            0,
        )

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
            if self._interval_lease is not None:
                self._interval_lease.grow(1024)
            key = f"{stream.market_type}:{stream.stream}"
            self.connection_opens[key] += 1
            if reconnect:
                self.reconnects[key] += 1
            self.connection_intervals.append(
                {
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
            )

    def closed(
        self,
        connection_id: str,
        *,
        closed_at: str,
        closed_monotonic_ns: int,
        close_reason: str,
    ) -> None:
        with self._lock:
            interval = next(
                (
                    item
                    for item in reversed(self.connection_intervals)
                    if item["connection_id"] == connection_id
                    and item["closed_at"] is None
                ),
                None,
            )
            if interval is None:
                raise RuntimeError("binance_recorder_connection_interval_missing")
            interval["closed_at"] = closed_at
            interval["closed_monotonic_ns"] = closed_monotonic_ns
            interval["close_reason"] = close_reason

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

    def close(self) -> None:
        if self._interval_lease is not None:
            self._interval_lease.close()
            self._interval_lease = None

    def payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "connection_opens": dict(sorted(self.connection_opens.items())),
                "reconnects": dict(sorted(self.reconnects.items())),
                "connection_intervals": self.connection_intervals,
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


class _StreamReader(threading.Thread):
    def __init__(
        self,
        *,
        stream: RecorderStream,
        writer: _RunWriter,
        stats: _CaptureStats,
        stop: threading.Event,
        config: RecorderConfig,
        on_generation_ready: Callable[[_StreamReader, int], None] | None = None,
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
        self.on_generation_ready = on_generation_ready
        self.ready = threading.Event()
        self.failure: str | None = None
        self._identity_lock = threading.Lock()
        self._connection_id: str | None = None
        self._segment_id: str | None = None
        self._generation = 0
        self._snapshot_ack: threading.Event | None = None

    def current_identity(self) -> tuple[int, str | None, str | None]:
        with self._identity_lock:
            return self._generation, self._connection_id, self._segment_id

    def acknowledge_snapshot(self, generation: int) -> bool:
        with self._identity_lock:
            if generation != self._generation or self._snapshot_ack is None:
                return False
            self._snapshot_ack.set()
            return True

    def wait_for_snapshot(self, generation: int, timeout: float) -> bool:
        with self._identity_lock:
            if generation != self._generation or self._snapshot_ack is None:
                return False
            acknowledgement = self._snapshot_ack
        return acknowledgement.wait(timeout)

    def _await_snapshot(self, generation: int) -> bool:
        while not self.stop.is_set():
            if self.wait_for_snapshot(generation, 0.1):
                return True
        return False

    def run(self) -> None:
        backoff = 1.0
        reconnect = False
        while not self.stop.is_set():
            try:
                self._read_connection(reconnect=reconnect)
            except (_ConnectionProblem, _DepthResyncRequired, _InvalidPayload):
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
            with self._identity_lock:
                self._generation += 1
                self._connection_id = connection_id
                self._segment_id = segment_id
                self._snapshot_ack = (
                    threading.Event()
                    if self.stream.data_kind == "depth"
                    else None
                )
            self.stats.opened(
                self.stream,
                reconnect=reconnect,
                connection_id=connection_id,
                segment_id=segment_id,
                opened_at=_utc_now(),
                opened_monotonic_ns=time.monotonic_ns(),
            )
            opened = True
            self.ready.set()
            if self.stream.data_kind == "depth":
                generation = self.current_identity()[0]
                if self.on_generation_ready is not None:
                    self.on_generation_ready(self, generation)
                if not self._await_snapshot(generation):
                    raise _ConnectionProblem()
            previous_u: int | None = None
            last_message_at = time.monotonic()
            while not self.stop.is_set():
                try:
                    raw = websocket.recv(timeout=1)
                except TimeoutError:
                    if (
                        time.monotonic() - last_message_at
                        >= self.config.connection_idle_timeout_seconds
                    ):
                        raise _ConnectionProblem() from None
                    continue
                except (OSError, WebSocketException) as exc:
                    raise _ConnectionProblem() from exc
                if raw is None:
                    raise _ConnectionProblem()
                last_message_at = time.monotonic()
                receive_time = _utc_now()
                receive_monotonic_ns = time.monotonic_ns()
                try:
                    if isinstance(raw, str):
                        raw = raw.encode("utf-8")
                    if not isinstance(raw, bytes):
                        raise ValueError("binance_recorder_payload_bytes_invalid")
                    payload = validate_stream_payload(
                        self.stream,
                        parse_qrp_json_object(raw),
                    )
                except (UnicodeError, ValueError):
                    self.stats.invalid()
                    raise _InvalidPayload() from None
                needs_resync = False
                if self.stream.data_kind == "depth":
                    current_u = payload["u"]
                    if previous_u is not None:
                        if current_u <= previous_u:
                            self.stats.stale()
                        elif self.stream.market_type == FUTURES_MARKET_TYPE:
                            needs_resync = payload["pu"] != previous_u
                        else:
                            needs_resync = payload["U"] > previous_u + 1
                    if current_u > (previous_u if previous_u is not None else -1):
                        previous_u = current_u
                self.writer.write(
                    market_type=self.stream.market_type,
                    data_kind=self.stream.data_kind,
                    stream=self.stream.stream,
                    payload=payload,
                    receive_time=receive_time,
                    receive_monotonic_ns=receive_monotonic_ns,
                    connection_id=connection_id,
                    segment_id=segment_id,
                )
                if needs_resync:
                    self.stats.sequence_gap()
                    raise _DepthResyncRequired()
        except _DepthResyncRequired:
            close_reason = "depth_resync"
            raise
        except _InvalidPayload:
            close_reason = "failure"
            raise
        except _ConnectionProblem:
            close_reason = "connection_problem"
            raise
        except BaseException:
            close_reason = "failure"
            raise
        finally:
            primary = sys.exception()
            if opened:
                closed_at = _utc_now()
                closed_monotonic_ns = time.monotonic_ns()
                try:
                    self.stats.closed(
                        connection_id,
                        closed_at=closed_at,
                        closed_monotonic_ns=closed_monotonic_ns,
                        close_reason=close_reason,
                    )
                except BaseException as cleanup_error:
                    if primary is not None:
                        primary.add_note(
                            "Binance connection interval close also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                    else:
                        raise
            if websocket is not None:
                try:
                    websocket.close()
                except BaseException as cleanup_error:
                    if primary is None:
                        raise
                    primary.add_note(
                        "Binance websocket close also failed: "
                        f"{type(cleanup_error).__name__}"
                    )


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


def _selected_symbol(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
        raise ValueError("binance_recorder_exchange_info_invalid")
    symbol = next(
        (
            item
            for item in payload["symbols"]
            if isinstance(item, dict) and item.get("symbol") == SYMBOL
        ),
        None,
    )
    if symbol is None:
        raise ValueError("binance_recorder_symbol_metadata_missing")
    return symbol


def _record_market_metadata(
    writer: _RunWriter,
    policy: RuntimeResourcePolicy,
) -> dict[str, dict[str, Any]]:
    requests = (
        (
            SPOT_MARKET_TYPE,
            SPOT_REST_BASE_URL,
            "/api/v3/exchangeInfo",
            {"symbol": SYMBOL},
        ),
        (
            FUTURES_MARKET_TYPE,
            FUTURES_REST_BASE_URL,
            "/fapi/v1/exchangeInfo",
            None,
        ),
    )
    result: dict[str, dict[str, Any]] = {}
    for market_type, base_url, endpoint, params in requests:
        with policy.lease("binance.recorder.rest", _REST_WORKING_SET_BYTES):
            payload, received, monotonic_ns = _http_get_json(
                base_url,
                endpoint,
                params=params,
            )
            symbol = _selected_symbol(payload)
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
            result[market_type] = {
                "tick_size": tick_size,
                "snapshot_limit": SNAPSHOT_LIMITS[market_type],
            }
            writer.write(
                market_type=market_type,
                data_kind="metadata",
                stream=f"REST:{endpoint}",
                payload=symbol,
                receive_time=received,
                receive_monotonic_ns=monotonic_ns,
            )
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
    policy: RuntimeResourcePolicy,
    reader: _StreamReader,
    *,
    generation: int,
) -> None:
    market_type = reader.stream.market_type
    base_url, endpoint = (
        (SPOT_REST_BASE_URL, "/api/v3/depth")
        if market_type == SPOT_MARKET_TYPE
        else (FUTURES_REST_BASE_URL, "/fapi/v1/depth")
    )
    before = reader.current_identity()
    if before[0] != generation:
        raise _ConnectionProblem()
    with policy.lease("binance.recorder.rest", _REST_WORKING_SET_BYTES):
        payload, received, monotonic_ns = _http_get_json(
            base_url,
            endpoint,
            params={
                "symbol": SYMBOL,
                "limit": str(SNAPSHOT_LIMITS[market_type]),
            },
        )
        after = reader.current_identity()
        if before != after:
            raise _ConnectionProblem()
        snapshot = _validate_snapshot(payload, market_type=market_type)
        writer.write(
            market_type=market_type,
            data_kind="depth_snapshot",
            stream=f"REST:{endpoint}",
            payload=snapshot,
            receive_time=received,
            receive_monotonic_ns=monotonic_ns,
            connection_id=after[1],
            segment_id=after[2],
        )
        if not reader.acknowledge_snapshot(generation):
            raise _ConnectionProblem()


def record(config: RecorderConfig) -> dict[str, Any]:
    """Record one bounded public-data run and publish its exact manifest."""

    config.validate()
    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    writer = _RunWriter(root)
    stats = _CaptureStats()
    stop = threading.Event()
    readers: list[_StreamReader] = []
    started_at = _utc_now()
    started_clock = time.monotonic()
    cpu_start = time.process_time()
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    market_metadata: dict[str, dict[str, Any]] = {}
    duration_reached = False
    failure: str | None = None
    cleanup_errors: list[str] = []
    files: list[dict[str, Any]] = []
    manifest: dict[str, Any] | None = None
    captured: dict[str, Any] = {}

    try:
        with (
            runtime_resource_policy_scope() as policy,
            _writer_lifecycle(
                writer,
                policy,
                files,
                cleanup_errors,
                capture_stats=stats,
            ),
        ):
            stats.bind(policy)
            market_metadata = _record_market_metadata(writer, policy)
            websocket_reserve = (
                len(RECORDER_STREAMS) * MAX_WS_QUEUE * MAX_WS_MESSAGE_BYTES * 4
            )
            with policy.lease(
                "binance.recorder.websocket_buffers",
                websocket_reserve,
            ):
                try:
                    readers = [
                        _StreamReader(
                            stream=stream,
                            writer=writer,
                            stats=stats,
                            stop=stop,
                            config=config,
                            on_generation_ready=(
                                lambda reader, generation: _record_depth_snapshot(
                                    writer,
                                    policy,
                                    reader,
                                    generation=generation,
                                )
                            ),
                        )
                        for stream in RECORDER_STREAMS
                    ]
                    for reader in readers:
                        reader.start()
                    deadline = time.monotonic() + config.connect_timeout_seconds
                    for reader in readers:
                        remaining = max(0.0, deadline - time.monotonic())
                        if not reader.ready.wait(remaining):
                            raise TimeoutError(
                                "binance_recorder_stream_connect_timeout"
                            )
                    depth_readers = [
                        reader
                        for reader in readers
                        if reader.stream.data_kind == "depth"
                    ]
                    for reader in depth_readers:
                        remaining = max(0.0, deadline - time.monotonic())
                        generation = reader.current_identity()[0]
                        if not reader.wait_for_snapshot(generation, remaining):
                            raise TimeoutError(
                                "binance_recorder_snapshot_ack_timeout"
                            )
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
                        peak_rss = max(peak_rss, process.memory_info().rss)
                        for reader in readers:
                            if reader.failure is not None:
                                raise RuntimeError("binance_recorder_reader_failure")
                            if not reader.is_alive():
                                raise RuntimeError(
                                    "binance_recorder_reader_thread_died"
                                )
                finally:
                    active_primary = sys.exception()
                    stop.set()
                    for reader in readers:
                        try:
                            reader.join(timeout=20)
                            if reader.is_alive():
                                cleanup_errors.append(f"{reader.name}:join_timeout")
                        except BaseException as cleanup_error:
                            if active_primary is not None:
                                active_primary.add_note(
                                    "Binance reader join also failed: "
                                    f"{type(cleanup_error).__name__}"
                                )
                            else:
                                cleanup_errors.append(type(cleanup_error).__name__)
            captured = stats.payload()
    except BaseException as error:
        failure = type(error).__name__
        raise
    finally:
        primary = sys.exception()
        stop.set()
        ended_at = _utc_now()
        duration_seconds = max(0.001, time.monotonic() - started_clock)
        if not captured:
            captured = stats.payload()
        missing_required_streams = [
            f"{stream.market_type}:{stream.stream}"
            for stream in RECORDER_STREAMS
            if not any(
                item["market"] == stream.market_type
                and item["stream"] == stream.stream
                and item["record_count"] > 0
                for item in files
            )
        ]
        status = (
            "FAIL"
            if failure is not None or cleanup_errors
            else "PARTIAL"
            if captured["connection_errors"]
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
            "status": status,
            "market_data_status": status,
            "venue": "binance",
            "symbol": SYMBOL,
            "market_types": [SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE],
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
                "peak_rss_bytes": peak_rss,
                "process_cpu_seconds": max(0.0, time.process_time() - cpu_start),
                "threads": len(readers),
                "descendants": len(process.children(recursive=True)),
                "failure": failure,
                "cleanup_errors": cleanup_errors,
            },
        }
        try:
            atomic_write_json(root / "run.json", manifest)
        except BaseException as publication_error:
            if primary is not None:
                primary.add_note(
                    "Binance manifest publication also failed: "
                    f"{type(publication_error).__name__}"
                )
            else:
                raise
    assert manifest is not None
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--connect-timeout-seconds", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = record(
            RecorderConfig(
                output_root=args.output_root,
                duration_seconds=args.duration_seconds,
                connect_timeout_seconds=args.connect_timeout_seconds,
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
