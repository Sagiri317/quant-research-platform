"""Exact recorder-run replay and deterministic Binance MBP reconstruction."""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import sqlite3
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from quant_research_platform.contracts._json import canonical_json_hash
from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    price_to_tick,
    validate_tick_size,
)
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    SPOT_MARKET_TYPE,
)
from quant_research_platform.sources.binance.raw_exports import (
    _capture_raw_file_identity,
    _json_record_working_set_bytes,
    _raw_path_session,
    _verify_raw_file_handle,
)
from quant_research_platform.sources.binance.validation import (
    strict_iso_timestamp,
    strict_non_negative_float,
    strict_non_negative_integer,
)
from quant_research_platform.storage._dataset_physical import DatasetFileIdentity

_FILE_MANIFEST_KEYS = frozenset(
    {
        "path",
        "venue",
        "symbol",
        "market",
        "stream",
        "data_kind",
        "record_count",
        "capture_seq_min",
        "capture_seq_max",
        "max_record_bytes",
        "compressed_bytes",
        "sha256",
    }
)
_LOWER_HEX = frozenset("0123456789abcdef")
_HASH_BUFFER_BYTES = 64 * 1024
_LEVEL_BYTES = 256


@dataclass
class ReconstructionStats:
    valid_segments: int = 0
    recovered_segments: int = 0
    invalid_segments: int = 0
    gaps: int = 0
    reconnect_boundaries: int = 0
    stale_events: int = 0
    duplicate_events: int = 0
    crossed_books: int = 0
    one_sided_books: int = 0
    unsupported_coverage: int = 0
    emitted_states: int = 0
    non_monotonic_timestamps: int = 0
    coverage_segments: dict[str, tuple[int, int]] = field(default_factory=dict)


@dataclass(frozen=True)
class _RunFile:
    metadata: dict[str, Any]
    identity: DatasetFileIdentity


@dataclass(frozen=True)
class RecorderRun:
    root: Path
    manifest: dict[str, Any]
    source_identity: dict[str, str]
    resource_policy: RuntimeResourcePolicy
    _files: tuple[_RunFile, ...] = field(repr=False)

    def iter_records(
        self,
        *,
        market_type: str | None = None,
        data_kinds: Sequence[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        kinds = None if data_kinds is None else frozenset(data_kinds)
        selected = [
            item
            for item in self._files
            if (market_type is None or item.metadata["market"] == market_type)
            and (kinds is None or item.metadata["data_kind"] in kinds)
        ]
        if not selected:
            raise ValueError("binance_replay_requested_files_missing")
        previous_capture_seq: int | None = None
        for group in _overlapping_capture_groups(selected):
            streams = [
                _iter_raw_file(
                    self.root,
                    item,
                    manifest=self.manifest,
                    resource_policy=self.resource_policy,
                )
                for item in group
            ]
            for record in heapq.merge(
                *streams,
                key=lambda value: value["capture_seq"],
            ):
                capture_seq = record["capture_seq"]
                if (
                    previous_capture_seq is not None
                    and capture_seq <= previous_capture_seq
                ):
                    raise ValueError(
                        "binance_replay_capture_seq_duplicate_or_non_monotonic"
                    )
                previous_capture_seq = capture_seq
                yield record


@contextmanager
def open_recorder_run(
    root: Path,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[RecorderRun]:
    """Bind one exact recorder run and verify it before and after use."""

    run_root = Path(root)
    with runtime_resource_policy_scope(resource_policy) as policy:
        with _raw_path_session(run_root) as verify_root:
            manifest, manifest_identity = _read_run_manifest(run_root, policy)
            files = _capture_run_files(run_root, manifest, policy)
            retained_bound = (
                _json_record_working_set_bytes(manifest_identity.size) + 64 * 1024
            )
            run = RecorderRun(
                root=run_root,
                manifest=manifest,
                source_identity={
                    "schema_id": "qrp-binance-recorder",
                    "artifact_id": manifest["run_id"],
                    "manifest_hash": canonical_json_hash(manifest),
                },
                resource_policy=policy,
                _files=files,
            )
            try:
                with policy.lease("binance.replay.run_metadata", retained_bound):
                    yield run
            finally:
                primary = sys.exception()
                try:
                    verify_root()
                    current_manifest, current_identity = _read_run_manifest(
                        run_root,
                        policy,
                    )
                    if (
                        current_identity != manifest_identity
                        or canonical_json_hash(current_manifest)
                        != run.source_identity["manifest_hash"]
                    ):
                        raise ValueError("binance_replay_run_identity_changed")
                    current_files = _capture_run_files(
                        run_root,
                        current_manifest,
                        policy,
                    )
                    if tuple(item.identity for item in current_files) != tuple(
                        item.identity for item in files
                    ):
                        raise ValueError("binance_replay_file_identity_changed")
                except BaseException as verify_error:
                    if primary is not None:
                        primary.add_note(
                            "Binance recorder run verification also failed: "
                            f"{type(verify_error).__name__}"
                        )
                    else:
                        raise


def _read_run_manifest(
    root: Path,
    policy: RuntimeResourcePolicy,
) -> tuple[dict[str, Any], DatasetFileIdentity]:
    path = root / "run.json"
    identity = _capture_raw_file_identity(path)
    if identity.size <= 0:
        raise ValueError("binance_replay_run_manifest_invalid")
    bound = _json_record_working_set_bytes(identity.size) + 64 * 1024
    with policy.lease("binance.replay.manifest", bound):
        try:
            raw = path.read_bytes()
            manifest = json.loads(raw.decode("utf-8", errors="strict"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("binance_replay_run_manifest_invalid") from exc
        if _capture_raw_file_identity(path) != identity:
            raise ValueError("binance_replay_run_identity_changed")
        _validate_run_manifest(manifest)
        return manifest, identity


def _validate_run_manifest(value: object) -> None:
    if not isinstance(value, dict) or value.get("schema_id") != "qrp-binance-recorder":
        raise ValueError("binance_replay_run_manifest_identity_invalid")
    for key in ("run_id", "venue", "symbol", "status", "market_data_status"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError("binance_replay_run_manifest_invalid")
    if value["venue"] != "binance" or value["symbol"] != "BTCUSDT":
        raise ValueError("binance_replay_run_manifest_identity_invalid")
    if value["market_data_status"] not in {"PASS", "PARTIAL", "FAIL"}:
        raise ValueError("binance_replay_run_manifest_invalid")
    market_types = value.get("market_types")
    if (
        not isinstance(market_types, list)
        or not market_types
        or any(
            market not in {SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE}
            for market in market_types
        )
        or len(set(market_types)) != len(market_types)
    ):
        raise ValueError("binance_replay_run_manifest_invalid")
    market_metadata = value.get("market_metadata")
    if not isinstance(market_metadata, dict) or set(market_metadata) != set(
        market_types
    ):
        raise ValueError("binance_replay_market_metadata_invalid")
    for _market, metadata in market_metadata.items():
        if not isinstance(metadata, dict) or set(metadata) != {
            "tick_size",
            "snapshot_limit",
        }:
            raise ValueError("binance_replay_market_metadata_invalid")
        validate_tick_size(metadata["tick_size"])
        limit = metadata["snapshot_limit"]
        if isinstance(limit, bool) or type(limit) is not int or limit <= 0:
            raise ValueError("binance_replay_market_metadata_invalid")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("binance_replay_run_files_missing")
    normalized: set[str] = set()
    for item in files:
        _validate_file_manifest(item, markets=set(market_types))
        path = Path(item["path"])
        normalized_path = path.as_posix()
        if normalized_path != item["path"] or normalized_path in normalized:
            raise ValueError("binance_replay_file_manifest_invalid")
        normalized.add(normalized_path)


def _validate_file_manifest(value: object, *, markets: set[str]) -> None:
    if not isinstance(value, dict) or set(value) != _FILE_MANIFEST_KEYS:
        raise ValueError("binance_replay_file_manifest_invalid")
    path = Path(value["path"]) if isinstance(value.get("path"), str) else None
    if (
        path is None
        or path.is_absolute()
        or ".." in path.parts
        or value.get("venue") != "binance"
        or value.get("symbol") != "BTCUSDT"
        or value.get("market") not in markets
        or not isinstance(value.get("stream"), str)
        or not value["stream"]
        or value.get("data_kind")
        not in {"metadata", "depth", "depth_snapshot", "agg_trades"}
    ):
        raise ValueError("binance_replay_file_manifest_invalid")
    for key in (
        "record_count",
        "capture_seq_min",
        "capture_seq_max",
        "max_record_bytes",
        "compressed_bytes",
    ):
        item = value.get(key)
        if isinstance(item, bool) or type(item) is not int or item <= 0:
            raise ValueError("binance_replay_file_manifest_invalid")
    if value["capture_seq_min"] > value["capture_seq_max"]:
        raise ValueError("binance_replay_file_manifest_invalid")
    digest = value.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in _LOWER_HEX for character in digest)
    ):
        raise ValueError("binance_replay_file_manifest_invalid")


def _capture_run_files(
    root: Path,
    manifest: dict[str, Any],
    policy: RuntimeResourcePolicy,
) -> tuple[_RunFile, ...]:
    root_resolved = root.resolve()
    actual_paths: set[str] = set()
    for path in root.rglob("*.jsonl.gz"):
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError as exc:
            raise ValueError("binance_replay_file_path_invalid") from exc
        _capture_raw_file_identity(path, missing_is_file_not_found=False)
        actual_paths.add(path.relative_to(root).as_posix())
    declared_paths = {item["path"] for item in manifest["files"]}
    if actual_paths != declared_paths:
        raise ValueError("binance_replay_file_manifest_mismatch")
    result = []
    for metadata in manifest["files"]:
        path = root / metadata["path"]
        identity = _capture_raw_file_identity(path, missing_is_file_not_found=False)
        if identity.size != metadata["compressed_bytes"]:
            raise ValueError("binance_replay_file_identity_mismatch")
        if _sha256_file(path, identity, policy) != metadata["sha256"]:
            raise ValueError("binance_replay_file_identity_mismatch")
        result.append(_RunFile(dict(metadata), identity))
    return tuple(result)


def _sha256_file(
    path: Path,
    identity: DatasetFileIdentity,
    policy: RuntimeResourcePolicy,
) -> str:
    with policy.lease("binance.replay.file_hash", _HASH_BUFFER_BYTES + 64 * 1024):
        try:
            handle = path.open("rb")
        except OSError as exc:
            raise ValueError("binance_replay_file_identity_mismatch") from exc
        try:
            _verify_raw_file_handle(handle, expected_identity=identity)
            digest = hashlib.sha256()
            buffer = bytearray(_HASH_BUFFER_BYTES)
            while count := handle.readinto(buffer):
                digest.update(memoryview(buffer)[:count])
            _verify_raw_file_handle(handle, expected_identity=identity)
            return digest.hexdigest()
        finally:
            handle.close()


def _overlapping_capture_groups(files: list[_RunFile]) -> Iterator[list[_RunFile]]:
    ordered = sorted(files, key=lambda item: item.metadata["capture_seq_min"])
    group: list[_RunFile] = []
    group_max = -1
    owners: set[tuple[str, str, str]] = set()
    for item in ordered:
        start = item.metadata["capture_seq_min"]
        owner = (
            item.metadata["market"],
            item.metadata["data_kind"],
            item.metadata["stream"],
        )
        if group and start > group_max:
            yield group
            group = []
            owners = set()
            group_max = -1
        if owner in owners:
            raise ValueError("binance_replay_file_capture_ranges_overlap")
        owners.add(owner)
        group.append(item)
        group_max = max(group_max, item.metadata["capture_seq_max"])
    if group:
        yield group


def _iter_raw_file(
    root: Path,
    item: _RunFile,
    *,
    manifest: dict[str, Any],
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[dict[str, Any]]:
    metadata = item.metadata
    path = root / metadata["path"]
    bound = _json_record_working_set_bytes(metadata["max_record_bytes"]) + 64 * 1024
    try:
        raw_handle = path.open("rb")
    except OSError as exc:
        raise ValueError("binance_replay_raw_file_open_failed") from exc
    try:
        _verify_raw_file_handle(raw_handle, expected_identity=item.identity)
        gzip_handle = gzip.GzipFile(fileobj=raw_handle, mode="rb")
        try:
            count = 0
            minimum: int | None = None
            maximum: int | None = None
            previous: int | None = None
            while True:
                with resource_policy.lease("binance.replay.record", bound):
                    line = gzip_handle.readline(metadata["max_record_bytes"] + 1)
                    if not line:
                        break
                    if len(line) > metadata["max_record_bytes"]:
                        raise ValueError("binance_replay_raw_record_too_large")
                    try:
                        record = json.loads(line.decode("utf-8", errors="strict"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        raise ValueError("binance_replay_raw_json_invalid") from exc
                    _validate_raw_record(record, metadata=metadata)
                    capture_seq = record["capture_seq"]
                    if previous is not None and capture_seq <= previous:
                        raise ValueError("binance_replay_capture_seq_non_monotonic")
                    previous = capture_seq
                    minimum = (
                        capture_seq if minimum is None else min(minimum, capture_seq)
                    )
                    maximum = (
                        capture_seq if maximum is None else max(maximum, capture_seq)
                    )
                    count += 1
                    market_metadata = manifest["market_metadata"][metadata["market"]]
                    yield {
                        **record,
                        "run_id": manifest["run_id"],
                        "venue": metadata["venue"],
                        "symbol": metadata["symbol"],
                        "market_type": metadata["market"],
                        "stream": metadata["stream"],
                        "data_kind": metadata["data_kind"],
                        "tick_size": market_metadata["tick_size"],
                        "snapshot_limit": market_metadata["snapshot_limit"],
                    }
            if (
                count != metadata["record_count"]
                or minimum != metadata["capture_seq_min"]
                or maximum != metadata["capture_seq_max"]
            ):
                raise ValueError("binance_replay_file_manifest_mismatch")
        finally:
            gzip_handle.close()
        _verify_raw_file_handle(raw_handle, expected_identity=item.identity)
    finally:
        raw_handle.close()


def _validate_raw_record(value: object, *, metadata: dict[str, Any]) -> None:
    required = {"capture_seq", "receive_time", "receive_monotonic_ns", "payload"}
    optional = {"connection_id", "segment_id"}
    if (
        not isinstance(value, dict)
        or not required.issubset(value)
        or set(value).difference(required, optional)
    ):
        raise ValueError("binance_replay_raw_record_invalid")
    strict_non_negative_integer(
        value["capture_seq"], error="binance_replay_capture_seq_missing"
    )
    strict_iso_timestamp(
        value["receive_time"], error="binance_replay_receive_time_invalid"
    )
    strict_non_negative_integer(
        value["receive_monotonic_ns"],
        error="binance_replay_receive_monotonic_invalid",
    )
    if metadata["data_kind"] in {"depth", "depth_snapshot"}:
        for key in ("connection_id", "segment_id"):
            if not isinstance(value.get(key), str) or not value[key]:
                raise ValueError("binance_replay_segment_identity_invalid")


def _levels(
    value: object,
    *,
    tick_size: str,
    error: str,
    allow_zero: bool,
    max_levels: int,
) -> dict[int, float]:
    if not isinstance(value, list) or len(value) > max_levels:
        raise ValueError(error)
    result: dict[int, float] = {}
    for level in value:
        if not isinstance(level, list) or len(level) != 2:
            raise ValueError(error)
        price_tick = price_to_tick(level[0], tick_size, error=error)
        quantity = strict_non_negative_float(level[1], error=error)
        if not allow_zero and quantity == 0:
            raise ValueError(error)
        if price_tick in result:
            raise ValueError(error)
        result[price_tick] = quantity
    return result


class DepthReconstructor:
    """One segment-local Spot or Futures deterministic order book."""

    def __init__(
        self,
        market_type: str,
        *,
        tick_size: str,
        snapshot_limit: int,
        required_levels: int,
        max_book_levels: int,
        resource_policy: RuntimeResourcePolicy,
        stats: ReconstructionStats,
    ) -> None:
        if market_type not in {SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE}:
            raise ValueError("binance_replay_market_type_invalid")
        if type(required_levels) is not int or required_levels <= 0:
            raise ValueError("binance_replay_required_levels_invalid")
        if type(snapshot_limit) is not int or snapshot_limit < required_levels:
            raise ValueError("binance_replay_snapshot_coverage_invalid")
        if type(max_book_levels) is not int or max_book_levels < snapshot_limit:
            raise ValueError("binance_replay_book_bound_invalid")
        self.market_type = market_type
        self.tick_size = validate_tick_size(tick_size)
        self.required_levels = required_levels
        self.max_book_levels = max_book_levels
        self.policy = resource_policy
        self.stats = stats
        self._book_lease = resource_policy.lease(
            "binance.replay.book",
            4 * max_book_levels * _LEVEL_BYTES + 256 * 1024,
        )
        self._bids: dict[int, float] = {}
        self._asks: dict[int, float] = {}
        self._segment_id: str | None = None
        self._connection_id: str | None = None
        self._last_update_id: int | None = None
        self._last_time: int | None = None
        self._bootstrapped = False
        self._invalid = False

    @property
    def valid(self) -> bool:
        return self._bootstrapped and not self._invalid

    def close(self) -> None:
        self._bids.clear()
        self._asks.clear()
        self._book_lease.close()

    def bootstrap(self, record: dict[str, Any]) -> None:
        if record.get("market_type") != self.market_type:
            raise ValueError("binance_replay_market_type_mismatch")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("binance_replay_snapshot_payload_invalid")
        segment_id, connection_id = _record_identity(record)
        bids = _levels(
            payload.get("bids"),
            tick_size=self.tick_size,
            error="binance_replay_snapshot_bids_invalid",
            allow_zero=False,
            max_levels=self.max_book_levels,
        )
        asks = _levels(
            payload.get("asks"),
            tick_size=self.tick_size,
            error="binance_replay_snapshot_asks_invalid",
            allow_zero=False,
            max_levels=self.max_book_levels,
        )
        reason = self._invalid_reason(bids, asks)
        if reason is not None:
            raise ValueError(f"binance_replay_snapshot_{reason}")
        self._bids = bids
        self._asks = asks
        self._segment_id = segment_id
        self._connection_id = connection_id
        self._last_update_id = strict_non_negative_integer(
            payload.get("lastUpdateId"),
            error="binance_replay_snapshot_id_invalid",
        )
        self._last_time = None
        self._bootstrapped = False
        self._invalid = False

    def feed(self, record: dict[str, Any]) -> Iterator[BookState | BookBoundary]:
        if self._invalid:
            return
        if record.get("market_type") != self.market_type:
            raise ValueError("binance_replay_market_type_mismatch")
        segment_id, connection_id = _record_identity(record)
        if segment_id != self._segment_id or connection_id != self._connection_id:
            raise ValueError("binance_replay_segment_identity_mismatch")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("binance_replay_delta_payload_invalid")
        first = strict_non_negative_integer(
            payload.get("U"), error="binance_replay_delta_id_invalid"
        )
        last = strict_non_negative_integer(
            payload.get("u"), error="binance_replay_delta_id_invalid"
        )
        if first > last:
            raise ValueError("binance_replay_delta_id_range_invalid")
        assert self._last_update_id is not None
        stale = (
            last <= self._last_update_id
            if self.market_type == SPOT_MARKET_TYPE or self._bootstrapped
            else last < self._last_update_id
        )
        if stale:
            self.stats.stale_events += 1
            if last == self._last_update_id:
                self.stats.duplicate_events += 1
            return
        if not self._bootstrapped:
            target = (
                self._last_update_id
                if self.market_type == FUTURES_MARKET_TYPE
                else self._last_update_id + 1
            )
            if not first <= target <= last:
                if first > target:
                    yield self._invalidate(record, "sequence_gap")
                return
        elif self.market_type == FUTURES_MARKET_TYPE:
            if payload.get("pu") != self._last_update_id:
                yield self._invalidate(record, "sequence_gap")
                return
        elif first > self._last_update_id + 1:
            yield self._invalidate(record, "sequence_gap")
            return
        bid_updates = _levels(
            payload.get("b"),
            tick_size=self.tick_size,
            error="binance_replay_delta_bids_invalid",
            allow_zero=True,
            max_levels=self.max_book_levels,
        )
        ask_updates = _levels(
            payload.get("a"),
            tick_size=self.tick_size,
            error="binance_replay_delta_asks_invalid",
            allow_zero=True,
            max_levels=self.max_book_levels,
        )
        if (
            _projected_level_count(self._bids, bid_updates) > self.max_book_levels
            or _projected_level_count(self._asks, ask_updates) > self.max_book_levels
        ):
            yield self._invalidate(record, "book_bound_exceeded")
            return
        _apply_levels(self._bids, bid_updates)
        _apply_levels(self._asks, ask_updates)
        reason = self._invalid_reason(self._bids, self._asks)
        if reason is not None:
            yield self._invalidate(record, reason)
            return
        event_time, transaction_time, research_time = _record_times(
            record,
            market_type=self.market_type,
        )
        if self._last_time is not None and research_time < self._last_time:
            self.stats.non_monotonic_timestamps += 1
            raise ValueError("binance_replay_research_time_non_monotonic")
        self._last_time = research_time
        self._last_update_id = last
        if not self._bootstrapped:
            self._bootstrapped = True
            self.stats.valid_segments += 1
            if self.stats.valid_segments > 1:
                self.stats.recovered_segments += 1
        previous = self.stats.coverage_segments.get(segment_id)
        self.stats.coverage_segments[segment_id] = (
            research_time if previous is None else previous[0],
            research_time,
        )
        state_bound = 64 * 1024 + (len(self._bids) + len(self._asks)) * _LEVEL_BYTES
        with self.policy.lease("binance.replay.state", state_bound):
            state = BookState(
                capture_seq=record["capture_seq"],
                event_time_ms=event_time,
                transaction_time_ms=transaction_time,
                receive_time=record["receive_time"],
                receive_monotonic_ns=record["receive_monotonic_ns"],
                update_id=last,
                segment_id=segment_id,
                connection_id=connection_id,
                market_type=self.market_type,
                tick_size=self.tick_size,
                bids=tuple(sorted(self._bids.items(), reverse=True)),
                asks=tuple(sorted(self._asks.items())),
            )
            self.stats.emitted_states += 1
            yield state

    def _invalid_reason(
        self,
        bids: dict[int, float],
        asks: dict[int, float],
    ) -> str | None:
        if len(bids) < self.required_levels or len(asks) < self.required_levels:
            if not bids or not asks:
                self.stats.one_sided_books += 1
                return "one_sided_book"
            self.stats.unsupported_coverage += 1
            return "unsupported_coverage"
        if max(bids) >= min(asks):
            self.stats.crossed_books += 1
            return "crossed_book"
        return None

    def _invalidate(self, record: dict[str, Any], reason: str) -> BookBoundary:
        event_time, transaction_time, _ = _record_times(
            record,
            market_type=self.market_type,
        )
        segment_id, connection_id = _record_identity(record)
        self._invalid = True
        self._bootstrapped = False
        self._bids.clear()
        self._asks.clear()
        self.stats.invalid_segments += 1
        if reason == "sequence_gap":
            self.stats.gaps += 1
        return BookBoundary(
            capture_seq=record["capture_seq"],
            event_time_ms=event_time,
            transaction_time_ms=transaction_time,
            receive_time=record["receive_time"],
            receive_monotonic_ns=record["receive_monotonic_ns"],
            segment_id=segment_id,
            connection_id=connection_id,
            market_type=self.market_type,
            reason=reason,
        )


def _record_identity(record: dict[str, Any]) -> tuple[str, str]:
    segment_id = record.get("segment_id")
    connection_id = record.get("connection_id")
    if not isinstance(segment_id, str) or not segment_id:
        raise ValueError("binance_replay_segment_identity_invalid")
    if not isinstance(connection_id, str) or not connection_id:
        raise ValueError("binance_replay_connection_identity_invalid")
    return segment_id, connection_id


def _record_times(
    record: dict[str, Any],
    *,
    market_type: str,
) -> tuple[int, int | None, int]:
    payload = record["payload"]
    event_time = strict_non_negative_integer(
        payload.get("E"), error="binance_replay_event_time_invalid"
    )
    transaction_time = (
        strict_non_negative_integer(
            payload.get("T"), error="binance_replay_transaction_time_invalid"
        )
        if market_type == FUTURES_MARKET_TYPE
        else None
    )
    return (
        event_time,
        transaction_time,
        transaction_time if transaction_time is not None else event_time,
    )


def _projected_level_count(
    current: dict[int, float],
    updates: dict[int, float],
) -> int:
    result = len(current)
    for price_tick, quantity in updates.items():
        if quantity == 0 and price_tick in current:
            result -= 1
        elif quantity > 0 and price_tick not in current:
            result += 1
    return result


def _apply_levels(target: dict[int, float], updates: dict[int, float]) -> None:
    for price_tick, quantity in updates.items():
        if quantity == 0:
            target.pop(price_tick, None)
        else:
            target[price_tick] = quantity


def iter_reconstructed_depth(
    run: RecorderRun,
    market_type: str,
    *,
    required_levels: int = 10,
    max_book_levels: int = 10_000,
    stats: ReconstructionStats | None = None,
) -> Iterator[BookState | BookBoundary]:
    """Replay one venue book with snapshots indexed on bounded local scratch."""

    if market_type not in run.manifest["market_types"]:
        raise ValueError("binance_replay_market_type_missing")
    metadata = run.manifest["market_metadata"][market_type]
    active_stats = stats if stats is not None else ReconstructionStats()
    with tempfile.TemporaryDirectory(prefix="qrp-binance-snapshots-") as scratch:
        database = sqlite3.connect(Path(scratch) / "snapshots.sqlite3")
        reconstructor: DepthReconstructor | None = None
        active_segment: str | None = None
        try:
            database.execute(
                "CREATE TABLE snapshots "
                "(segment_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )
            for record in run.iter_records(
                market_type=market_type,
                data_kinds=("depth_snapshot",),
            ):
                segment_id, _ = _record_identity(record)
                try:
                    database.execute(
                        "INSERT INTO snapshots VALUES (?, ?)",
                        (
                            segment_id,
                            json.dumps(record, separators=(",", ":"), allow_nan=False),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        "binance_replay_snapshot_identity_duplicate"
                    ) from exc
            database.commit()
            for record in run.iter_records(
                market_type=market_type,
                data_kinds=("depth",),
            ):
                segment_id, _ = _record_identity(record)
                if segment_id != active_segment:
                    if reconstructor is not None:
                        if reconstructor.valid:
                            event_time, transaction_time, _ = _record_times(
                                record,
                                market_type=market_type,
                            )
                            active_stats.reconnect_boundaries += 1
                            yield BookBoundary(
                                capture_seq=record["capture_seq"],
                                event_time_ms=event_time,
                                transaction_time_ms=transaction_time,
                                receive_time=record["receive_time"],
                                receive_monotonic_ns=record["receive_monotonic_ns"],
                                segment_id=active_segment or segment_id,
                                connection_id=reconstructor._connection_id or "",
                                market_type=market_type,
                                reason="reconnect_boundary",
                            )
                        reconstructor.close()
                    row = database.execute(
                        "SELECT payload FROM snapshots WHERE segment_id = ?",
                        (segment_id,),
                    ).fetchone()
                    if row is None:
                        raise ValueError("binance_replay_snapshot_missing")
                    snapshot = json.loads(row[0])
                    reconstructor = DepthReconstructor(
                        market_type,
                        tick_size=metadata["tick_size"],
                        snapshot_limit=metadata["snapshot_limit"],
                        required_levels=required_levels,
                        max_book_levels=max_book_levels,
                        resource_policy=run.resource_policy,
                        stats=active_stats,
                    )
                    reconstructor.bootstrap(snapshot)
                    active_segment = segment_id
                assert reconstructor is not None
                yield from reconstructor.feed(record)
        finally:
            primary = sys.exception()
            if reconstructor is not None:
                try:
                    reconstructor.close()
                except BaseException as cleanup_error:
                    if primary is not None:
                        primary.add_note(
                            "Binance book release also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                    else:
                        raise
            try:
                database.close()
            except BaseException as cleanup_error:
                if primary is not None:
                    primary.add_note(
                        "Binance snapshot index close also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
                else:
                    raise
