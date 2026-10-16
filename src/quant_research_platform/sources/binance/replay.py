"""Exact recorder-run replay and deterministic Binance MBP reconstruction."""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import heapq
import json
import os
import re
import stat
from bisect import bisect_left, insort
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, fields
from decimal import Decimal
from fractions import Fraction
from itertools import groupby, islice
from pathlib import Path
from typing import Any

from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeBoundary,
    TradeEvent,
    _price_to_index,
    observation_order_key,
    price_to_index,
    validate_price_unit,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.sources.binance.ids import (
    FUTURES_HFT_RECORDING_SCOPE,
    FUTURES_MARKET_TYPE,
    RECORDING_SCOPE_MARKET_TYPES,
    RESEARCH_SYMBOLS,
    SPOT_MARKET_TYPE,
    WS_STREAM_SUFFIXES,
)
from quant_research_platform.sources.binance.validation import (
    _visible_trade_quantity,
    strict_boolean,
    strict_iso_timestamp,
    strict_non_negative_float,
    strict_non_negative_integer,
    strict_positive_float,
    validate_market_payload,
)
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    _DigestingReader,
    dataset_file_identity,
    dataset_file_identity_from_stat,
    filesystem_scandir,
    filesystem_stat,
    open_filesystem_file,
    require_unaliased_path,
)
from quant_research_platform.storage.manifests import read_qrp_json_file

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
_FILE_STREAM_IDENTITY_KEYS = ("market", "symbol", "data_kind", "stream")
_LOWER_HEX = frozenset("0123456789abcdef")
_HASH_BUFFER_BYTES = 64 * 1024
_RAW_IDENTITY_ERROR = "binance_raw_export_identity_changed"
RECONSTRUCTION_REQUIRED_LEVELS = 10
_CONNECTION_INTERVAL_KEYS = frozenset(
    {
        "market",
        "data_kind",
        "stream",
        "connection_id",
        "segment_id",
        "opened_at",
        "opened_monotonic_ns",
        "closed_at",
        "closed_monotonic_ns",
        "close_reason",
    }
)
_CONNECTION_CLOSE_REASONS = frozenset(
    {"stop", "connection_problem", "depth_resync", "failure"}
)


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

    @classmethod
    def from_payload(cls, value):
        if type(value) is not dict or set(value) != {item.name for item in fields(cls)}:
            raise ValueError("binance_reconstruction_diagnostics_invalid")
        if any(
            type(count) is not int or count < 0
            for key, count in value.items()
            if key != "coverage_segments"
        ):
            raise ValueError("binance_reconstruction_diagnostics_invalid")
        coverage = value["coverage_segments"]
        if type(coverage) is not dict or any(
            type(key) is not str
            or not key
            or type(bounds) is not list
            or len(bounds) != 2
            or any(type(bound) is not int for bound in bounds)
            or not 0 <= bounds[0] < bounds[1]
            for key, bounds in coverage.items()
        ):
            raise ValueError("binance_reconstruction_diagnostics_invalid")
        return cls(
            **{
                **value,
                "coverage_segments": {
                    key: tuple(bounds) for key, bounds in coverage.items()
                },
            }
        )


@dataclass(frozen=True)
class _RunFile:
    metadata: dict[str, Any]
    identity: DatasetFileIdentity


class ObservationCoverageIndex:
    def __init__(
        self,
        root: Path,
        descriptor: dict[str, Any],
        *,
        markets: set[str],
        decode_interval: Callable,
    ) -> None:
        self._decode_interval = decode_interval
        self._root = root
        self._descriptor = descriptor
        self._path = root / descriptor["path"]
        self.identity = _capture_raw_file_identity(
            self._path,
            missing_is_file_not_found=False,
        )
        if self.identity.size != descriptor["compressed_bytes"]:
            raise ValueError("binance_replay_connection_intervals_invalid")
        self._rows = {}
        self._groups = {}
        self._invalid_trade_times = {}
        self._load(markets=markets)
        for row in self._rows.values():
            observation_id, market, kind, stream, _segment, opened, closed = row
            key = (market, kind, stream.split("@", 1)[0])
            self._groups.setdefault(key, []).append((opened, closed, observation_id))
        for rows in self._groups.values():
            rows.sort()

    def _load(self, *, markets: set[str]) -> None:
        try:
            raw_handle = open_filesystem_file(self._path, "rb")
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError("binance_replay_connection_intervals_invalid") from exc
        with closing_iterator(raw_handle):
            digesting_handle = _DigestingReader(raw_handle)
            count = 0
            with closing_iterator(
                gzip.GzipFile(fileobj=digesting_handle, mode="rb")
            ) as gzip_handle:
                while True:
                    line: bytes | None = None
                    interval: dict[str, Any] | None = None
                    row = None
                    line = gzip_handle.readline()
                    if not line:
                        break
                    try:
                        interval = parse_qrp_json_object(line)
                    except (UnicodeError, ValueError) as exc:
                        raise ValueError(
                            "binance_replay_connection_intervals_invalid"
                        ) from exc
                    row = self._decode_interval(interval, markets=markets)
                    if row[0] in self._rows:
                        raise ValueError("binance_replay_connection_intervals_invalid")
                    self._rows[row[0]] = row
                    count += 1
            digesting_handle.drain()
            if (
                count != self._descriptor["record_count"]
                or digesting_handle.hexdigest() != self._descriptor["sha256"]
            ):
                raise ValueError("binance_replay_connection_intervals_invalid")
            _verify_raw_file_handle(raw_handle, expected_identity=self.identity)

    def bounds(
        self,
        observation_id: object,
        *,
        symbol: str,
        market_type: str,
        data_kind: str,
    ) -> tuple[int, int] | None:
        row = self._rows.get(observation_id)
        if (
            row is None
            or row[1:3] != (market_type, data_kind)
            or not row[3].startswith(symbol.lower() + "@")
        ):
            return None
        return row[5], row[6]

    def segment_id(self, observation_id: object) -> str | None:
        row = self._rows.get(observation_id)
        return None if row is None else row[4]

    def invalidate_trade(self, observation_id, observed_ns):
        """Record only a consumed invalid event; authenticated sidecars stay intact."""
        times = self._invalid_trade_times.setdefault(observation_id, [])
        if observed_ns not in times:
            insort(times, observed_ns)

    def trade_intervals(self, observation_id):
        row = self._rows.get(observation_id)
        if row is None:
            return
        start, stop = row[5:7]
        identity = observation_id
        for observed in self._invalid_trade_times.get(observation_id, ()):
            if start <= observed:
                yield start, observed, identity
            start = observed + 1
            identity = f"{observation_id}:after_invalid:{observed}"
        if start <= stop:
            yield start, stop, identity

    def trade_bounds(self, observation_id, *, symbol, market_type):
        base = observation_id
        if base not in self._rows and isinstance(base, str):
            base = base.rsplit(":after_invalid:", 1)[0]
        if (
            self.bounds(
                base, symbol=symbol, market_type=market_type, data_kind="agg_trades"
            )
            is None
        ):
            return None
        return next(
            (
                (start, stop)
                for start, stop, identity in self.trade_intervals(base)
                if identity == observation_id
            ),
            None,
        )

    def trade_observation_id(self, observation_id, observed_ns):
        return next(
            (
                identity
                for start, stop, identity in self.trade_intervals(observation_id)
                if start <= observed_ns < stop
            ),
            observation_id,
        )


class ObservationCoverage:
    """Exact lookups and a single forward sweep over sorted connection intervals."""

    def __init__(self, index, market_type, data_kind, symbol):
        self.index = index
        self.market_type = market_type
        self.data_kind = data_kind
        self.symbol = symbol
        self.rows = index._groups.get((market_type, data_kind, symbol.lower()), ())
        self.position = 0
        self.active = []
        self.last_bounds = None

    def get(self, observation_id, default=None):
        row = (
            self.index.trade_bounds(
                observation_id, symbol=self.symbol, market_type=self.market_type
            )
            if self.data_kind == "agg_trades"
            else self.index.bounds(
                observation_id,
                symbol=self.symbol,
                market_type=self.market_type,
                data_kind=self.data_kind,
            )
        )
        return default if row is None else row

    def __getitem__(self, observation_id):
        row = self.get(observation_id)
        if row is None:
            raise KeyError(observation_id)
        return row

    def covering(self, start_ns, end_ns):
        if self.last_bounds is not None and (
            start_ns < self.last_bounds[0] or end_ns < self.last_bounds[1]
        ):
            raise ValueError("binance_coverage_window_non_monotonic")
        self.last_bounds = (start_ns, end_ns)
        while self.active and self.active[0][0] <= start_ns:
            heapq.heappop(self.active)
        while self.position < len(self.rows) and self.rows[self.position][0] <= end_ns:
            opened_ns, closed_ns, observation_id = self.rows[self.position]
            self.position += 1
            if closed_ns > start_ns and opened_ns < closed_ns:
                heapq.heappush(self.active, (closed_ns, observation_id, opened_ns))
        if len(self.active) != 1:
            return ()
        closed_ns, observation_id, opened_ns = self.active[0]
        if self.data_kind == "agg_trades":
            return tuple(
                identity
                for opened, closed, identity in self.index.trade_intervals(
                    observation_id
                )
                if opened <= start_ns and closed > end_ns
            )
        return (observation_id,) if opened_ns <= start_ns and closed_ns > end_ns else ()

    def intervals(self):
        """Yield only unambiguous half-open coverage in chronological order."""
        rows = (
            (
                interval
                for _, _, identity in self.rows
                for interval in self.index.trade_intervals(identity)
            )
            if self.data_kind == "agg_trades"
            else self.rows
        )
        events = sorted(
            (timestamp, delta, identity)
            for opened, closed, identity in rows
            if opened < closed
            for timestamp, delta in ((opened, 1), (closed, -1))
        )
        active = set()
        previous = None
        for timestamp, changes in groupby(events, key=lambda row: row[0]):
            if previous is not None and previous < timestamp and len(active) == 1:
                yield previous, timestamp
            for _stamp, delta, identity in changes:
                if delta > 0:
                    active.add(identity)
                else:
                    active.remove(identity)
            previous = timestamp


@dataclass
class RecorderRun:
    root: Path
    manifest: dict[str, Any]
    source_identity: dict[str, str]
    _files: tuple[_RunFile, ...] = field(repr=False)
    _coverage_index: ObservationCoverageIndex = field(repr=False)

    def funding_roots(self, symbol, *, time_window):
        return (self.root.parent / "funding" / symbol,)

    def hft_source_info(self):
        if self.manifest["status"] == "FAIL":
            raise ValueError("hft_recorder_run_failed")
        if (
            self.manifest["recording_scope"] != FUTURES_HFT_RECORDING_SCOPE
            or FUTURES_MARKET_TYPE not in self.manifest["market_types"]
        ):
            raise ValueError("hft_futures_run_missing")
        return {
            "raw_record_count": sum(
                item["record_count"] for item in self.manifest["files"]
            ),
            "source_file_count": len(self.manifest["files"]),
        }

    def market_metadata(self, market_type, symbol):
        if market_type not in self.manifest["market_types"]:
            raise ValueError("binance_replay_market_type_missing")
        if symbol not in self.manifest["symbols"]:
            raise ValueError("binance_replay_symbol_missing")
        native = self.manifest["market_metadata"][symbol][market_type]
        return {
            "price_unit": native["tick_size"],
            "snapshot_limit": native["snapshot_limit"],
        }

    def iter_records(
        self,
        *,
        symbol: str | None = None,
        market_type: str | None = None,
        data_kinds: Sequence[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        kinds = None if data_kinds is None else frozenset(data_kinds)
        selected = [
            item
            for item in self._files
            if (symbol is None or item.metadata["symbol"] == symbol)
            and (market_type is None or item.metadata["market"] == market_type)
            and (kinds is None or item.metadata["data_kind"] in kinds)
        ]
        if not selected:
            raise ValueError("binance_replay_requested_files_missing")

        def owner_records(files):
            for item in files:
                with closing_iterator(
                    _iter_raw_file(self.root, item, manifest=self.manifest)
                ) as records:
                    for _yield_value in records:
                        try:
                            yield _yield_value
                        except GeneratorExit:
                            return

        previous_capture_seq: int | None = None
        for group in _overlapping_capture_groups(selected):
            by_owner: dict[tuple[str, str, str, str], list[_RunFile]] = {}
            for item in group:
                owner = tuple(item.metadata[key] for key in _FILE_STREAM_IDENTITY_KEYS)
                by_owner.setdefault(owner, []).append(item)
            with ExitStack() as stack:
                streams = [
                    stack.enter_context(closing_iterator(owner_records(files)))
                    for files in by_owner.values()
                ]
                records = stack.enter_context(
                    closing_iterator(
                        heapq.merge(*streams, key=lambda value: value["capture_seq"])
                    )
                )
                for record in records:
                    _validate_record_coverage(
                        record,
                        coverage_index=self._coverage_index,
                    )
                    capture_seq = record["capture_seq"]
                    if (
                        previous_capture_seq is not None
                        and capture_seq <= previous_capture_seq
                    ):
                        raise ValueError(
                            "binance_replay_capture_seq_duplicate_or_non_monotonic"
                        )
                    previous_capture_seq = capture_seq
                    try:
                        yield record
                    except GeneratorExit:
                        return

    def iter_market_records(self, *, symbol, market_type, data_kinds, end_ns=None):
        """Decode recorder-specific identity into the shared venue event contract."""
        with ExitStack() as resources:
            records = resources.enter_context(
                closing_iterator(
                    self.iter_records(
                        symbol=symbol, market_type=market_type, data_kinds=data_kinds
                    )
                )
            )
            observe = native_trade_identities() if "agg_trades" in data_kinds else None
            for record in records:
                if end_ns is not None and record["observed_time_ns"] > end_ns:
                    continue
                normalized = normalize_recorder_record(record)
                if normalized["data_kind"] == "agg_trades":
                    accepted, _previous = observe(
                        market_type, symbol, normalized["payload"]
                    )
                    if not accepted:
                        continue
                try:
                    yield normalized
                except GeneratorExit:
                    return

    def observation_coverage(
        self,
        *,
        market_type: str,
        data_kind: str,
        symbol: str,
    ) -> ObservationCoverage:
        if data_kind == "depth_snapshot":
            data_kind = "depth"
        return ObservationCoverage(self._coverage_index, market_type, data_kind, symbol)


def _validate_raw_path(root: Path) -> None:
    """Validate the selected source path once, without a lifetime snapshot."""
    raw_root = require_unaliased_path(Path(root), error=_RAW_IDENTITY_ERROR)
    try:
        metadata = filesystem_stat(raw_root, follow_symlinks=False)
    except NotADirectoryError as error:
        raise ValueError(_RAW_IDENTITY_ERROR) from error
    except FileNotFoundError:
        return
    _reject_raw_reparse(metadata)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(_RAW_IDENTITY_ERROR)


def _capture_raw_file_identity(
    path: Path,
    *,
    expected_identity: DatasetFileIdentity | None = None,
    missing_is_file_not_found: bool = True,
) -> DatasetFileIdentity:
    raw_path = Path(path)
    try:
        filesystem_stat(raw_path, follow_symlinks=False)
    except NotADirectoryError as error:
        raise ValueError(_RAW_IDENTITY_ERROR) from error
    except FileNotFoundError as error:
        if missing_is_file_not_found and expected_identity is None:
            raise FileNotFoundError(raw_path) from error
        raise ValueError(_RAW_IDENTITY_ERROR) from error
    try:
        identity = dataset_file_identity(raw_path)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        raise ValueError(_RAW_IDENTITY_ERROR) from error
    if expected_identity is not None and identity != expected_identity:
        raise ValueError(_RAW_IDENTITY_ERROR)
    return identity


def _verify_raw_file_handle(
    handle: Any,
    *,
    expected_identity: DatasetFileIdentity,
) -> None:
    try:
        handle_identity = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
    except ValueError as error:
        raise ValueError(_RAW_IDENTITY_ERROR) from error
    if handle_identity != expected_identity:
        raise ValueError(_RAW_IDENTITY_ERROR)


def _reject_raw_reparse(metadata: os.stat_result) -> None:
    reparse_flag = getattr(metadata, "st_file_attributes", 0) & getattr(
        stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        0,
    )
    if reparse_flag:
        raise ValueError(_RAW_IDENTITY_ERROR)


@contextmanager
def open_recorder_run(root: Path) -> Iterator[RecorderRun]:
    """Admit one declared recorder source and yield its native observations."""
    run_root = Path(root)
    _validate_raw_path(run_root)
    manifest, manifest_identity = _read_run_manifest(run_root)
    coverage_index = ObservationCoverageIndex(
        run_root,
        manifest["capture"]["connection_intervals"],
        markets=set(manifest["market_types"]),
        decode_interval=_decode_recorder_interval,
    )
    files = _capture_run_files(run_root, manifest)
    run = RecorderRun(
        root=run_root,
        manifest=manifest,
        source_identity={
            "schema_id": "qrp-binance-recorder",
            "artifact_id": manifest["run_id"],
            "manifest_hash": canonical_json_hash(manifest),
        },
        _files=files,
        _coverage_index=coverage_index,
    )
    yield run


def _read_run_manifest(
    root: Path,
) -> tuple[dict[str, Any], DatasetFileIdentity]:
    path = root / "run.json"
    identity = _capture_raw_file_identity(path)
    if identity.size <= 0:
        raise ValueError("binance_replay_run_manifest_invalid")
    manifest: dict[str, Any] | None = None
    try:
        manifest = read_qrp_json_file(path)[1]
    except (
        FileNotFoundError,
        NotADirectoryError,
        UnicodeError,
        ValueError,
    ) as exc:
        if str(exc) in {
            "dataset_metadata_file_identity_changed",
            "dataset_metadata_file_too_large",
        }:
            raise ValueError("binance_replay_run_identity_changed") from exc
        raise ValueError("binance_replay_run_manifest_invalid") from exc
    if _capture_raw_file_identity(path) != identity:
        raise ValueError("binance_replay_run_identity_changed")
    _validate_run_manifest(manifest)
    assert manifest is not None
    return manifest, identity


def _validate_run_manifest(value: object) -> None:
    if not isinstance(value, dict) or value.get("schema_id") != "qrp-binance-recorder":
        raise ValueError("binance_replay_run_manifest_identity_invalid")
    for key in (
        "run_id",
        "recording_scope",
        "venue",
        "status",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError("binance_replay_run_manifest_invalid")
    # This is the recorder vendor manifest identifier, not a research run.
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", value["run_id"]) is None:
        raise ValueError("binance_replay_run_manifest_invalid")
    if (
        value["venue"] != "binance"
        or not isinstance(value.get("symbols"), list)
        or not value["symbols"]
        or any(
            type(symbol) is not str or symbol not in RESEARCH_SYMBOLS
            for symbol in value["symbols"]
        )
        or len(set(value["symbols"])) != len(value["symbols"])
    ):
        raise ValueError("binance_replay_run_manifest_identity_invalid")
    statuses = {"PASS", "PARTIAL", "FAIL"}
    if value["status"] not in statuses:
        raise ValueError("binance_replay_run_manifest_invalid")
    market_types = value.get("market_types")
    recording_scope = value["recording_scope"]
    if (
        recording_scope not in RECORDING_SCOPE_MARKET_TYPES
        or not isinstance(market_types, list)
        or market_types != list(RECORDING_SCOPE_MARKET_TYPES[recording_scope])
    ):
        raise ValueError("binance_replay_run_manifest_invalid")
    market_metadata = value.get("market_metadata")
    if not isinstance(market_metadata, dict) or set(market_metadata) != set(
        value["symbols"]
    ):
        raise ValueError("binance_replay_market_metadata_invalid")
    for symbol_metadata in market_metadata.values():
        if not isinstance(symbol_metadata, dict) or set(symbol_metadata) != set(
            market_types
        ):
            raise ValueError("binance_replay_market_metadata_invalid")
        for metadata in symbol_metadata.values():
            if not isinstance(metadata, dict) or set(metadata) != {
                "tick_size",
                "snapshot_limit",
            }:
                raise ValueError("binance_replay_market_metadata_invalid")
            validate_price_unit(metadata["tick_size"])
            limit = metadata["snapshot_limit"]
            if isinstance(limit, bool) or type(limit) is not int or limit <= 0:
                raise ValueError("binance_replay_market_metadata_invalid")
    capture = value.get("capture")
    if not isinstance(capture, dict):
        raise ValueError("binance_replay_connection_intervals_invalid")
    _validate_connection_interval_descriptor(
        capture.get("connection_intervals"),
    )
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("binance_replay_run_files_missing")
    normalized: set[str] = set()
    present_tuples: set[tuple[str, str, str]] = set()
    for item in files:
        _validate_file_manifest(
            item, markets=set(market_types), symbols=set(value["symbols"])
        )
        path = Path(item["path"])
        normalized_path = path.as_posix()
        if normalized_path != item["path"] or normalized_path in normalized:
            raise ValueError("binance_replay_file_manifest_invalid")
        normalized.add(normalized_path)
        present_tuples.add((item["symbol"], item["market"], item["data_kind"]))
    stream_kinds = (
        {"metadata", "depth", "depth_snapshot", "agg_trades"}
        if recording_scope == "futures-hft"
        else {"metadata", "depth", "depth_snapshot"}
    )
    required_tuples = {
        (symbol, market, data_kind)
        for symbol in value["symbols"]
        for market in market_types
        for data_kind in stream_kinds
    }
    if value["status"] == "PASS" and present_tuples != required_tuples:
        raise ValueError("binance_replay_run_completeness_invalid")


def _validate_file_manifest(
    value: object, *, markets: set[str], symbols: set[str]
) -> None:
    if not isinstance(value, dict) or set(value) != _FILE_MANIFEST_KEYS:
        raise ValueError("binance_replay_file_manifest_invalid")
    path = Path(value["path"]) if isinstance(value.get("path"), str) else None
    if (
        path is None
        or path.anchor
        or ".." in path.parts
        or not path.name.endswith(".jsonl.gz")
        or value.get("venue") != "binance"
        or value.get("symbol") not in symbols
        or value.get("market") not in markets
        or not isinstance(value.get("stream"), str)
        or not value["stream"]
        or value.get("data_kind")
        not in {"metadata", "depth", "depth_snapshot", "agg_trades"}
    ):
        raise ValueError("binance_replay_file_manifest_invalid")
    if value["data_kind"] in WS_STREAM_SUFFIXES and value["stream"] != (
        f"{value['symbol'].lower()}@{WS_STREAM_SUFFIXES[value['data_kind']]}"
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


def _validate_connection_interval_descriptor(
    value: object,
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "record_count",
        "compressed_bytes",
        "sha256",
    }:
        raise ValueError("binance_replay_connection_intervals_invalid")
    path = Path(value["path"]) if isinstance(value.get("path"), str) else None
    if (
        path is None
        or path.anchor
        or ".." in path.parts
        or path.as_posix() != value["path"]
        or path.name != "connection_intervals.jsonl.gz"
    ):
        raise ValueError("binance_replay_connection_intervals_invalid")
    record_count = value.get("record_count")
    compressed_bytes = value.get("compressed_bytes")
    if (
        isinstance(record_count, bool)
        or type(record_count) is not int
        or record_count < 0
        or isinstance(compressed_bytes, bool)
        or type(compressed_bytes) is not int
        or compressed_bytes <= 0
    ):
        raise ValueError("binance_replay_connection_intervals_invalid")
    digest = value.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in _LOWER_HEX for character in digest)
    ):
        raise ValueError("binance_replay_connection_intervals_invalid")


def _validate_connection_interval(
    value: object,
    *,
    markets: set[str],
) -> tuple[int, int]:
    if not isinstance(value, dict) or set(value) != _CONNECTION_INTERVAL_KEYS:
        raise ValueError("binance_replay_connection_intervals_invalid")
    if (
        value.get("market") not in markets
        or value.get("data_kind") not in {"depth", "agg_trades"}
        or not isinstance(value.get("stream"), str)
        or not value["stream"]
        or not isinstance(value.get("connection_id"), str)
        or not value["connection_id"]
    ):
        raise ValueError("binance_replay_connection_intervals_invalid")
    if value["stream"] not in {
        f"{symbol.lower()}@{WS_STREAM_SUFFIXES[value['data_kind']]}"
        for symbol in RESEARCH_SYMBOLS
    }:
        raise ValueError("binance_replay_connection_intervals_invalid")
    if value["data_kind"] == "depth":
        if not isinstance(value.get("segment_id"), str) or not value["segment_id"]:
            raise ValueError("binance_replay_connection_intervals_invalid")
    elif value.get("segment_id") is not None:
        raise ValueError("binance_replay_connection_intervals_invalid")
    opened = strict_iso_timestamp(
        value.get("opened_at"),
        error="binance_replay_connection_intervals_invalid",
    )
    closed = strict_iso_timestamp(
        value.get("closed_at"),
        error="binance_replay_connection_intervals_invalid",
    )
    if closed < opened:
        raise ValueError("binance_replay_connection_intervals_invalid")
    for key in ("opened_monotonic_ns", "closed_monotonic_ns"):
        strict_non_negative_integer(
            value.get(key),
            error="binance_replay_connection_intervals_invalid",
        )
    if value["closed_monotonic_ns"] < value["opened_monotonic_ns"]:
        raise ValueError("binance_replay_connection_intervals_invalid")
    if value["close_reason"] not in _CONNECTION_CLOSE_REASONS:
        raise ValueError("binance_replay_connection_intervals_invalid")
    return opened.value, closed.value


def _decode_recorder_interval(value, *, markets):
    opened_ns, closed_ns = _validate_connection_interval(value, markets=markets)
    return (
        value["connection_id"],
        value["market"],
        value["data_kind"],
        value["stream"],
        value["segment_id"],
        opened_ns,
        closed_ns,
    )


def _validate_record_coverage(
    record: dict[str, Any],
    *,
    coverage_index: ObservationCoverageIndex,
) -> None:
    data_kind = record["data_kind"]
    if data_kind == "metadata":
        return
    interval_kind = "depth" if data_kind == "depth_snapshot" else data_kind
    if interval_kind not in {"depth", "agg_trades"}:
        raise ValueError("binance_replay_connection_coverage_invalid")
    connection_id = record.get("connection_id")
    bounds = coverage_index.bounds(
        connection_id,
        symbol=record["symbol"],
        market_type=record["market_type"],
        data_kind=interval_kind,
    )
    if bounds is None:
        raise ValueError("binance_replay_connection_coverage_invalid")
    if interval_kind == "depth" and record.get(
        "segment_id"
    ) != coverage_index.segment_id(connection_id):
        raise ValueError("binance_replay_connection_coverage_invalid")
    observed_time = record["observed_time_ns"]
    if not bounds[0] <= observed_time < bounds[1]:
        raise ValueError("binance_replay_connection_coverage_invalid")


def _capture_run_files(root: Path, manifest: dict[str, Any]) -> tuple[_RunFile, ...]:
    declared_paths = {item["path"] for item in manifest["files"]} | {
        manifest["capture"]["connection_intervals"]["path"]
    }
    _verify_exact_run_tree(root, declared_paths)
    result = []
    for metadata in manifest["files"]:
        path = root / metadata["path"]
        identity = _capture_raw_file_identity(path, missing_is_file_not_found=False)
        if identity.size != metadata["compressed_bytes"]:
            raise ValueError("binance_replay_file_identity_mismatch")
        result.append(_RunFile(dict(metadata), identity))
    return tuple(result)


def _verify_exact_run_tree(root: Path, declared_paths: set[str]) -> None:
    declared = {Path(value) for value in declared_paths}
    allowed_files = {Path("run.json"), *declared}
    allowed_directories = {Path(".")}
    for path in declared:
        allowed_directories.update(
            parent for parent in path.parents if parent != Path(".")
        )
    seen_files: set[Path] = set()
    pending = [Path(".")]
    while pending:
        relative_directory = pending.pop()
        directory = (
            root if relative_directory == Path(".") else root / relative_directory
        )
        require_unaliased_path(directory, error="binance_replay_file_path_invalid")
        try:
            entries = filesystem_scandir(directory)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError("binance_replay_file_manifest_mismatch") from exc
        with entries:
            for entry in entries:
                relative = (
                    Path(entry.name)
                    if relative_directory == Path(".")
                    else relative_directory / entry.name
                )
                path = directory / entry.name
                require_unaliased_path(path, error="binance_replay_file_path_invalid")
                if relative in allowed_directories:
                    if not entry.is_dir(follow_symlinks=False):
                        raise ValueError("binance_replay_file_manifest_mismatch")
                    pending.append(relative)
                elif relative in allowed_files:
                    if not entry.is_file(follow_symlinks=False):
                        raise ValueError("binance_replay_file_manifest_mismatch")
                    seen_files.add(relative)
                else:
                    raise ValueError("binance_replay_file_manifest_mismatch")
    if (
        seen_files != allowed_files
        or {path.as_posix() for path in seen_files if path != Path("run.json")}
        != declared_paths
    ):
        raise ValueError("binance_replay_file_manifest_mismatch")


def _sha256_file(path: Path, identity: DatasetFileIdentity) -> str:
    try:
        handle = open_filesystem_file(path, "rb")
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("binance_replay_file_identity_mismatch") from exc
    with closing_iterator(handle):
        _verify_raw_file_handle(handle, expected_identity=identity)
        digest = hashlib.sha256()
        buffer = bytearray(_HASH_BUFFER_BYTES)
        while count := handle.readinto(buffer):
            digest.update(memoryview(buffer)[:count])
        _verify_raw_file_handle(handle, expected_identity=identity)
        return digest.hexdigest()


def _overlapping_capture_groups(files: list[_RunFile]) -> Iterator[list[_RunFile]]:
    ordered = sorted(files, key=lambda item: item.metadata["capture_seq_min"])
    group: list[_RunFile] = []
    group_max = -1
    owner_ends: dict[tuple[str, str, str, str], int] = {}
    for item in ordered:
        start = item.metadata["capture_seq_min"]
        owner = tuple(item.metadata[key] for key in _FILE_STREAM_IDENTITY_KEYS)
        if group and start > group_max:
            yield group
            group = []
            group_max = -1
        if start <= owner_ends.get(owner, -1):
            raise ValueError("binance_replay_file_capture_ranges_overlap")
        owner_ends[owner] = item.metadata["capture_seq_max"]
        group.append(item)
        group_max = max(group_max, item.metadata["capture_seq_max"])
    if group:
        yield group


def _iter_raw_file(
    root: Path, item: _RunFile, *, manifest: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    metadata = item.metadata
    path = root / metadata["path"]
    try:
        raw_handle = open_filesystem_file(path, "rb")
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("binance_replay_raw_file_open_failed") from exc
    primary = None
    try:
        _verify_raw_file_handle(raw_handle, expected_identity=item.identity)
        digesting_handle = _DigestingReader(raw_handle)
        gzip_handle = gzip.GzipFile(fileobj=digesting_handle, mode="rb")
        primary = None
        try:
            count = 0
            minimum: int | None = None
            maximum: int | None = None
            previous: int | None = None
            while True:
                line = gzip_handle.readline(metadata["max_record_bytes"] + 1)
                if not line:
                    line = None
                    break
                record: dict[str, Any] | None = None
                if len(line) > metadata["max_record_bytes"]:
                    raise ValueError("binance_replay_raw_record_too_large")
                try:
                    record = parse_qrp_json_object(line)
                except (UnicodeError, ValueError) as exc:
                    raise ValueError("binance_replay_raw_json_invalid") from exc
                _validate_raw_record(record, metadata=metadata)
                capture_seq = record["capture_seq"]
                if previous is not None and capture_seq <= previous:
                    raise ValueError("binance_replay_capture_seq_non_monotonic")
                previous = capture_seq
                minimum = capture_seq if minimum is None else min(minimum, capture_seq)
                maximum = capture_seq if maximum is None else max(maximum, capture_seq)
                count += 1
                market_metadata = manifest["market_metadata"][metadata["symbol"]][
                    metadata["market"]
                ]
                output = {
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
                try:
                    yield output
                except GeneratorExit:
                    return
            if (
                count != metadata["record_count"]
                or minimum != metadata["capture_seq_min"]
                or maximum != metadata["capture_seq_max"]
            ):
                raise ValueError("binance_replay_file_manifest_mismatch")
        except BaseException as body_error:
            primary = body_error
            raise
        finally:
            try:
                gzip_handle.close()
                digesting_handle.drain()
                if digesting_handle.hexdigest() != metadata["sha256"]:
                    raise ValueError("binance_replay_file_identity_mismatch")
                _verify_raw_file_handle(raw_handle, expected_identity=item.identity)
            except BaseException as verify_error:
                if primary is not None:
                    primary.add_note(
                        "Binance consumed-file verification also failed: "
                        f"{type(verify_error).__name__}"
                    )
                else:
                    raise

    except BaseException as body_error:
        primary = body_error
        raise
    finally:
        try:
            raw_handle.close()
        except BaseException as close_error:
            if primary is None:
                raise
            primary.add_note(
                "Binance replay raw handle close also failed: "
                f"{type(close_error).__name__}"
            )


def _validate_raw_record(value: object, *, metadata: dict[str, Any]) -> None:
    required = {
        "capture_seq",
        "receive_time",
        "observed_time_ns",
        "receive_monotonic_ns",
        "payload",
    }
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
    observed_time_ns = strict_iso_timestamp(
        value["receive_time"], error="binance_replay_receive_time_invalid"
    ).value
    if value["observed_time_ns"] != observed_time_ns:
        raise ValueError("binance_replay_observed_time_invalid")
    strict_non_negative_integer(
        value["observed_time_ns"], error="binance_replay_observed_time_invalid"
    )
    strict_non_negative_integer(
        value["receive_monotonic_ns"],
        error="binance_replay_receive_monotonic_invalid",
    )
    if metadata["data_kind"] in {"depth", "depth_snapshot"}:
        for key in ("connection_id", "segment_id"):
            if not isinstance(value.get(key), str) or not value[key]:
                raise ValueError("binance_replay_segment_identity_invalid")
    elif metadata["data_kind"] == "agg_trades":
        if (
            not isinstance(value.get("connection_id"), str)
            or not value["connection_id"]
        ):
            raise ValueError("binance_replay_connection_identity_invalid")
        if "segment_id" in value:
            raise ValueError("binance_replay_segment_identity_invalid")
    if metadata["data_kind"] in {"depth", "agg_trades"}:
        validate_market_payload(
            metadata["market"],
            metadata["data_kind"],
            metadata["symbol"],
            value["payload"],
            validate_values=False,
        )


def _levels(
    value: object,
    *,
    price_unit: Fraction,
    error: str,
    allow_zero: bool,
) -> dict[int, float]:
    if not isinstance(value, list):
        raise ValueError(error)
    result: dict[int, float] = {}
    for level in value:
        if not isinstance(level, list) or len(level) != 2:
            raise ValueError(error)
        price_index = _price_to_index(level[0], price_unit, error=error)
        quantity = strict_non_negative_float(level[1], error=error)
        if not allow_zero and quantity == 0:
            raise ValueError(error)
        if price_index in result:
            raise ValueError(error)
        result[price_index] = quantity
    return result


def normalize_recorder_record(record):
    """Preserve native recorder clocks while exposing source-neutral identity."""
    result = dict(record)
    result["source_sequence"] = result.pop("capture_seq")
    result["observation_id"] = result.pop("connection_id")
    result["price_unit"] = result.pop("tick_size")
    return result


class DepthReconstructor:
    """One segment-local Spot or Futures deterministic order book."""

    def __init__(
        self,
        market_type: str,
        *,
        price_unit: str,
        snapshot_limit: int,
        required_levels: int,
        connection_close_ns: int | None,
        stats: ReconstructionStats,
        output_levels: int | None = None,
    ) -> None:
        if market_type not in {SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE}:
            raise ValueError("binance_replay_market_type_invalid")
        if type(required_levels) is not int or required_levels <= 0:
            raise ValueError("binance_replay_required_levels_invalid")
        if type(snapshot_limit) is not int or snapshot_limit < required_levels:
            raise ValueError("binance_replay_snapshot_coverage_invalid")
        if connection_close_ns is not None and (
            type(connection_close_ns) is not int or connection_close_ns < 0
        ):
            raise ValueError("binance_replay_connection_coverage_invalid")
        self.market_type = market_type
        self.price_unit = validate_price_unit(price_unit)
        self._price_unit = Fraction(Decimal(self.price_unit))
        self.required_levels = required_levels
        self.output_levels = output_levels
        self.connection_close_ns = connection_close_ns
        self.stats = stats
        self._bids: dict[int, float] = {}
        self._asks: dict[int, float] = {}
        self._segment_id: str | None = None
        self._source_segment_id: str | None = None
        self._observation_id: str | None = None
        self._last_update_id: int | None = None
        self._last_times: tuple[int, int | None, int] | None = None
        self._known_bid_floor_tick: int | None = None
        self._known_ask_ceiling_tick: int | None = None
        self._bootstrapped = False
        self._invalid = False

    @property
    def valid(self) -> bool:
        return self._bootstrapped and not self._invalid

    def bootstrap(self, record: dict[str, Any]) -> None:
        if record.get("market_type") != self.market_type:
            raise ValueError("binance_replay_market_type_mismatch")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("binance_replay_snapshot_payload_invalid")
        segment_id, observation_id = _record_identity(record)
        # Schema/identity errors still reject the source; identifiable numeric
        # corruption invalidates only this reconstruction until a real snapshot.
        for name in ("bids", "asks"):
            levels = payload.get(name)
            if not isinstance(levels, list) or any(
                not isinstance(level, list) or len(level) != 2 for level in levels
            ):
                raise ValueError(f"binance_replay_snapshot_{name}_invalid")
        last_update_id = strict_non_negative_integer(
            payload.get("lastUpdateId"),
            error="binance_replay_snapshot_id_invalid",
        )
        try:
            bids = _levels(
                payload.get("bids"),
                price_unit=self._price_unit,
                error="binance_replay_snapshot_bids_invalid",
                allow_zero=False,
            )
            asks = _levels(
                payload.get("asks"),
                price_unit=self._price_unit,
                error="binance_replay_snapshot_asks_invalid",
                allow_zero=False,
            )
        except ValueError as error:
            raise ValueError("binance_replay_snapshot_invalid_numeric") from error
        bid_prices, ask_prices = sorted(bids), sorted(asks)
        reason = self._invalid_reason(bid_prices, ask_prices)
        if reason is not None:
            raise ValueError(f"binance_replay_snapshot_{reason}")
        self._bids = bids
        self._asks = asks
        self._bid_prices = bid_prices
        self._ask_prices = ask_prices
        self._known_bid_floor_tick = bid_prices[0]
        self._known_ask_ceiling_tick = ask_prices[-1]
        self._segment_id = self._source_segment_id = segment_id
        self._observation_id = observation_id
        self._last_update_id = last_update_id
        self._last_times = None
        self._bootstrapped = False
        self._invalid = False

    def advance(
        self,
        record: dict[str, Any],
    ) -> bool | BookBoundary:
        """Apply one update without materializing a book for live validity checks."""
        if self._invalid:
            return False
        if record.get("market_type") != self.market_type:
            raise ValueError("binance_replay_market_type_mismatch")
        segment_id, observation_id = _record_identity(record)
        if (
            segment_id != self._source_segment_id
            or observation_id != self._observation_id
        ):
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
            return False
        if not self._bootstrapped:
            target = (
                self._last_update_id
                if self.market_type == FUTURES_MARKET_TYPE
                else self._last_update_id + 1
            )
            if not first <= target <= last:
                return self._invalidate(record, "sequence_gap")
        elif self.market_type == FUTURES_MARKET_TYPE:
            if payload.get("pu") != self._last_update_id:
                return self._invalidate(record, "sequence_gap")
        elif first > self._last_update_id + 1:
            return self._invalidate(record, "sequence_gap")
        for name in ("b", "a"):
            levels = payload.get(name)
            if not isinstance(levels, list) or any(
                not isinstance(level, list) or len(level) != 2 for level in levels
            ):
                raise ValueError("binance_replay_delta_levels_invalid")
        try:
            bid_updates = _levels(
                payload.get("b"),
                price_unit=self._price_unit,
                error="binance_replay_delta_bids_invalid",
                allow_zero=True,
            )
            ask_updates = _levels(
                payload.get("a"),
                price_unit=self._price_unit,
                error="binance_replay_delta_asks_invalid",
                allow_zero=True,
            )
        except ValueError:
            return self._invalidate(record, "invalid_numeric")
        assert self._known_bid_floor_tick is not None
        assert self._known_ask_ceiling_tick is not None
        known_bid_updates = {
            price_index: quantity
            for price_index, quantity in bid_updates.items()
            if price_index >= self._known_bid_floor_tick
        }
        known_ask_updates = {
            price_index: quantity
            for price_index, quantity in ask_updates.items()
            if price_index <= self._known_ask_ceiling_tick
        }
        _apply_levels(self._bids, self._bid_prices, known_bid_updates)
        _apply_levels(self._asks, self._ask_prices, known_ask_updates)
        reason = self._invalid_reason(self._bid_prices, self._ask_prices)
        if reason is not None:
            return self._invalidate(record, reason)
        event_time, transaction_time, observed_time = _record_times(
            record,
            market_type=self.market_type,
        )
        if self._last_times is not None and observed_time < self._last_times[2]:
            self.stats.non_monotonic_timestamps += 1
            raise ValueError("binance_replay_observed_time_non_monotonic")
        self._last_times = (event_time, transaction_time, observed_time)
        self._last_update_id = last
        if not self._bootstrapped:
            self._bootstrapped = True
            self.stats.valid_segments += 1
            if self.stats.valid_segments > 1:
                self.stats.recovered_segments += 1
        return True

    def feed(
        self,
        record: dict[str, Any],
    ) -> Iterator[BookState | BookBoundary]:
        result = self.advance(record)
        if isinstance(result, BookBoundary):
            yield result
            return
        if not result:
            return
        assert self._last_times is not None
        event_time, transaction_time, observed_time = self._last_times
        segment_id, observation_id = self._segment_id, self._observation_id
        last = self._last_update_id
        assert last is not None
        limit = (
            None
            if self.output_levels is None
            else max(
                0,
                min(
                    self.output_levels,
                    max(len(self._bid_prices), len(self._ask_prices)),
                ),
            )
        )
        state = BookState(
            source_sequence=record["source_sequence"],
            event_time_ms=event_time,
            transaction_time_ms=transaction_time,
            observed_time_ns=observed_time,
            receive_time=record["receive_time"],
            receive_monotonic_ns=record["receive_monotonic_ns"],
            update_id=last,
            segment_id=segment_id,
            observation_id=observation_id,
            market_type=self.market_type,
            price_unit=self.price_unit,
            bids=tuple(
                (price, self._bids[price])
                for price in islice(reversed(self._bid_prices), limit)
            ),
            asks=tuple(
                (price, self._asks[price]) for price in islice(self._ask_prices, limit)
            ),
        )
        if self.connection_close_ns is not None:
            previous = self.stats.coverage_segments.get(segment_id)
            if previous is None:
                self.stats.coverage_segments[segment_id] = (
                    observed_time,
                    self.connection_close_ns,
                )
        self.stats.emitted_states += 1
        yield state

    def _invalid_reason(
        self,
        bids: list[int],
        asks: list[int],
    ) -> str | None:
        if len(bids) < self.required_levels or len(asks) < self.required_levels:
            if not bids or not asks:
                self.stats.one_sided_books += 1
                return "one_sided_book"
            self.stats.unsupported_coverage += 1
            return "unsupported_coverage"
        if bids[-1] >= asks[0]:
            self.stats.crossed_books += 1
            return "crossed_book"
        return None

    def _close_coverage(self, observed_time: int) -> None:
        if self._segment_id in self.stats.coverage_segments:
            start, end = self.stats.coverage_segments[self._segment_id]
            stop = min(end, observed_time)
            if stop <= start:
                del self.stats.coverage_segments[self._segment_id]
            else:
                self.stats.coverage_segments[self._segment_id] = (start, stop)

    def _invalidate(self, record: dict[str, Any], reason: str) -> BookBoundary:
        event_time, transaction_time, observed_time = _record_times(
            record,
            market_type=self.market_type,
        )
        segment_id, observation_id = self._segment_id, self._observation_id
        self._close_coverage(observed_time)
        self._invalid = True
        self._bootstrapped = False
        self._bids.clear()
        self._asks.clear()
        self.stats.invalid_segments += 1
        if reason == "sequence_gap":
            self.stats.gaps += 1
        return BookBoundary(
            source_sequence=record["source_sequence"],
            event_time_ms=event_time,
            transaction_time_ms=transaction_time,
            observed_time_ns=observed_time,
            receive_time=record["receive_time"],
            receive_monotonic_ns=record["receive_monotonic_ns"],
            segment_id=segment_id,
            observation_id=observation_id,
            market_type=self.market_type,
            reason=reason,
        )


def _record_identity(record: dict[str, Any]) -> tuple[str, str]:
    segment_id = record.get("segment_id")
    observation_id = record.get("observation_id")
    if not isinstance(segment_id, str) or not segment_id:
        raise ValueError("binance_replay_segment_identity_invalid")
    if not isinstance(observation_id, str) or not observation_id:
        raise ValueError("binance_replay_observation_identity_invalid")
    return segment_id, observation_id


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
    observed_time = strict_iso_timestamp(
        record.get("receive_time"),
        error="binance_replay_receive_time_invalid",
    ).value
    return (
        event_time,
        transaction_time,
        observed_time,
    )


def _apply_levels(
    target: dict[int, float], prices: list[int], updates: dict[int, float]
) -> None:
    for price_index, quantity in updates.items():
        if quantity == 0:
            if price_index in target:
                del target[price_index]
                prices.pop(bisect_left(prices, price_index))
        else:
            if price_index not in target:
                insort(prices, price_index)
            target[price_index] = quantity


def iter_reconstructed_depth(
    run,
    market_type: str,
    *,
    symbol: str,
    required_levels: int = RECONSTRUCTION_REQUIRED_LEVELS,
    output_levels: int | None = None,
    stats: ReconstructionStats | None = None,
    end_ns: int | None = None,
) -> Iterator[BookState | BookBoundary]:
    """Replay one venue book with one capture-ordered snapshot/delta pass."""

    metadata = run.market_metadata(market_type, symbol)
    active_stats = stats if stats is not None else ReconstructionStats()
    reconstructor: DepthReconstructor | None = None
    active_segment: str | None = None
    reconstruction_segment: str | None = None
    pending_reconnect: tuple[str, str] | None = None
    pending_invalid_reason: str | None = None
    last_state = None

    def close_boundary(state):
        closed = active_stats.coverage_segments[state.segment_id][1]
        seconds, nanos = divmod(closed, 1_000_000_000)
        received = (
            dt.datetime.fromtimestamp(seconds, dt.UTC).strftime("%Y-%m-%dT%H:%M:%S")
            + f".{nanos:09d}Z"
        )
        return BookBoundary(
            # A derived half-open close precedes native events at the same clock.
            source_sequence=0,
            event_time_ms=state.event_time_ms,
            transaction_time_ms=state.transaction_time_ms,
            observed_time_ns=closed,
            receive_time=received,
            receive_monotonic_ns=None,
            segment_id=state.segment_id,
            observation_id=state.observation_id,
            market_type=market_type,
            reason="connection_closed",
        )

    records = run.iter_market_records(
        symbol=symbol,
        market_type=market_type,
        data_kinds=("depth_snapshot", "depth"),
        end_ns=end_ns,
    )
    with closing_iterator(records):
        for record in records:
            if last_state is not None:
                bounds = active_stats.coverage_segments.get(last_state.segment_id)
                observed = strict_iso_timestamp(
                    record["receive_time"], error="binance_replay_receive_time_invalid"
                ).value
                if bounds is not None and bounds[1] <= observed:
                    yield close_boundary(last_state)
                    last_state = None
                    reconstructor = None
                    pending_reconnect = None
            segment_id, observation_id = _record_identity(record)
            if record["data_kind"] == "depth_snapshot":
                if reconstructor is not None:
                    if active_segment == segment_id:
                        if reconstructor._observation_id != observation_id:
                            raise ValueError("binance_replay_segment_identity_mismatch")
                        if reconstructor.valid:
                            continue
                    if reconstructor.valid:
                        pending_reconnect = (
                            reconstructor._segment_id or segment_id,
                            reconstructor._observation_id or "",
                        )
                    reconstructor = None
                bounds = run._coverage_index.bounds(  # noqa: SLF001
                    observation_id,
                    symbol=symbol,
                    market_type=market_type,
                    data_kind="depth",
                )
                if bounds is None:
                    raise ValueError("binance_replay_connection_coverage_invalid")
                if stats is None:
                    active_stats.coverage_segments.clear()
                reconstructor = DepthReconstructor(
                    market_type,
                    price_unit=metadata["price_unit"],
                    snapshot_limit=metadata["snapshot_limit"],
                    required_levels=required_levels,
                    output_levels=output_levels,
                    connection_close_ns=bounds[1],
                    stats=active_stats,
                )
                reconstruction_segment = (
                    f"{segment_id}:snapshot:{record['source_sequence']}"
                    if active_segment == segment_id
                    else segment_id
                )
                active_segment = segment_id
                pending_invalid_reason = None
                try:
                    reconstructor.bootstrap(record)
                    reconstructor._segment_id = reconstruction_segment
                except ValueError as error:
                    reason = str(error)
                    prefix = "binance_replay_snapshot_"
                    if not reason.startswith(prefix):
                        raise
                    reason = reason.removeprefix(prefix)
                    if reason not in {
                        "one_sided_book",
                        "crossed_book",
                        "unsupported_coverage",
                        "invalid_numeric",
                    }:
                        raise
                    reconstructor = None
                    pending_invalid_reason = reason
                    active_stats.invalid_segments += 1
                continue

            if segment_id != active_segment:
                if reconstructor is not None:
                    if reconstructor.valid:
                        event_time, transaction_time, observed_time = _record_times(
                            record,
                            market_type=market_type,
                        )
                        reconstructor._close_coverage(observed_time)
                        active_stats.reconnect_boundaries += 1
                        _yield_value = BookBoundary(
                            source_sequence=record["source_sequence"],
                            event_time_ms=event_time,
                            transaction_time_ms=transaction_time,
                            observed_time_ns=observed_time,
                            receive_time=record["receive_time"],
                            receive_monotonic_ns=record["receive_monotonic_ns"],
                            segment_id=reconstructor._segment_id or segment_id,
                            observation_id=reconstructor._observation_id or "",
                            market_type=market_type,
                            reason="reconnect_boundary",
                        )
                        try:
                            yield _yield_value
                        except GeneratorExit:
                            return
                raise ValueError("binance_replay_snapshot_missing")

            if record["data_kind"] == "depth_bootstrap":
                if reconstructor is not None:
                    reconstructor.advance(record)
                continue

            if pending_reconnect is not None:
                event_time, transaction_time, observed_time = _record_times(
                    record,
                    market_type=market_type,
                )
                active_stats.reconnect_boundaries += 1
                _yield_value = BookBoundary(
                    source_sequence=record["source_sequence"],
                    event_time_ms=event_time,
                    transaction_time_ms=transaction_time,
                    observed_time_ns=observed_time,
                    receive_time=record["receive_time"],
                    receive_monotonic_ns=record["receive_monotonic_ns"],
                    segment_id=pending_reconnect[0],
                    observation_id=pending_reconnect[1],
                    market_type=market_type,
                    reason="reconnect_boundary",
                )
                try:
                    yield _yield_value
                except GeneratorExit:
                    return
                pending_reconnect = None
            if pending_invalid_reason is not None:
                event_time, transaction_time, observed_time = _record_times(
                    record,
                    market_type=market_type,
                )
                _yield_value = BookBoundary(
                    source_sequence=record["source_sequence"],
                    event_time_ms=event_time,
                    transaction_time_ms=transaction_time,
                    observed_time_ns=observed_time,
                    receive_time=record["receive_time"],
                    receive_monotonic_ns=record["receive_monotonic_ns"],
                    segment_id=reconstruction_segment,
                    observation_id=observation_id,
                    market_type=market_type,
                    reason=pending_invalid_reason,
                )
                try:
                    yield _yield_value
                except GeneratorExit:
                    return
                pending_invalid_reason = None
                continue
            if reconstructor is not None:
                for _yield_value in reconstructor.feed(record):
                    last_state = (
                        _yield_value if isinstance(_yield_value, BookState) else None
                    )
                    try:
                        yield _yield_value
                    except GeneratorExit:
                        return

    if last_state is not None:
        bounds = active_stats.coverage_segments.get(last_state.segment_id)
        if bounds is not None and (end_ns is None or bounds[1] <= end_ns):
            yield close_boundary(last_state)


def native_trade_identities():
    """Track exact native trade identity across resends and historical conflicts."""
    index = ExactPrimaryKeyIndex(primary_key=("market", "symbol", "trade_id"))
    latest = {}

    def observe(market_type, symbol, payload):
        namespace = market_type + ":" + symbol
        trade_id = payload["a"]
        key = str(trade_id).encode("ascii")
        previous = latest.get(namespace)
        value = json.dumps(
            {key: value for key, value in payload.items() if key != "E"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if previous is not None and trade_id <= previous:
            stored = index.identity_payload_value(namespace=namespace, payload=key)
            if stored == value:
                return False, previous
            raise ValueError("binance_native_sequence_non_monotonic")
        index.add_identity_payload_rows(rows=((namespace, key, value),))
        latest[namespace] = trade_id
        return True, previous

    return observe


def iter_agg_trades(
    run,
    market_type: str,
    *,
    symbol: str,
    end_ms: int | None = None,
) -> Iterator[TradeEvent | TradeBoundary]:
    """Normalize one recorder trade stream while preserving local identity."""

    run.market_metadata(market_type, symbol)
    previous_key: tuple[int, ...] | None = None
    records = run.iter_market_records(
        symbol=symbol,
        market_type=market_type,
        data_kinds=("agg_trades",),
        end_ns=None if end_ms is None else end_ms * 1_000_000,
    )
    with closing_iterator(records):
        for record in records:
            payload = record["payload"]
            event_time = strict_non_negative_integer(
                payload.get("E"),
                error="binance_replay_event_time_invalid",
            )
            transaction_time = strict_non_negative_integer(
                payload.get("T"),
                error="binance_replay_transaction_time_invalid",
            )
            observed_time = strict_iso_timestamp(
                record["receive_time"],
                error="binance_replay_receive_time_invalid",
            ).value
            schema = run.source_identity["schema_id"]
            validate_market_payload(
                market_type,
                "agg_trades",
                symbol,
                payload,
                source_schema=schema,
                observed_time_ns=observed_time,
                validate_values=False,
            )
            identity = record["observation_id"]
            index = run._coverage_index if isinstance(run, RecorderRun) else None
            validate_price_unit(record["price_unit"])
            try:
                buyer_is_maker = strict_boolean(
                    payload.get("m"), error="binance_replay_trade_side_invalid"
                )
                quantity = strict_positive_float(
                    payload.get("q"), error="binance_replay_trade_quantity_invalid"
                )
                price_index = price_to_index(
                    payload.get("p"),
                    record["price_unit"],
                    error="binance_replay_trade_price_invalid",
                )
                visible_quantity = _visible_trade_quantity(
                    market_type,
                    payload,
                    historical_time=observed_time // 1_000_000
                    if schema == "qrp-tardis-native-archive"
                    else None,
                )
            except ValueError:
                boundary = TradeBoundary(
                    source_sequence=record["source_sequence"],
                    event_time_ms=event_time,
                    transaction_time_ms=transaction_time,
                    observed_time_ns=observed_time,
                    receive_time=record["receive_time"],
                    receive_monotonic_ns=record["receive_monotonic_ns"],
                    observation_id=index.trade_observation_id(identity, observed_time)
                    if index is not None
                    else identity,
                    market_type=market_type,
                    reason="invalid_trade_values",
                )
                key = observation_order_key(boundary)
                if previous_key is not None and key < previous_key:
                    raise ValueError(
                        "binance_replay_trade_order_non_monotonic"
                    ) from None
                previous_key = key
                try:
                    try:
                        yield boundary
                    finally:
                        # A prefetched future boundary must not affect earlier
                        # coverage. Closing after consuming it still records it.
                        if index is not None:
                            index.invalidate_trade(identity, observed_time)
                except GeneratorExit:
                    return
                continue
            trade = TradeEvent(
                source_sequence=record["source_sequence"],
                event_time_ms=event_time,
                transaction_time_ms=transaction_time,
                observed_time_ns=observed_time,
                receive_time=record["receive_time"],
                receive_monotonic_ns=record["receive_monotonic_ns"],
                observation_id=index.trade_observation_id(identity, observed_time)
                if index is not None
                else identity,
                market_type=market_type,
                price_unit=record["price_unit"],
                price_index=price_index,
                quantity=quantity,
                visible_quantity=visible_quantity,
                buyer_is_maker=buyer_is_maker,
            )

            key = observation_order_key(trade)
            if previous_key is not None and key < previous_key:
                raise ValueError("binance_replay_trade_order_non_monotonic")
            previous_key = key
            try:
                yield trade
            except GeneratorExit:
                return
