"""Exact recorder-run replay and deterministic Binance MBP reconstruction."""

from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import os
import sqlite3
import stat
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, fields
from itertools import groupby
from pathlib import Path
from typing import Any

from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
    json_record_working_set_bytes,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeEvent,
    observation_order_key,
    price_to_index,
    validate_price_unit,
)
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.validation import clear_exception_frames
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    RuntimeResourcePolicy,
    _close_lease_preserving_primary,
    require_free_disk_headroom,
    runtime_resource_policy_scope,
)
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
from quant_research_platform.storage._scratch_ownership import (
    ScratchReservation,
    handle_preserving_primary,
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
from quant_research_platform.storage.manifests import read_bounded_qrp_json

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
_LEVEL_BYTES = 256
RECONSTRUCTION_REQUIRED_LEVELS = 10
RECONSTRUCTION_MAX_BOOK_LEVELS = 10_000
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
_CONNECTION_INTERVAL_MAX_RECORD_BYTES = 64 * 1024
_COVERAGE_SQLITE_PAGE_BYTES = 4096
_COVERAGE_SQLITE_STARTUP_BYTES = 1024 * 1024


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
        resource_policy: RuntimeResourcePolicy,
        decode_interval: Callable,
    ) -> None:
        self._decode_interval = decode_interval
        self._root = root
        self._descriptor = descriptor
        self._path = root / descriptor["path"]
        try:
            self.identity = _capture_raw_file_identity(
                self._path,
                missing_is_file_not_found=False,
            )
        except ValueError as exc:
            raise ValueError("binance_replay_connection_intervals_invalid") from exc
        if self.identity.size != descriptor["compressed_bytes"]:
            raise ValueError("binance_replay_connection_intervals_invalid")
        require_free_disk_headroom(
            Path(tempfile.gettempdir()),
            reserve_bytes=2 * _COVERAGE_SQLITE_STARTUP_BYTES,
        )
        self._scratch = None
        self._scratch_reservation = None
        self._growth_credit_bytes = 0
        self._growth_page_limit = 0
        self._database = None
        self._lease = None
        self._policy = resource_policy
        self._coverage_readers = ExitStack()
        try:
            self._lease = resource_policy.lease(
                "binance.replay.coverage_index", 3 * 1024 * 1024
            )
            self._scratch_reservation = ScratchReservation(
                Path(tempfile.gettempdir()), resource_policy=resource_policy
            )
            self._scratch_reservation.resize(2 * _COVERAGE_SQLITE_STARTUP_BYTES)
            self._scratch = tempfile.TemporaryDirectory(prefix="qrp-binance-coverage-")
            self._database = sqlite3.connect(
                Path(self._scratch.name) / "coverage.sqlite3"
            )
            startup_pages = (
                _COVERAGE_SQLITE_STARTUP_BYTES + _COVERAGE_SQLITE_PAGE_BYTES - 1
            ) // _COVERAGE_SQLITE_PAGE_BYTES
            self._database.execute(f"PRAGMA page_size={_COVERAGE_SQLITE_PAGE_BYTES}")
            self._database.execute(f"PRAGMA main.max_page_count={startup_pages}")
            self._database.execute(
                "CREATE TABLE coverage ("
                "observation_id TEXT PRIMARY KEY, market TEXT NOT NULL, "
                "data_kind TEXT NOT NULL, stream TEXT NOT NULL, "
                "segment_id TEXT, opened_ns INTEGER NOT NULL, "
                "closed_ns INTEGER NOT NULL)"
            )
            self._database.execute("PRAGMA cache_size=-1024")
            self._database.execute("PRAGMA temp_store=FILE")
            self._database.execute(
                "CREATE INDEX coverage_window ON coverage "
                "(market, data_kind, opened_ns)"
            )
            self._database.execute(
                "CREATE INDEX coverage_close ON coverage (market, data_kind, closed_ns)"
            )
            self._scratch_reservation.resize(
                int(self._database.execute("PRAGMA main.page_count").fetchone()[0])
                * _COVERAGE_SQLITE_PAGE_BYTES
            )
            self._load(markets=markets, resource_policy=resource_policy)
            self._scratch_reservation.resize(
                int(self._database.execute("PRAGMA main.page_count").fetchone()[0])
                * _COVERAGE_SQLITE_PAGE_BYTES
            )
        except BaseException:
            self.close()
            raise

    def _load(
        self,
        *,
        markets: set[str],
        resource_policy: RuntimeResourcePolicy,
    ) -> None:
        try:
            raw_handle = open_filesystem_file(self._path, "rb")
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError("binance_replay_connection_intervals_invalid") from exc
        digesting_handle = _DigestingReader(raw_handle)
        try:
            gzip_handle = gzip.GzipFile(fileobj=digesting_handle, mode="rb")
        except BaseException as primary:
            try:
                raw_handle.close()
            except BaseException as close_error:
                primary.add_note(
                    "Binance connection coverage raw handle close also failed: "
                    f"{type(close_error).__name__}"
                )
            raise
        count = 0
        try:
            interval_bound = (
                json_record_working_set_bytes(_CONNECTION_INTERVAL_MAX_RECORD_BYTES)
                + 64 * 1024
            )
            while True:
                with resource_policy.lease(
                    "binance.replay.connection_interval",
                    interval_bound,
                ):
                    line: bytes | None = None
                    interval: dict[str, Any] | None = None
                    row = None
                    try:
                        line = gzip_handle.readline(
                            _CONNECTION_INTERVAL_MAX_RECORD_BYTES + 1
                        )
                        if not line:
                            break
                        if len(line) > _CONNECTION_INTERVAL_MAX_RECORD_BYTES:
                            raise ValueError(
                                "binance_replay_connection_intervals_invalid"
                            )
                        try:
                            interval = parse_qrp_json_object(line)
                        except (UnicodeError, ValueError) as exc:
                            raise ValueError(
                                "binance_replay_connection_intervals_invalid"
                            ) from exc
                        row = self._decode_interval(interval, markets=markets)
                        self._admit_row(row)
                        try:
                            self._database.execute(
                                "INSERT INTO coverage VALUES (?, ?, ?, ?, ?, ?, ?)",
                                row,
                            )
                        except sqlite3.IntegrityError as exc:
                            raise ValueError(
                                "binance_replay_connection_intervals_invalid"
                            ) from exc
                        count += 1
                    finally:
                        del interval, line, row
            gzip_handle.close()
            digesting_handle.drain()
            if (
                count != self._descriptor["record_count"]
                or digesting_handle.hexdigest() != self._descriptor["sha256"]
            ):
                raise ValueError("binance_replay_connection_intervals_invalid")
            _verify_raw_file_handle(raw_handle, expected_identity=self.identity)
            self._database.commit()
        finally:
            primary = sys.exception()
            first_failure: BaseException | None = None
            try:
                gzip_handle.close()
            except BaseException as close_error:
                if primary is not None:
                    primary.add_note(
                        "Binance connection coverage gzip close also failed: "
                        f"{type(close_error).__name__}"
                    )
                else:
                    first_failure = close_error
            try:
                raw_handle.close()
            except BaseException as close_error:
                if primary is not None:
                    primary.add_note(
                        "Binance connection coverage close also failed: "
                        f"{type(close_error).__name__}"
                    )
                elif first_failure is None:
                    first_failure = close_error
                else:
                    first_failure.add_note(
                        "Binance connection coverage close also failed: "
                        f"{type(close_error).__name__}"
                    )
            if first_failure is not None:
                raise first_failure

    def _admit_row(self, row: tuple[object, ...]) -> None:
        logical_bytes = sum(
            9 + (len(value.encode("utf-8")) if isinstance(value, str) else 8)
            for value in row
        )
        overflow_payload = _COVERAGE_SQLITE_PAGE_BYTES - 4
        payload_pages = (logical_bytes + overflow_payload - 1) // overflow_payload

        def row_pages(page_limit: int) -> int:
            # Four trees. SQLite balance_nonroot reuses at least one of at most
            # five result pages; balance_deeper adds one root child. A balanced
            # tree cannot be deeper than the admitted page count's bit length.
            return 4 * (payload_pages + 4 * page_limit.bit_length() + 1)

        row_growth = row_pages(self._growth_page_limit) * _COVERAGE_SQLITE_PAGE_BYTES
        if row_growth <= self._growth_credit_bytes:
            self._growth_credit_bytes -= row_growth
            return
        current_pages = int(
            self._database.execute("PRAGMA main.page_count").fetchone()[0]
        )
        minimum_pages = 0
        while (required := row_pages(current_pages + minimum_pages)) > minimum_pages:
            minimum_pages = required
        reserved_bytes = self._scratch_reservation.reserved_bytes
        free_bytes = require_free_disk_headroom(
            Path(self._scratch.name),
            reserve_bytes=max(
                0,
                2 * (current_pages + minimum_pages) * _COVERAGE_SQLITE_PAGE_BYTES
                - reserved_bytes,
            ),
        )
        capacity = free_bytes + reserved_bytes
        memory_capacity = self._scratch_reservation.memory_capacity_bytes
        if memory_capacity is not None:
            capacity = min(capacity, memory_capacity)
        growth_pages = max(
            minimum_pages,
            min(
                max(
                    _COVERAGE_SQLITE_STARTUP_BYTES // _COVERAGE_SQLITE_PAGE_BYTES,
                    8 * minimum_pages,
                ),
                capacity // (2 * _COVERAGE_SQLITE_PAGE_BYTES) - current_pages,
            ),
        )
        if row_pages(current_pages + growth_pages) > growth_pages:
            growth_pages = minimum_pages
        pages = current_pages + growth_pages
        # The rollback journal may retain the original version of every page.
        self._scratch_reservation.resize(
            max(reserved_bytes, 2 * pages * _COVERAGE_SQLITE_PAGE_BYTES)
        )
        self._database.execute(f"PRAGMA main.max_page_count={pages}")
        self._growth_page_limit = pages
        self._growth_credit_bytes = (
            growth_pages - row_pages(pages)
        ) * _COVERAGE_SQLITE_PAGE_BYTES

    def bounds(
        self,
        observation_id: object,
        *,
        symbol: str,
        market_type: str,
        data_kind: str,
    ) -> tuple[int, int] | None:
        row = self._database.execute(
            "SELECT opened_ns, closed_ns FROM coverage "
            "WHERE observation_id = ? AND market = ? AND data_kind = ? "
            "AND stream LIKE ?",
            (observation_id, market_type, data_kind, symbol.lower() + "@%"),
        ).fetchone()
        return None if row is None else (int(row[0]), int(row[1]))

    def segment_id(self, observation_id: object) -> str | None:
        row = self._database.execute(
            "SELECT segment_id FROM coverage WHERE observation_id = ?",
            (observation_id,),
        ).fetchone()
        return None if row is None else row[0]

    def verify_file(
        self,
        root: Path,
        *,
        resource_policy: RuntimeResourcePolicy,
    ) -> None:
        path = root / self._descriptor["path"]
        identity = _capture_raw_file_identity(
            path,
            expected_identity=self.identity,
            missing_is_file_not_found=False,
        )
        if (
            identity.size != self._descriptor["compressed_bytes"]
            or _sha256_file(path, identity, resource_policy)
            != self._descriptor["sha256"]
        ):
            raise ValueError("binance_replay_file_identity_mismatch")

    def close(self) -> None:
        primary = sys.exception()
        failure: BaseException | None = None
        database = self._database
        scratch = self._scratch
        connection_closed = True
        removed = scratch is None
        self._database = None
        self._scratch = None
        self._descriptor = {}
        try:
            self._coverage_readers.close()
        except BaseException as close_error:
            if primary is not None:
                primary.add_note(
                    "Binance coverage window close also failed: "
                    + type(close_error).__name__
                )
            else:
                failure = close_error
        try:
            if database is not None:
                database.close()
        except BaseException as close_error:
            connection_closed = False
            if primary is not None:
                primary.add_note(
                    "Binance connection coverage database close also failed: "
                    f"{type(close_error).__name__}"
                )
            elif failure is None:
                failure = close_error
            else:
                failure.add_note(
                    "Binance coverage database close also failed: "
                    + type(close_error).__name__
                )
        try:
            if scratch is not None:
                scratch.cleanup()
                removed = True
        except BaseException as cleanup_error:
            note = (
                "Binance connection coverage scratch cleanup also failed: "
                f"{type(cleanup_error).__name__}"
            )
            if primary is not None:
                primary.add_note(note)
            elif failure is None:
                failure = cleanup_error
            else:
                failure.add_note(note)
        reservation = self._scratch_reservation
        self._scratch_reservation = None
        if reservation is not None:
            if connection_closed and removed:
                reservation.resize(0)
            try:
                reservation.close()
            except BaseException as close_error:
                note = (
                    "Binance coverage reservation close also failed: "
                    + type(close_error).__name__
                )
                if primary is not None:
                    primary.add_note(note)
                elif failure is None:
                    failure = close_error
                else:
                    failure.add_note(note)
        lease = self._lease
        if lease is not None and connection_closed:
            try:
                _close_lease_preserving_primary(
                    lease, owner="binance.replay.coverage_index"
                )
            except BaseException as close_error:
                if failure is None:
                    failure = close_error
                else:
                    failure.add_note(
                        "coverage lease close also failed: "
                        f"{type(close_error).__name__}"
                    )
        if failure is not None:
            raise failure


class ObservationCoverage:
    """Run-owned exact lookups and a monotone, budgeted interval sweep."""

    def __init__(self, index, market_type, data_kind, symbol):
        self.index = index
        self.parameters = (market_type, data_kind, f"{symbol.lower()}@%")
        self.cursor = None
        self.pending = None
        self.active = []
        self.last_bounds = None
        self.lease = index._policy.lease(
            "binance.replay.coverage_window",
            2 * _CONNECTION_INTERVAL_MAX_RECORD_BYTES + 65536,
        )

    def get(self, observation_id, default=None):
        row = self.index._database.execute(
            "SELECT opened_ns, closed_ns FROM coverage "
            "WHERE market=? AND data_kind=? AND stream LIKE ? AND observation_id=?",
            (*self.parameters, observation_id),
        ).fetchone()
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
        if self.cursor is None:
            self.cursor = self.index._database.execute(
                "SELECT opened_ns, closed_ns, observation_id FROM coverage "
                "WHERE market=? AND data_kind=? AND stream LIKE ? AND closed_ns>? "
                "ORDER BY opened_ns",
                (*self.parameters, end_ns),
            )
            self.pending = self.cursor.fetchone()
        while self.active and self.active[0][0] <= end_ns:
            _, _, size = heapq.heappop(self.active)
            self.lease.shrink_to(self.lease.reserved_bytes - size)
        while self.pending is not None and self.pending[0] <= start_ns:
            _, closed_ns, observation_id = self.pending
            if closed_ns > end_ns:
                size = 512 + 4 * len(observation_id)
                self.lease.grow(size)
                heapq.heappush(self.active, (closed_ns, observation_id, size))
            self.pending = self.cursor.fetchone()
        # Two entries suffice to reject ambiguous coverage; no per-grid history scan.
        return tuple(item[1] for item in self.active[:2])

    def intervals(self):
        """Yield only unambiguous half-open coverage in chronological order."""
        query = (
            "WHERE market=? AND data_kind=? AND stream LIKE ? AND opened_ns<closed_ns"
        )
        active = set()
        previous = None
        retained = 0
        with ExitStack() as readers:
            cursors = [
                readers.enter_context(
                    handle_preserving_primary(
                        self.index._database.execute(
                            f"SELECT {column}, observation_id, {delta} FROM coverage "
                            + query
                            + f" ORDER BY {column}",
                            self.parameters,
                        )
                    )
                )
                for column, delta in (("opened_ns", 1), ("closed_ns", -1))
            ]
            cursor = heapq.merge(*cursors, key=lambda row: (row[0], row[2]))
            try:
                for timestamp, events in groupby(cursor, key=lambda row: row[0]):
                    if (
                        previous is not None
                        and previous < timestamp
                        and len(active) == 1
                    ):
                        yield previous, timestamp
                    for _stamp, identity, delta in events:
                        size = 512 + 4 * len(identity)
                        if delta > 0:
                            self.lease.grow(size)
                            retained += size
                            active.add(identity)
                        else:
                            active.remove(identity)
                            self.lease.shrink_to(self.lease.reserved_bytes - size)
                            retained -= size
                    previous = timestamp
            finally:
                self.lease.shrink_to(self.lease.reserved_bytes - retained)

    def close(self):
        try:
            if self.cursor is not None:
                self.cursor.close()
        finally:
            self.cursor = self.pending = None
            self.active.clear()
            _close_lease_preserving_primary(
                self.lease, owner="binance.replay.coverage_window"
            )


@dataclass
class RecorderRun:
    root: Path
    manifest: dict[str, Any]
    source_identity: dict[str, str]
    resource_policy: RuntimeResourcePolicy
    _files: tuple[_RunFile, ...] = field(repr=False)
    _coverage_index: ObservationCoverageIndex = field(repr=False)

    def funding_roots(self, symbol, *, time_window):
        return (self.root.parent / "funding" / symbol,)

    def hft_source_info(self):
        if self.manifest["market_data_status"] == "FAIL":
            raise ValueError("hft_recorder_run_failed")
        if (
            self.manifest["recording_scope"] != FUTURES_HFT_RECORDING_SCOPE
            or FUTURES_MARKET_TYPE not in self.manifest["market_types"]
        ):
            raise ValueError("hft_futures_run_missing")
        if tuple(self.manifest["symbols"]) != RESEARCH_SYMBOLS:
            raise ValueError("hft_symbols_incomplete")
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

    def _retire(self) -> None:
        self.manifest = {}
        self._files = ()

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
                    _iter_raw_file(
                        self.root,
                        item,
                        manifest=self.manifest,
                        resource_policy=self.resource_policy,
                    )
                ) as records:
                    yield from records

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
                    yield record

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
            observe = (
                resources.enter_context(native_trade_identities(self.resource_policy))
                if "agg_trades" in data_kinds
                else None
            )
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
                yield normalized

    def observation_coverage(
        self,
        *,
        market_type: str,
        data_kind: str,
        symbol: str,
    ) -> ObservationCoverage:
        if data_kind == "depth_snapshot":
            data_kind = "depth"
        return self._coverage_index._coverage_readers.enter_context(
            handle_preserving_primary(
                ObservationCoverage(
                    self._coverage_index, market_type, data_kind, symbol
                )
            )
        )


@contextmanager
def _raw_path_session(root: Path) -> Iterator[Callable[[], None]]:
    raw_root = Path(root)

    def read_root_identity() -> os.stat_result | None:
        require_unaliased_path(raw_root, error=_RAW_IDENTITY_ERROR)
        try:
            metadata = filesystem_stat(raw_root, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except NotADirectoryError as error:
            raise ValueError(_RAW_IDENTITY_ERROR) from error
        _reject_raw_reparse(metadata)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(_RAW_IDENTITY_ERROR)
        return metadata

    root_identity = read_root_identity()

    def verify_root() -> None:
        current = read_root_identity()
        if root_identity is None:
            if current is not None:
                raise ValueError(_RAW_IDENTITY_ERROR)
            return
        if current is None:
            raise ValueError(_RAW_IDENTITY_ERROR)
        unchanged = os.path.samestat(root_identity, current)
        if not unchanged:
            raise ValueError(_RAW_IDENTITY_ERROR)

    try:
        yield verify_root
    finally:
        primary = sys.exception()
        try:
            verify_root()
        except BaseException as verify_error:
            if primary is not None:
                primary.add_note(
                    "Binance raw root verification also failed: "
                    f"{type(verify_error).__name__}"
                )
            else:
                raise


def _capture_raw_file_identity(
    path: Path,
    *,
    expected_identity: DatasetFileIdentity | None = None,
    missing_is_file_not_found: bool = True,
) -> DatasetFileIdentity:
    raw_path = Path(path)
    try:
        filesystem_stat(raw_path, follow_symlinks=False)
    except FileNotFoundError as error:
        if missing_is_file_not_found and expected_identity is None:
            raise FileNotFoundError(raw_path) from error
        raise ValueError(_RAW_IDENTITY_ERROR) from error
    except NotADirectoryError as error:
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
def open_recorder_run(
    root: Path,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[RecorderRun]:
    """Bind one exact recorder run and verify it before and after use."""

    run_root = Path(root)
    with runtime_resource_policy_scope(resource_policy) as policy:
        with _raw_path_session(run_root) as verify_root:
            with _read_run_manifest(run_root, policy) as (
                manifest,
                manifest_identity,
            ):
                coverage_index: ObservationCoverageIndex | None = None
                files: tuple[_RunFile, ...] = ()
                run: RecorderRun | None = None
                try:
                    coverage_index = ObservationCoverageIndex(
                        run_root,
                        manifest["capture"]["connection_intervals"],
                        markets=set(manifest["market_types"]),
                        resource_policy=policy,
                        decode_interval=_decode_recorder_interval,
                    )
                    try:
                        files = _capture_run_files(
                            run_root,
                            manifest,
                            policy,
                            coverage_index=coverage_index,
                        )
                    except BaseException as primary:
                        try:
                            coverage_index.close()
                        except BaseException as cleanup_error:
                            primary.add_note(
                                "Binance connection coverage cleanup also failed: "
                                f"{type(cleanup_error).__name__}"
                            )
                        raise
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
                        _coverage_index=coverage_index,
                    )
                    try:
                        yield run
                    finally:
                        primary = sys.exception()
                        try:
                            verify_root()
                            with _read_run_manifest(run_root, policy) as (
                                current_manifest,
                                current_identity,
                            ):
                                current_files: tuple[_RunFile, ...] = ()
                                try:
                                    if (
                                        current_identity != manifest_identity
                                        or canonical_json_hash(current_manifest)
                                        != run.source_identity["manifest_hash"]
                                    ):
                                        raise ValueError(
                                            "binance_replay_run_identity_changed"
                                        )
                                    current_files = _capture_run_files(
                                        run_root,
                                        current_manifest,
                                        policy,
                                        coverage_index=coverage_index,
                                    )
                                    if tuple(
                                        item.identity for item in current_files
                                    ) != tuple(item.identity for item in files):
                                        raise ValueError(
                                            "binance_replay_file_identity_changed"
                                        )
                                finally:
                                    current_files = ()
                                    current_manifest = None
                        except BaseException as verify_error:
                            if primary is not None:
                                primary.add_note(
                                    "Binance recorder run verification also failed: "
                                    f"{type(verify_error).__name__}"
                                )
                            else:
                                raise
                        finally:
                            cleanup_primary = sys.exception()
                            try:
                                coverage_index.close()
                            except BaseException as cleanup_error:
                                if cleanup_primary is None:
                                    raise
                                cleanup_primary.add_note(
                                    "Binance connection coverage cleanup also failed: "
                                    f"{type(cleanup_error).__name__}"
                                )
                finally:
                    if run is not None:
                        run._retire()
                    files = ()
                    coverage_index = None
                    manifest = None


@contextmanager
def _read_run_manifest(
    root: Path,
    policy: RuntimeResourcePolicy,
) -> Iterator[tuple[dict[str, Any], DatasetFileIdentity]]:
    path = root / "run.json"
    identity = _capture_raw_file_identity(path)
    if identity.size <= 0:
        raise ValueError("binance_replay_run_manifest_invalid")
    bound = json_record_working_set_bytes(identity.size) + 64 * 1024
    with policy.lease("binance.replay.manifest", bound) as lease:
        policy.require_working_set(
            0, error="binance_replay_manifest_memory_budget_exceeded"
        )
        manifest: dict[str, Any] | None = None
        try:
            try:
                manifest = read_bounded_qrp_json(
                    path, expected_identity=identity, max_bytes=identity.size
                )[1]
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
        except BaseException as error:
            clear_exception_frames(error)
            manifest = None
            raise
        assert manifest is not None
        lease.transfer("binance.replay.run_metadata")
        try:
            yield manifest, identity
        finally:
            manifest = None


def _validate_run_manifest(value: object) -> None:
    if not isinstance(value, dict) or value.get("schema_id") != "qrp-binance-recorder":
        raise ValueError("binance_replay_run_manifest_identity_invalid")
    for key in (
        "run_id",
        "recording_scope",
        "venue",
        "status",
        "market_data_status",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError("binance_replay_run_manifest_invalid")
    try:
        validate_run_id(value["run_id"])
    except ValueError as exc:
        raise ValueError("binance_replay_run_manifest_invalid") from exc
    if (
        value["venue"] != "binance"
        or not isinstance(value.get("symbols"), list)
        or value["symbols"] != list(RESEARCH_SYMBOLS)
    ):
        raise ValueError("binance_replay_run_manifest_identity_invalid")
    statuses = {"PASS", "PARTIAL", "FAIL"}
    if (
        value["status"] not in statuses
        or value["market_data_status"] not in statuses
        or value["status"] != value["market_data_status"]
    ):
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
        RESEARCH_SYMBOLS
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
            item, markets=set(market_types), symbols=set(RESEARCH_SYMBOLS)
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
        for symbol in RESEARCH_SYMBOLS
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
        or path.is_absolute()
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
        or path.is_absolute()
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
    try:
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
    except BaseException:
        value = None
        raise


def _decode_recorder_interval(value, *, markets):
    try:
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
    finally:
        value = None


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


def _capture_run_files(
    root: Path,
    manifest: dict[str, Any],
    policy: RuntimeResourcePolicy,
    *,
    coverage_index: ObservationCoverageIndex,
) -> tuple[_RunFile, ...]:
    declared_paths = {item["path"] for item in manifest["files"]} | {
        manifest["capture"]["connection_intervals"]["path"]
    }
    _verify_exact_run_tree(root, declared_paths)
    coverage_index.verify_file(root, resource_policy=policy)
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


def _sha256_file(
    path: Path,
    identity: DatasetFileIdentity,
    policy: RuntimeResourcePolicy,
) -> str:
    with policy.lease("binance.replay.file_hash", _HASH_BUFFER_BYTES + 64 * 1024):
        try:
            handle = open_filesystem_file(path, "rb")
        except (FileNotFoundError, NotADirectoryError) as exc:
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
            primary = sys.exception()
            try:
                handle.close()
            except BaseException as close_error:
                if primary is None:
                    raise
                primary.add_note(
                    "Binance replay file handle close also failed: "
                    f"{type(close_error).__name__}"
                )


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
    root: Path,
    item: _RunFile,
    *,
    manifest: dict[str, Any],
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[dict[str, Any]]:
    metadata = item.metadata
    path = root / metadata["path"]
    bound = json_record_working_set_bytes(metadata["max_record_bytes"]) + 64 * 1024
    try:
        raw_handle = open_filesystem_file(path, "rb")
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ValueError("binance_replay_raw_file_open_failed") from exc
    try:
        _verify_raw_file_handle(raw_handle, expected_identity=item.identity)
        digesting_handle = _DigestingReader(raw_handle)
        gzip_handle = gzip.GzipFile(fileobj=digesting_handle, mode="rb")
        try:
            count = 0
            minimum: int | None = None
            maximum: int | None = None
            previous: int | None = None
            while True:
                with resource_policy.lease("binance.replay.record", bound):
                    line = gzip_handle.readline(metadata["max_record_bytes"] + 1)
                    if not line:
                        line = None
                        break
                    record: dict[str, Any] | None = None
                    try:
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
                        minimum = (
                            capture_seq
                            if minimum is None
                            else min(minimum, capture_seq)
                        )
                        maximum = (
                            capture_seq
                            if maximum is None
                            else max(maximum, capture_seq)
                        )
                        count += 1
                        market_metadata = manifest["market_metadata"][
                            metadata["symbol"]
                        ][metadata["market"]]
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
                    except BaseException as error:
                        clear_exception_frames(error)
                        record = None
                        line = None
                        raise
                    try:
                        yield output
                    finally:
                        output = None
                        record = None
                        line = None
            if (
                count != metadata["record_count"]
                or minimum != metadata["capture_seq_min"]
                or maximum != metadata["capture_seq_max"]
            ):
                raise ValueError("binance_replay_file_manifest_mismatch")
        finally:
            primary = sys.exception()
            try:
                gzip_handle.close()
                digesting_handle.drain()
                if digesting_handle.hexdigest() != metadata["sha256"]:
                    raise ValueError("binance_replay_file_identity_mismatch")
                _verify_raw_file_handle(raw_handle, expected_identity=item.identity)
            except BaseException as verify_error:
                if primary is not None and not isinstance(primary, GeneratorExit):
                    primary.add_note(
                        "Binance consumed-file verification also failed: "
                        f"{type(verify_error).__name__}"
                    )
                else:
                    raise

    finally:
        primary = sys.exception()
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
        )


def _levels(
    value: object,
    *,
    price_unit: str,
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
        price_index = price_to_index(level[0], price_unit, error=error)
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
        max_book_levels: int,
        connection_close_ns: int | None,
        resource_policy: RuntimeResourcePolicy,
        stats: ReconstructionStats,
        coverage_lease: MemoryLease,
        output_levels: int | None = None,
    ) -> None:
        if market_type not in {SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE}:
            raise ValueError("binance_replay_market_type_invalid")
        if type(required_levels) is not int or required_levels <= 0:
            raise ValueError("binance_replay_required_levels_invalid")
        if type(snapshot_limit) is not int or snapshot_limit < required_levels:
            raise ValueError("binance_replay_snapshot_coverage_invalid")
        if type(max_book_levels) is not int or max_book_levels < snapshot_limit:
            raise ValueError("binance_replay_book_bound_invalid")
        if connection_close_ns is not None and (
            type(connection_close_ns) is not int or connection_close_ns < 0
        ):
            raise ValueError("binance_replay_connection_coverage_invalid")
        self.market_type = market_type
        self.price_unit = validate_price_unit(price_unit)
        self.required_levels = required_levels
        self.output_levels = output_levels
        self.max_book_levels = max_book_levels
        self.connection_close_ns = connection_close_ns
        self.policy = resource_policy
        self.stats = stats
        self.coverage_lease = coverage_lease
        self._book_lease = resource_policy.lease(
            "binance.replay.book",
            4 * max_book_levels * _LEVEL_BYTES + 256 * 1024,
        )
        self._bids: dict[int, float] = {}
        self._asks: dict[int, float] = {}
        self._segment_id: str | None = None
        self._observation_id: str | None = None
        self._last_update_id: int | None = None
        self._last_time: int | None = None
        self._known_bid_floor_tick: int | None = None
        self._known_ask_ceiling_tick: int | None = None
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
        segment_id, observation_id = _record_identity(record)
        bids = _levels(
            payload.get("bids"),
            price_unit=self.price_unit,
            error="binance_replay_snapshot_bids_invalid",
            allow_zero=False,
            max_levels=self.max_book_levels,
        )
        asks = _levels(
            payload.get("asks"),
            price_unit=self.price_unit,
            error="binance_replay_snapshot_asks_invalid",
            allow_zero=False,
            max_levels=self.max_book_levels,
        )
        reason = self._invalid_reason(bids, asks)
        if reason is not None:
            raise ValueError(f"binance_replay_snapshot_{reason}")
        self._bids = bids
        self._asks = asks
        self._known_bid_floor_tick = min(bids)
        self._known_ask_ceiling_tick = max(asks)
        self._segment_id = segment_id
        self._observation_id = observation_id
        self._last_update_id = strict_non_negative_integer(
            payload.get("lastUpdateId"),
            error="binance_replay_snapshot_id_invalid",
        )
        self._last_time = None
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
        if segment_id != self._segment_id or observation_id != self._observation_id:
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
        bid_updates = _levels(
            payload.get("b"),
            price_unit=self.price_unit,
            error="binance_replay_delta_bids_invalid",
            allow_zero=True,
            max_levels=self.max_book_levels,
        )
        ask_updates = _levels(
            payload.get("a"),
            price_unit=self.price_unit,
            error="binance_replay_delta_asks_invalid",
            allow_zero=True,
            max_levels=self.max_book_levels,
        )
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
        if (
            _projected_level_count(self._bids, known_bid_updates) > self.max_book_levels
            or _projected_level_count(self._asks, known_ask_updates)
            > self.max_book_levels
        ):
            return self._invalidate(record, "book_bound_exceeded")
        _apply_levels(self._bids, known_bid_updates)
        _apply_levels(self._asks, known_ask_updates)
        reason = self._invalid_reason(self._bids, self._asks)
        if reason is not None:
            return self._invalidate(record, reason)
        event_time, transaction_time, observed_time = _record_times(
            record,
            market_type=self.market_type,
        )
        if self._last_time is not None and observed_time < self._last_time:
            self.stats.non_monotonic_timestamps += 1
            raise ValueError("binance_replay_observed_time_non_monotonic")
        self._last_time = observed_time
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
    ) -> Iterator[OwnedValue[BookState] | BookBoundary]:
        result = self.advance(record)
        if isinstance(result, BookBoundary):
            yield result
            return
        if not result:
            return
        event_time, transaction_time, observed_time = _record_times(
            record, market_type=self.market_type
        )
        segment_id, observation_id = _record_identity(record)
        last = self._last_update_id
        assert last is not None
        levels = (
            len(self._bids) + len(self._asks)
            if self.output_levels is None
            else min(len(self._bids), self.output_levels)
            + min(len(self._asks), self.output_levels)
        )
        state_bound = 64 * 1024 + levels * _LEVEL_BYTES
        state_lease = self.policy.lease("binance.replay.state", state_bound)
        transferred = False
        try:
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
                bids=tuple(sorted(self._bids.items(), reverse=True))
                if self.output_levels is None
                else tuple(
                    (price, self._bids[price])
                    for price in heapq.nlargest(self.output_levels, self._bids)
                ),
                asks=tuple(sorted(self._asks.items()))
                if self.output_levels is None
                else tuple(
                    (price, self._asks[price])
                    for price in heapq.nsmallest(self.output_levels, self._asks)
                ),
            )
            if self.connection_close_ns is not None:
                previous = self.stats.coverage_segments.get(segment_id)
                if previous is None:
                    self.coverage_lease.grow(4 * (512 + 4 * len(segment_id)))
                    self.stats.coverage_segments[segment_id] = (
                        observed_time,
                        self.connection_close_ns,
                    )
            self.stats.emitted_states += 1
            state_lease.transfer("binance.replay.book_state_consumer")
            transferred = True
            yield OwnedValue(state, state_lease)
        except BaseException:
            if not transferred:
                state_lease.close()
            raise

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

    def _close_coverage(self, observed_time: int) -> None:
        if self._segment_id in self.stats.coverage_segments:
            start, end = self.stats.coverage_segments[self._segment_id]
            stop = min(end, observed_time)
            if stop <= start:
                del self.stats.coverage_segments[self._segment_id]
                self.coverage_lease.shrink_to(
                    self.coverage_lease.reserved_bytes
                    - 4 * (512 + 4 * len(self._segment_id))
                )
            else:
                self.stats.coverage_segments[self._segment_id] = (start, stop)

    def _invalidate(self, record: dict[str, Any], reason: str) -> BookBoundary:
        event_time, transaction_time, observed_time = _record_times(
            record,
            market_type=self.market_type,
        )
        segment_id, observation_id = _record_identity(record)
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


def _projected_level_count(
    current: dict[int, float],
    updates: dict[int, float],
) -> int:
    result = len(current)
    for price_index, quantity in updates.items():
        if quantity == 0 and price_index in current:
            result -= 1
        elif quantity > 0 and price_index not in current:
            result += 1
    return result


def _apply_levels(target: dict[int, float], updates: dict[int, float]) -> None:
    for price_index, quantity in updates.items():
        if quantity == 0:
            target.pop(price_index, None)
        else:
            target[price_index] = quantity


def iter_reconstructed_depth(
    run,
    market_type: str,
    *,
    symbol: str,
    required_levels: int = RECONSTRUCTION_REQUIRED_LEVELS,
    max_book_levels: int = RECONSTRUCTION_MAX_BOOK_LEVELS,
    output_levels: int | None = None,
    stats: ReconstructionStats | None = None,
    end_ns: int | None = None,
) -> Iterator[OwnedValue[BookState] | BookBoundary]:
    """Replay one venue book with one capture-ordered snapshot/delta pass."""

    metadata = run.market_metadata(market_type, symbol)
    active_stats = stats if stats is not None else ReconstructionStats()
    reconstructor: DepthReconstructor | None = None
    active_segment: str | None = None
    pending_reconnect: tuple[str, str] | None = None
    pending_invalid_reason: str | None = None
    records = run.iter_market_records(
        symbol=symbol,
        market_type=market_type,
        data_kinds=("depth_snapshot", "depth"),
        end_ns=end_ns,
    )
    with (
        closing_iterator(records),
        run.resource_policy.lease(
            "binance.replay.transient_coverage", 0
        ) as transient_coverage,
    ):
        try:
            for record in records:
                segment_id, observation_id = _record_identity(record)
                if record["data_kind"] == "depth_snapshot":
                    if reconstructor is not None:
                        if active_segment == segment_id:
                            raise ValueError(
                                "binance_replay_snapshot_identity_duplicate"
                            )
                        if reconstructor.valid:
                            pending_reconnect = (
                                active_segment or segment_id,
                                reconstructor._observation_id or "",
                            )
                        reconstructor.close()
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
                        transient_coverage.shrink_to(0)
                    reconstructor = DepthReconstructor(
                        market_type,
                        price_unit=metadata["price_unit"],
                        snapshot_limit=metadata["snapshot_limit"],
                        required_levels=required_levels,
                        max_book_levels=max_book_levels,
                        output_levels=output_levels,
                        connection_close_ns=bounds[1],
                        resource_policy=run.resource_policy,
                        stats=active_stats,
                        coverage_lease=run._coverage_index._lease
                        if stats is not None
                        else transient_coverage,
                    )
                    active_segment = segment_id
                    pending_invalid_reason = None
                    try:
                        with run.resource_policy.lease(
                            "binance.replay.snapshot",
                            json_record_working_set_bytes(1024) + 64 * 1024,
                        ):
                            reconstructor.bootstrap(record)
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
                        }:
                            raise
                        reconstructor.close()
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
                            yield BookBoundary(
                                source_sequence=record["source_sequence"],
                                event_time_ms=event_time,
                                transaction_time_ms=transaction_time,
                                observed_time_ns=observed_time,
                                receive_time=record["receive_time"],
                                receive_monotonic_ns=record["receive_monotonic_ns"],
                                segment_id=active_segment or segment_id,
                                observation_id=reconstructor._observation_id or "",
                                market_type=market_type,
                                reason="reconnect_boundary",
                            )
                        reconstructor.close()
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
                    yield BookBoundary(
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
                    pending_reconnect = None
                if pending_invalid_reason is not None:
                    event_time, transaction_time, observed_time = _record_times(
                        record,
                        market_type=market_type,
                    )
                    yield BookBoundary(
                        source_sequence=record["source_sequence"],
                        event_time_ms=event_time,
                        transaction_time_ms=transaction_time,
                        observed_time_ns=observed_time,
                        receive_time=record["receive_time"],
                        receive_monotonic_ns=record["receive_monotonic_ns"],
                        segment_id=segment_id,
                        observation_id=observation_id,
                        market_type=market_type,
                        reason=pending_invalid_reason,
                    )
                    pending_invalid_reason = None
                    continue
                if reconstructor is not None:
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


@contextmanager
def native_trade_identities(resource_policy):
    """Own exact native trade identity across resends and historical conflicts."""
    index = ExactPrimaryKeyIndex(primary_key=("market", "symbol", "trade_id"))
    with (
        resource_policy.lease(
            "binance.native_trade_identities", index.retained_memory_bytes + 128 * 1024
        ),
        index,
    ):
        latest = {}
        pending = ()
        pending_bytes = 0
        pointer_bytes = sys.getsizeof((None,)) - sys.getsizeof(())

        def observe(market_type, symbol, payload):
            nonlocal pending, pending_bytes
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
                stored = next(
                    (
                        value
                        for ns, identity, value in reversed(pending)
                        if ns == namespace and identity == key
                    ),
                    None,
                )
                if stored is None:
                    stored = index.identity_payload_value(
                        namespace=namespace, payload=key
                    )
                if stored == value:
                    return False, previous
                raise ValueError("binance_native_sequence_non_monotonic")
            row = (namespace, key, value)
            row_bytes = sys.getsizeof(row) + sum(map(sys.getsizeof, row))
            if (
                pending
                and pending_bytes
                + row_bytes
                + 2 * sys.getsizeof(pending)
                + pointer_bytes
                > 64 * 1024
            ):
                index.add_identity_payload_rows(rows=pending)
                pending, pending_bytes = (), 0
            if row_bytes + 2 * sys.getsizeof(()) + pointer_bytes > 64 * 1024:
                index.add_identity_payload_rows(rows=(row,))
            else:
                # ponytail: tuple copying is bounded by 64 KiB; a larger batch would
                # need a mutable container with its own growth admission.
                pending += (row,)
                pending_bytes += row_bytes
            latest[namespace] = trade_id
            return True, previous

        try:
            yield observe
        finally:
            # The index is operation-local; unspilled rows have no recovery reader.
            pending, pending_bytes = (), 0
            latest.clear()


def iter_agg_trades(
    run,
    market_type: str,
    *,
    symbol: str,
    end_ms: int | None = None,
) -> Iterator[TradeEvent]:
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
            quantity = strict_positive_float(
                payload.get("q"),
                error="binance_replay_trade_quantity_invalid",
            )
            schema = run.source_identity["schema_id"]
            validate_market_payload(
                market_type,
                "agg_trades",
                symbol,
                payload,
                source_schema=schema,
                observed_time_ns=observed_time,
            )
            visible_quantity = _visible_trade_quantity(
                market_type,
                payload,
                historical_time=(
                    observed_time // 1_000_000
                    if schema == "qrp-tardis-native-archive"
                    else None
                ),
            )
            trade = TradeEvent(
                source_sequence=record["source_sequence"],
                event_time_ms=event_time,
                transaction_time_ms=transaction_time,
                observed_time_ns=observed_time,
                receive_time=record["receive_time"],
                receive_monotonic_ns=record["receive_monotonic_ns"],
                observation_id=record["observation_id"],
                market_type=market_type,
                price_unit=record["price_unit"],
                price_index=price_to_index(
                    payload.get("p"),
                    record["price_unit"],
                    error="binance_replay_trade_price_invalid",
                ),
                quantity=quantity,
                visible_quantity=visible_quantity,
                buyer_is_maker=strict_boolean(
                    payload.get("m"),
                    error="binance_replay_trade_side_invalid",
                ),
            )

            key = observation_order_key(trade)
            if previous_key is not None and key < previous_key:
                raise ValueError("binance_replay_trade_order_non_monotonic")
            previous_key = key
            yield trade
