"""Internal provider raw-file parsing and export-inspection helpers."""

from __future__ import annotations

import codecs
import io
import os
import re
import struct
import sys
import tempfile
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from quant_research_platform.runtime.resources import (
    MIB,
    MemoryLease,
    RuntimeResourcePolicy,
    _close_lease_preserving_primary,
    current_runtime_resource_policy,
    require_free_disk_headroom,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.csmar import field_mappings as _field_mappings
from quant_research_platform.storage.errors import DatasetReadCapacityExceeded
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    dataset_file_identity,
    dataset_file_identity_from_stat,
    require_unaliased_path,
)

MAX_LOGICAL_RECORD_BYTES = MIB
# Provider financial exports include optional columns beyond the normalized owner
# mapping; keep a bounded header limit without rejecting the observed 154-column
# balance export.
_MAX_PROVIDER_HEADER_FIELDS = 160
LOGICAL_RECORD_LIMIT_ERROR = "csmar_logical_record_exceeds_source_limit"
_CSV_RECORD_LEASE_BOUND_BYTES = 5 * MAX_LOGICAL_RECORD_BYTES
_CSV_PARSER_WORKSPACE_BOUND_BYTES = 128 * 1024


@dataclass(frozen=True)
class RawFileMetadata:
    """Metadata parsed from a raw CSMAR export file name."""

    dataset: str
    market_scope: str
    start_date: str
    end_date: str
    export_date: str
    extension: str
    part: str | None = None


@dataclass(frozen=True)
class RawExportInspection:
    """Header-level inspection result for a raw CSMAR export file."""

    path: Path
    dataset: str
    table_code: str
    columns: tuple[str, ...]
    missing_required_columns: list[str]
    has_metadata: bool


FILENAME_RE = re.compile(
    r"^csmar__(?P<dataset>[a-z0-9_]+)__"
    r"(?P<market_scope>[a-z0-9_]+)__"
    r"(?:(?P<start>\d{8})_(?P<end>\d{8})|(?P<full>full))__"
    r"(?:(?:part-(?P<part>\d{2})__)?)"
    r"export-(?P<export>\d{8})\.(?P<extension>csv)$"
)


def parse_raw_filename(path: str | Path) -> RawFileMetadata:
    """Parse the recommended raw CSMAR file name format."""

    name = Path(path).name
    match = FILENAME_RE.match(name)
    if match is None:
        raise ValueError(f"Invalid CSMAR raw filename: {name}")

    dataset = match.group("dataset")
    if dataset not in _field_mappings.RAW_TABLE_CONTRACTS:
        raise ValueError(f"Unknown CSMAR dataset in filename: {dataset}")

    return RawFileMetadata(
        dataset=dataset,
        market_scope=match.group("market_scope"),
        start_date="full" if match.group("full") else match.group("start"),
        end_date="full" if match.group("full") else match.group("end"),
        export_date=match.group("export"),
        extension=match.group("extension"),
        part=match.group("part"),
    )


def infer_dataset_from_native_export(path: str | Path) -> str:
    """Infer the dataset from a native CSMAR table file name or parent path."""

    export_path = Path(path)
    stem = re.sub(r"\d+$", "", export_path.stem)
    for dataset, contract in _field_mappings.RAW_TABLE_CONTRACTS.items():
        if contract.table_code == stem:
            return dataset

    for part in reversed(export_path.parts):
        if part in _field_mappings.RAW_TABLE_CONTRACTS:
            return part

    raise ValueError(f"Cannot infer CSMAR dataset from native filename: {export_path}")


def _infer_dataset_from_header(columns: tuple[str, ...]) -> str:
    matches = [
        dataset
        for dataset in _field_mappings.RAW_TABLE_CONTRACTS
        if not _field_mappings.missing_raw_column_names(columns, dataset)
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot infer CSMAR dataset from raw header: {matches}")
    return matches[0]


def require_raw_resource_policy(
    resource_policy: RuntimeResourcePolicy | None = None,
) -> RuntimeResourcePolicy:
    policy = resource_policy or current_runtime_resource_policy()
    if policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    if type(policy) is not RuntimeResourcePolicy:
        raise TypeError("invalid_runtime_resource_policy")
    return policy


class BoundedCsvRecord:
    """One validated CSV record backed by the owner's reusable private spool."""

    def __init__(
        self,
        owner: _BoundedCsvRecords,
        *,
        generation: int,
        field_count: int,
        source_bytes: int,
        canonical_bytes: int,
        decoded_fields: list[str] | None = None,
        largest_field_bytes: int = 0,
    ) -> None:
        self._owner = owner
        self._generation = generation
        self.field_count = field_count
        self.source_bytes = source_bytes
        self.canonical_bytes = canonical_bytes
        self._decoded_fields = decoded_fields
        self._largest_field_bytes = largest_field_bytes

    def iter_fields(self) -> Iterator[str]:
        self._require_live()
        if self._decoded_fields is not None:
            yield from self._decoded_fields
            return
        spool = self._owner._spool
        if spool is None:
            raise RuntimeError("csmar_csv_record_expired")
        spool.seek(0)
        for _ in range(self.field_count):
            encoded_length = spool.read(4)
            if len(encoded_length) != 4:
                raise ValueError("malformed_csmar_csv_row")
            length = struct.unpack(">I", encoded_length)[0]
            self._require_materialization_capacity(
                current_encoded_field_bytes=length,
                total_decoded_output_bytes=length,
                decoded_list_slot_bytes=0,
            )
            payload = spool.read(length)
            if len(payload) != length:
                raise ValueError("malformed_csmar_csv_row")
            try:
                yield payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("malformed_csmar_csv_row") from exc

    def materialize(
        self,
        *,
        max_fields: int | None = None,
        field_count_error: str = "malformed_csmar_csv_row",
    ) -> list[str]:
        if max_fields is not None and self.field_count > max_fields:
            raise ValueError(field_count_error)
        if self._decoded_fields is not None:
            original_decoded_list_slot_bytes = self.field_count * struct.calcsize("P")
            copied_decoded_list_slot_bytes = self.field_count * struct.calcsize("P")
            self._require_materialization_capacity(
                current_encoded_field_bytes=0,
                total_decoded_output_bytes=self.canonical_bytes,
                decoded_list_slot_bytes=(
                    original_decoded_list_slot_bytes + copied_decoded_list_slot_bytes
                ),
            )
            return self._decoded_fields.copy()
        self._require_materialization_capacity(
            current_encoded_field_bytes=self._largest_field_bytes,
            total_decoded_output_bytes=self.canonical_bytes,
            decoded_list_slot_bytes=self.field_count * struct.calcsize("P"),
        )
        return list(self.iter_fields())

    def _require_materialization_capacity(
        self,
        *,
        current_encoded_field_bytes: int,
        total_decoded_output_bytes: int,
        decoded_list_slot_bytes: int,
    ) -> None:
        owner = self._owner
        materialization_control_span_bytes = (
            owner._source_buffer_span_bytes
            + len(owner._buffer)
            + owner._spool_buffer_span_bytes
        )
        parser_lease = owner._parser_lease
        lease = owner._record_lease
        if (
            parser_lease is None
            or not parser_lease.active
            or materialization_control_span_bytes > parser_lease.reserved_bytes
            or lease is None
            or not lease.active
            or current_encoded_field_bytes
            + total_decoded_output_bytes
            + decoded_list_slot_bytes
            > lease.reserved_bytes
        ):
            raise DatasetReadCapacityExceeded(
                "csmar_raw_working_set_exceeds_task_memory_budget"
            )

    def _require_live(self) -> None:
        lease = self._owner._record_lease
        if (
            self._owner.generation != self._generation
            or lease is None
            or not lease.active
        ):
            raise RuntimeError("csmar_csv_record_expired")


class _BoundedCsvRecords:
    """Parse the supported CSV grammar without whole decoded-line ownership."""

    _START_FIELD = 0
    _IN_UNQUOTED = 1
    _IN_QUOTED = 2
    _AFTER_QUOTE = 3

    def __init__(
        self,
        path: Path,
        *,
        limit_bytes: int,
        resource_policy: RuntimeResourcePolicy,
        payload_lease_ref: list[MemoryLease | None] | None = None,
        expected_identity: DatasetFileIdentity | None = None,
    ) -> None:
        try:
            path_identity = dataset_file_identity(path)
        except ValueError as exc:
            raise ValueError("csmar_raw_export_identity_changed") from exc
        if expected_identity is not None and path_identity != expected_identity:
            raise ValueError("csmar_raw_export_identity_changed")
        handle = None
        try:
            handle = path.open("rb")
            identity = dataset_file_identity_from_stat(os.fstat(handle.fileno()))
            if identity != path_identity:
                raise ValueError("csmar_raw_export_identity_changed")
            self._handle = handle
            self._identity = identity
            self._spool: object | None = None
            self._path = path
            self._limit_bytes = limit_bytes
            self._resource_policy = resource_policy
            self._payload_lease_ref = payload_lease_ref
            self._record_lease: MemoryLease | None = None
            self._parser_lease: MemoryLease | None = None
            self._owns_record_lease = payload_lease_ref is None
            read_buffer_span_bytes = io.DEFAULT_BUFFER_SIZE
            validation_buffer_span_bytes = io.DEFAULT_BUFFER_SIZE
            field_buffer_span_bytes = io.DEFAULT_BUFFER_SIZE
            self._source_buffer_span_bytes = io.DEFAULT_BUFFER_SIZE
            self._owner_read_result_span_bytes = io.DEFAULT_BUFFER_SIZE
            self._spool_buffer_span_bytes = io.DEFAULT_BUFFER_SIZE
            self._validation_buffer_span_bytes = io.DEFAULT_BUFFER_SIZE
            self._validation_input_copy_span_bytes = io.DEFAULT_BUFFER_SIZE
            self._validation_decoded_output_span_bytes = io.DEFAULT_BUFFER_SIZE
            self._field_buffer_span_bytes = io.DEFAULT_BUFFER_SIZE
            self._slow_control_span_bytes = (
                self._source_buffer_span_bytes
                + self._owner_read_result_span_bytes
                + self._spool_buffer_span_bytes
                + self._validation_buffer_span_bytes
                + self._validation_input_copy_span_bytes
                + self._validation_decoded_output_span_bytes
                + self._field_buffer_span_bytes
            )
            self._fast_control_span_bytes = (
                read_buffer_span_bytes
                + validation_buffer_span_bytes
                + field_buffer_span_bytes
            )
            self._buffer = b""
            self._buffer_position = 0
            self._pushback: deque[int] = deque()
            self._first_record = True
            self.generation = 0
        except BaseException as primary:
            if handle is not None:
                try:
                    handle.close()
                except BaseException as close_error:
                    primary.add_note(
                        "csmar raw handle close also failed: "
                        f"{type(close_error).__name__}"
                    )
            if isinstance(primary, (FileNotFoundError, NotADirectoryError)):
                raise ValueError("csmar_raw_export_identity_changed") from primary
            raise

    def __iter__(self) -> _BoundedCsvRecords:
        return self

    def __next__(self) -> BoundedCsvRecord:
        self._start_record()
        self.generation += 1
        try:
            fast_fields = self._try_fast_fields()
            if fast_fields is not None:
                fields, source_bytes, canonical_bytes = fast_fields
                return BoundedCsvRecord(
                    self,
                    generation=self.generation,
                    field_count=len(fields),
                    source_bytes=source_bytes,
                    canonical_bytes=canonical_bytes,
                    decoded_fields=fields,
                )
            self._ensure_spool()
            self._spool.seek(0)
            self._spool.truncate(0)
            return self._parse_record()
        except UnicodeDecodeError as exc:
            raise ValueError("malformed_csmar_csv_row") from exc

    def _start_record(self) -> None:
        if self._parser_lease is not None:
            _close_lease_preserving_primary(
                self._parser_lease,
                owner="source.csmar.csv.parser",
            )
            self._parser_lease = None
        if self._payload_lease_ref is not None:
            payload_lease = self._payload_lease_ref[0]
            if payload_lease is None or not payload_lease.active:
                raise DatasetReadCapacityExceeded(
                    "csmar_raw_working_set_exceeds_task_memory_budget"
                )
            self._record_lease = payload_lease
            parser_bound = min(
                _CSV_PARSER_WORKSPACE_BOUND_BYTES,
                self._resource_policy.operation_headroom_bytes,
            )
            try:
                self._parser_lease = self._resource_policy.lease(
                    "source.csmar.csv.parser",
                    parser_bound,
                )
            except MemoryError:
                raise DatasetReadCapacityExceeded(
                    "csmar_raw_working_set_exceeds_task_memory_budget"
                ) from None
            return
        if self._record_lease is not None:
            _close_lease_preserving_primary(
                self._record_lease,
                owner="source.csmar.csv.record",
            )
            self._record_lease = None
        available = self._resource_policy.operation_headroom_bytes
        parser_bound = min(_CSV_PARSER_WORKSPACE_BOUND_BYTES, available)
        record_bound = min(
            _CSV_RECORD_LEASE_BOUND_BYTES,
            max(0, available - parser_bound),
        )
        try:
            self._record_lease = self._resource_policy.lease(
                "source.csmar.csv.record",
                record_bound,
            )
            self._parser_lease = self._resource_policy.lease(
                "source.csmar.csv.parser",
                parser_bound,
            )
        except MemoryError:
            if self._parser_lease is not None:
                _close_lease_preserving_primary(
                    self._parser_lease,
                    owner="source.csmar.csv.parser",
                )
                self._parser_lease = None
            if self._record_lease is not None:
                _close_lease_preserving_primary(
                    self._record_lease,
                    owner="source.csmar.csv.record",
                )
                self._record_lease = None
            raise DatasetReadCapacityExceeded(
                "csmar_raw_working_set_exceeds_task_memory_budget"
            ) from None

    def _ensure_spool(self) -> None:
        if self._spool is not None:
            return
        self._require_capacity(self._slow_control_span_bytes)
        scratch_parent = Path(tempfile.gettempdir())
        scratch_parent = require_unaliased_path(
            Path(scratch_parent), error="csmar_raw_scratch_parent_invalid"
        )
        require_free_disk_headroom(
            scratch_parent,
            reserve_bytes=self._limit_bytes + 4 * (self._limit_bytes + 1),
        )
        require_unaliased_path(scratch_parent, error="csmar_raw_scratch_parent_invalid")
        self._spool = tempfile.TemporaryFile(
            mode="w+b",
            dir=scratch_parent,
        )

    def _try_fast_fields(self) -> tuple[list[str], int, int] | None:
        """Parse common unquoted legal-width rows without per-byte dispatch."""

        if (
            self._pushback
            or self._buffer_position != len(self._buffer)
            or not isinstance(self._handle, io.BufferedReader)
        ):
            return None
        lease = self._record_lease
        available = 0 if lease is None else lease.reserved_bytes - self._limit_bytes
        if available <= self._limit_bytes:
            return None
        line = self._handle.readline(self._limit_bytes + 1)
        if not line:
            raise StopIteration
        source_bytes = len(line)
        if source_bytes > self._limit_bytes:
            raise ValueError(LOGICAL_RECORD_LIMIT_ERROR)
        comma_count = line.count(b",")
        if b'"' in line or comma_count + 1 > _MAX_PROVIDER_HEADER_FIELDS:
            self._handle.seek(self._handle.tell() - source_bytes)
            return None
        content_end = source_bytes
        legal_cr_position: int | None = None
        if source_bytes >= 2 and line[-2] == 13 and line[-1] == 10:
            content_end -= 2
            legal_cr_position = content_end
        elif source_bytes and line[-1] in (10, 13):
            content_end -= 1
            if line[-1] == 13:
                legal_cr_position = content_end
        first_cr = line.find(b"\r")
        if first_cr != -1:
            if first_cr != legal_cr_position or line.find(b"\r", first_cr + 1) != -1:
                self._handle.seek(self._handle.tell() - source_bytes)
                return None
        content_start = 0
        if self._first_record and content_end >= 3 and line.startswith(b"\xef\xbb\xbf"):
            content_start = 3
        content_bytes = content_end - content_start
        field_count = 0 if content_bytes == 0 else comma_count + 1
        canonical_bytes = content_bytes - max(0, field_count - 1)
        source_line_span_bytes = source_bytes
        content_span_bytes = (
            content_bytes if content_start != 0 or content_end != source_bytes else 0
        )
        decoded_record_source_span_bytes = content_bytes
        decoded_fields_source_span_bytes = canonical_bytes
        decoded_list_slot_span_bytes = field_count * struct.calcsize("P")
        if (
            lease is None
            or source_line_span_bytes
            + content_span_bytes
            + decoded_record_source_span_bytes
            + decoded_fields_source_span_bytes
            + decoded_list_slot_span_bytes
            > lease.reserved_bytes
        ):
            raise DatasetReadCapacityExceeded(
                "csmar_raw_working_set_exceeds_task_memory_budget"
            )
        self._first_record = False
        content = line[content_start:content_end]
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("malformed_csmar_csv_row") from exc
        fields = [] if not decoded else decoded.split(",")
        return fields, source_bytes, canonical_bytes

    def _parse_record(self) -> BoundedCsvRecord:
        state = self._START_FIELD
        record_bytes = 0
        canonical_bytes = 0
        field_count = 0
        field_prefix = 0
        field_bytes = 0
        largest_field_bytes = 0
        record_has_content = False
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        validation = bytearray()
        field_buffer = bytearray()

        def validate(byte: int, *, final: bool = False) -> None:
            validation.append(byte)
            if len(validation) == io.DEFAULT_BUFFER_SIZE or final:
                self._require_capacity(self._slow_control_span_bytes)
                decoder.decode(bytes(validation), final=final)
                validation.clear()

        def consume(byte: int) -> None:
            nonlocal record_bytes
            record_bytes += 1
            if record_bytes > self._limit_bytes:
                raise ValueError(LOGICAL_RECORD_LIMIT_ERROR)
            validate(byte)

        def write_field(byte: int) -> None:
            nonlocal field_bytes, canonical_bytes
            field_buffer.append(byte)
            if len(field_buffer) == io.DEFAULT_BUFFER_SIZE:
                self._require_capacity(self._slow_control_span_bytes)
                self._spool.write(field_buffer)
                field_buffer.clear()
            field_bytes += 1
            canonical_bytes += 1

        def finish_field() -> None:
            nonlocal field_count, field_prefix, field_bytes, largest_field_bytes
            if field_buffer:
                self._require_capacity(self._slow_control_span_bytes)
                self._spool.write(field_buffer)
                field_buffer.clear()
            end = self._spool.tell()
            self._spool.seek(field_prefix)
            self._require_capacity(self._slow_control_span_bytes)
            self._spool.write(struct.pack(">I", field_bytes))
            self._spool.seek(end)
            field_count += 1
            largest_field_bytes = max(largest_field_bytes, field_bytes)
            field_prefix = end
            self._spool.write(b"\0\0\0\0")
            field_bytes = 0

        self._spool.write(b"\0\0\0\0")
        if self._first_record:
            self._first_record = False
            prefix = [self._take_byte() for _ in range(3)]
            present = [byte for byte in prefix if byte is not None]
            if bytes(present) == b"\xef\xbb\xbf":
                for byte in present:
                    consume(byte)
            else:
                self._pushback.extendleft(reversed(present))

        while True:
            byte = self._take_byte()
            if byte is None:
                if not record_has_content:
                    self._spool.seek(0)
                    self._spool.truncate(0)
                    if record_bytes:
                        decoder.decode(bytes(validation), final=True)
                    raise StopIteration
                if state == self._IN_QUOTED:
                    raise ValueError("malformed_csmar_csv_row")
                finish_field()
                decoder.decode(bytes(validation), final=True)
                break

            if state == self._IN_QUOTED:
                consume(byte)
                if byte == 34:
                    state = self._AFTER_QUOTE
                else:
                    write_field(byte)
                continue

            if state == self._AFTER_QUOTE:
                consume(byte)
                if byte == 34:
                    write_field(byte)
                    state = self._IN_QUOTED
                    continue
                if byte == 44:
                    finish_field()
                    state = self._START_FIELD
                    record_has_content = True
                    continue
                if byte in (10, 13):
                    if byte == 13:
                        following = self._take_byte()
                        if following == 10:
                            consume(following)
                        elif following is not None:
                            self._pushback.appendleft(following)
                    finish_field()
                    decoder.decode(bytes(validation), final=True)
                    break
                write_field(byte)
                state = self._IN_UNQUOTED
                continue

            if byte in (10, 13):
                consume(byte)
                if byte == 13:
                    following = self._take_byte()
                    if following == 10:
                        consume(following)
                    elif following is not None:
                        self._pushback.appendleft(following)
                if record_has_content:
                    finish_field()
                else:
                    self._spool.seek(0)
                    self._spool.truncate(0)
                decoder.decode(bytes(validation), final=True)
                break

            consume(byte)
            record_has_content = True
            if state == self._START_FIELD and byte == 34:
                state = self._IN_QUOTED
            elif byte == 44:
                finish_field()
                state = self._START_FIELD
            else:
                write_field(byte)
                state = self._IN_UNQUOTED

        self._spool.flush()
        decoded_field_input_span_bytes = largest_field_bytes
        decoded_field_output_source_span_bytes = largest_field_bytes
        self._require_capacity(
            self._slow_control_span_bytes,
            decoded_field_input_span_bytes,
            decoded_field_output_source_span_bytes,
        )
        return BoundedCsvRecord(
            self,
            generation=self.generation,
            field_count=field_count,
            source_bytes=record_bytes,
            canonical_bytes=canonical_bytes,
            largest_field_bytes=largest_field_bytes,
        )

    def _take_byte(self) -> int | None:
        if self._pushback:
            return self._pushback.popleft()
        if self._buffer_position == len(self._buffer):
            self._require_capacity(self._slow_control_span_bytes)
            lease = self._parser_lease
            available = (
                0
                if lease is None
                else lease.reserved_bytes - self._slow_control_span_bytes
            )
            if available <= 0:
                raise DatasetReadCapacityExceeded(
                    "csmar_raw_working_set_exceeds_task_memory_budget"
                )
            self._buffer = self._handle.read(min(io.DEFAULT_BUFFER_SIZE, available))
            self._buffer_position = 0
            if not self._buffer:
                return None
        byte = self._buffer[self._buffer_position]
        self._buffer_position += 1
        return byte

    def _require_capacity(self, *span_bytes: int) -> None:
        parser_lease = self._parser_lease
        record_lease = self._record_lease
        control_bytes = span_bytes[0] if span_bytes else 0
        payload_bytes = sum(span_bytes[1:])
        if (
            parser_lease is None
            or control_bytes > parser_lease.reserved_bytes
            or payload_bytes > 0
            and (record_lease is None or payload_bytes > record_lease.reserved_bytes)
        ):
            raise DatasetReadCapacityExceeded(
                "csmar_raw_working_set_exceeds_task_memory_budget"
            )

    def close(self) -> None:
        primary = sys.exception()
        failure: BaseException | None = None
        self.generation += 1
        self._buffer = b""
        self._buffer_position = 0
        self._pushback.clear()
        try:
            after = dataset_file_identity_from_stat(os.fstat(self._handle.fileno()))
            if after != self._identity:
                raise ValueError("csmar_raw_export_identity_changed")
        except BaseException as exc:
            if primary is not None:
                primary.add_note(
                    f"csmar raw handle verification also failed: {type(exc).__name__}"
                )
            elif failure is None:
                failure = exc
            else:
                failure.add_note(
                    f"csmar raw handle verification also failed: {type(exc).__name__}"
                )
        try:
            self._handle.close()
        except BaseException as exc:
            if primary is not None:
                primary.add_note(
                    f"csmar raw handle close also failed: {type(exc).__name__}"
                )
            elif failure is None:
                failure = exc
            else:
                failure.add_note(
                    f"csmar raw handle close also failed: {type(exc).__name__}"
                )
        if self._spool is not None:
            spool = self._spool
            self._spool = None
            try:
                spool.close()
            except BaseException as exc:
                if primary is not None:
                    primary.add_note(
                        f"csmar raw spool close also failed: {type(exc).__name__}"
                    )
                elif failure is None:
                    failure = exc
                else:
                    failure.add_note(
                        f"csmar raw spool close also failed: {type(exc).__name__}"
                    )
        parser_lease = self._parser_lease
        if parser_lease is not None:
            try:
                parser_lease.close()
            except BaseException as exc:
                if primary is not None:
                    primary.add_note(
                        "source.csmar.csv.parser lease close also failed: "
                        f"{type(exc).__name__}"
                    )
                elif failure is None:
                    failure = exc
                else:
                    failure.add_note(
                        "source.csmar.csv.parser lease close also failed: "
                        f"{type(exc).__name__}"
                    )
            else:
                self._parser_lease = None
        lease = self._record_lease
        if lease is not None and self._owns_record_lease:
            try:
                lease.close()
            except BaseException as exc:
                if primary is not None:
                    primary.add_note(
                        "source.csmar.csv.record lease close also failed: "
                        f"{type(exc).__name__}"
                    )
                elif failure is None:
                    failure = exc
                else:
                    failure.add_note(
                        "source.csmar.csv.record lease close also failed: "
                        f"{type(exc).__name__}"
                    )
            else:
                self._record_lease = None
        elif lease is not None:
            self._record_lease = None
        if failure is not None:
            raise failure


def iter_export_csv_records(
    path: str | Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    expected_identity: DatasetFileIdentity | None = None,
    payload_lease_ref: list[MemoryLease | None] | None = None,
) -> Iterator[tuple[BoundedCsvRecord, int]]:
    bound_policy = require_raw_resource_policy(resource_policy)

    def activate() -> Iterator[tuple[BoundedCsvRecord, int]]:
        assert bound_policy is not None
        records: _BoundedCsvRecords | None = None
        with runtime_resource_policy_scope(bound_policy):
            try:
                records = _BoundedCsvRecords(
                    Path(path),
                    limit_bytes=MAX_LOGICAL_RECORD_BYTES,
                    resource_policy=bound_policy,
                    payload_lease_ref=payload_lease_ref,
                    expected_identity=expected_identity,
                )
                for record in records:
                    yield record, record.source_bytes
                    del record
            finally:
                primary = sys.exception()
                if records is not None:
                    try:
                        records.close()
                    except BaseException as close_error:
                        if primary is None:
                            raise
                        primary.add_note(
                            "CSMAR CSV record cleanup also failed: "
                            f"{type(close_error).__name__}"
                        )

    return activate()


def read_export_header(
    path: str | Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
    expected_identity: DatasetFileIdentity | None = None,
) -> tuple[str, ...]:
    """Read one CSV header under one managed operation scope."""

    with runtime_resource_policy_scope(resource_policy) as policy:
        return _read_export_header_under_policy(
            path,
            resource_policy=policy,
            expected_identity=expected_identity,
        )


def _read_export_header_under_policy(
    path: str | Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    expected_identity: DatasetFileIdentity | None = None,
) -> tuple[str, ...]:
    """Read the header row from a supported CSV raw export."""

    policy = require_raw_resource_policy(resource_policy)
    export_path = Path(path)
    suffix = export_path.suffix.lower()
    if suffix != ".csv":
        raise ValueError(f"Unsupported CSMAR export extension: {suffix}")
    rows = iter_export_csv_records(
        export_path,
        resource_policy=policy,
        expected_identity=expected_identity,
    )
    try:
        record = next(rows, None)
        columns = (
            tuple(materialize_header_record(record[0])) if record is not None else ()
        )
        if not columns:
            raise ValueError("malformed_csmar_csv_row")
    finally:
        primary = sys.exception()
        try:
            rows.close()
        except BaseException as close_error:
            if primary is None:
                raise
            primary.add_note(
                f"CSMAR CSV row cleanup also failed: {type(close_error).__name__}"
            )
    if len(columns) != len(set(columns)):
        raise ValueError("duplicate_csmar_provider_column")
    return tuple(str(column) for column in columns)


def materialize_header_record(record: BoundedCsvRecord) -> list[str]:
    """Materialize a legal header while preserving duplicate-header errors."""

    if record.field_count > _MAX_PROVIDER_HEADER_FIELDS:
        raise ValueError("csmar_raw_export_schema_invalid")
    columns = record.materialize(
        max_fields=_MAX_PROVIDER_HEADER_FIELDS,
        field_count_error="csmar_raw_export_schema_invalid",
    )
    if len(columns) != len(set(columns)):
        raise ValueError("duplicate_csmar_provider_column")
    return columns


def expected_metadata_sidecar_path(path: str | Path) -> Path:
    """Return the recommended metadata sidecar path for a raw export."""

    export_path = Path(path)
    return export_path.with_suffix(".metadata.json")


def inspect_raw_export_file(
    path: str | Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
    expected_identity: DatasetFileIdentity | None = None,
) -> RawExportInspection:
    """Inspect one raw export under one managed operation scope."""

    with runtime_resource_policy_scope(resource_policy) as policy:
        return _inspect_raw_export_file_under_policy(
            path,
            resource_policy=policy,
            expected_identity=expected_identity,
        )


def _inspect_raw_export_file_under_policy(
    path: str | Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    expected_identity: DatasetFileIdentity | None = None,
) -> RawExportInspection:
    """Inspect the file name, header, and metadata sidecar without full data reads."""

    policy = require_raw_resource_policy(resource_policy)
    export_path = Path(path)
    export_identity = raw_export_file_identity(export_path)
    if expected_identity is not None and export_identity != expected_identity:
        raise ValueError("csmar_raw_export_identity_changed")
    try:
        metadata = parse_raw_filename(export_path)
        dataset = metadata.dataset
    except ValueError:
        if export_path.suffix.lower() != ".csv":
            dataset = infer_dataset_from_native_export(export_path)
            columns = read_export_header(
                export_path,
                resource_policy=policy,
                expected_identity=export_identity,
            )
        else:
            columns = read_export_header(
                export_path,
                resource_policy=policy,
                expected_identity=export_identity,
            )
            try:
                dataset = infer_dataset_from_native_export(export_path)
            except ValueError:
                dataset = _infer_dataset_from_header(columns)
    else:
        columns = read_export_header(
            export_path,
            resource_policy=policy,
            expected_identity=export_identity,
        )
    contract = _field_mappings.RAW_TABLE_CONTRACTS[dataset]

    _field_mappings.validate_raw_column_names(columns, dataset)
    missing = _field_mappings.missing_raw_column_names(columns, dataset)
    metadata_path = require_unaliased_path(
        expected_metadata_sidecar_path(export_path),
        error="csmar_raw_export_identity_changed",
    )
    metadata_identity = (
        raw_export_file_identity(metadata_path) if metadata_path.exists() else None
    )
    return RawExportInspection(
        path=export_path,
        dataset=dataset,
        table_code=contract.table_code,
        columns=columns,
        missing_required_columns=missing,
        has_metadata=metadata_identity is not None,
    )


def raw_export_file_identity(path: Path) -> DatasetFileIdentity:
    try:
        return dataset_file_identity(path)
    except ValueError as exc:
        raise ValueError("csmar_raw_export_identity_changed") from exc
