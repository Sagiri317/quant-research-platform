"""CSMAR raw export inspection and source-local IO helpers."""

from __future__ import annotations

import codecs
import io
import os
import re
import stat
import struct
import sys
import tempfile
import weakref
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import pandas as pd

from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    bind_runtime_resource_policy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    require_free_disk_headroom,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_key_tracking import ExactPrimaryKeyIndex
from quant_research_platform.storage._dataset_paths import require_unaliased_path
from quant_research_platform.storage._dataset_physical import (
    DatasetFileIdentity,
)
from quant_research_platform.storage._dataset_reading import (
    closing_dataset_iterator,
    iter_bounded_frame_chunks,
)
from quant_research_platform.storage.errors import DatasetReadCapacityExceeded

from .raw_schemas import (
    LOGICAL_RECORD_LIMIT_ERROR,
    RAW_TABLE_CONTRACTS,
    BoundedCsvRecord,
    RawExportInspection,
    inspect_raw_export_file,
    iter_export_csv_records,
    logical_record_limit_bytes,
    materialize_header_record,
    raw_export_file_identity,
    require_raw_resource_policy,
)

SUPPORTED_RAW_EXPORT_SUFFIXES = {".csv"}
STRING_CODE_DTYPES = {"Stkcd": str, "Symbol": str, "Indexcd": str}
INDEX_INFO_CODE_RE = re.compile(r"^[0-9A-Za-z_.-]{1,32}$")
INDEX_INFO_SOURCE_CODE_ENUMS = (
    frozenset({"1", "2", "3", "4"}),
    frozenset({"1", "2", "3", "4"}),
    frozenset({"1", "2", "3", "4", "5", "6", "7"}),
    frozenset({"1", "2", "3", "4", "5"}),
    frozenset({"1", "2", "3", "4", "5", "6", "7", "8", "9"}),
)
INDEX_INFO_SOURCE_CODE_COLUMNS = (
    "Idxinfo05",
    "Idxinfo06",
    "Idxinfo07",
    "Idxinfo08",
    "Idxinfo09",
)


def _utf8_size(value: str) -> int:
    size = 0
    index = 0
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            raise UnicodeEncodeError(
                "utf-8",
                value,
                index,
                index + 1,
                "surrogates not allowed",
            )
        size += 1 + (codepoint > 0x7F) + (codepoint > 0x7FF) + (codepoint > 0xFFFF)
        index += 1
    return size


@dataclass(frozen=True)
class RawExportIntakeResult:
    """Record one raw export inspection result."""

    path: Path
    inspection: RawExportInspection | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Return whether inspection succeeded."""

        return (
            self.error is None
            and self.inspection is not None
            and not self.inspection.missing_required_columns
        )


class _IndexInfoContinuationSpool:
    """Bounded private scratch for malformed physical-row continuations."""

    def __init__(self, *, record_limit_bytes: int, scratch_parent: Path) -> None:
        scratch_parent = require_unaliased_path(
            Path(scratch_parent), error="csmar_raw_scratch_parent_invalid"
        )
        require_free_disk_headroom(
            scratch_parent,
            reserve_bytes=record_limit_bytes + 4 * (record_limit_bytes + 1),
        )
        require_unaliased_path(scratch_parent, error="csmar_raw_scratch_parent_invalid")
        self._handle = tempfile.TemporaryFile(
            mode="w+b",
            buffering=0,
            dir=scratch_parent,
        )
        self._count = 0

    def __enter__(self) -> _IndexInfoContinuationSpool:
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> bool:  # noqa: ANN001
        try:
            self._handle.close()
        except BaseException:
            if exc_type is None:
                raise
        return False

    def clear(self) -> None:
        self._handle.seek(0)
        self._handle.truncate(0)
        self._count = 0

    def append_fields(
        self,
        fields: Iterable[str],
        *,
        field_count: int,
        suffix_count: int = 0,
        include_empty: bool = True,
        retained_payload_bytes: int,
        source_bytes: int,
        resource_policy: RuntimeResourcePolicy,
    ) -> list[str]:
        """Stream one continuation prefix and retain only its bounded suffix."""

        prefix_count = field_count - suffix_count
        if prefix_count < 0:
            raise ValueError("malformed_csmar_csv_row")
        _require_raw_conversion_headroom(
            logical_bytes=source_bytes + 4,
            retained_previous_bytes=retained_payload_bytes,
            resource_policy=resource_policy,
        )
        self._handle.seek(0, 2)
        frame_start = self._handle.tell()
        self._handle.write(b"\0\0\0\0")
        payload_bytes = 0
        suffix: list[str] = []
        for position, value in enumerate(fields):
            if position >= prefix_count:
                suffix.append(value)
                continue
            if position:
                self._handle.write(b",")
                payload_bytes += 1
            payload_bytes += self._write_text(
                value,
                retained_payload_bytes=retained_payload_bytes,
                source_bytes=source_bytes,
                resource_policy=resource_policy,
            )
        if payload_bytes == 0 and not include_empty:
            self._handle.seek(frame_start)
            self._handle.truncate()
            return suffix
        frame_end = self._handle.tell()
        self._handle.seek(frame_start)
        self._handle.write(struct.pack(">I", payload_bytes))
        self._handle.seek(frame_end)
        self._count += 1
        return suffix

    def _write_text(
        self,
        value: str,
        *,
        retained_payload_bytes: int,
        source_bytes: int,
        resource_policy: RuntimeResourcePolicy,
    ) -> int:
        payload_bytes = 0
        codepoint_span = max(1, io.DEFAULT_BUFFER_SIZE // 4)
        for start in range(0, len(value), codepoint_span):
            stop = min(len(value), start + codepoint_span)
            transient_bytes = (stop - start) * 8
            _require_raw_conversion_headroom(
                logical_bytes=source_bytes + transient_bytes,
                retained_previous_bytes=retained_payload_bytes,
                resource_policy=resource_policy,
            )
            text_slice = value[start:stop]
            payload = text_slice.encode("utf-8")
            self._handle.write(payload)
            payload_bytes += len(payload)
            del text_slice, payload
        return payload_bytes

    def __bool__(self) -> bool:
        return self._count > 0

    def joined(
        self,
        current: str,
        *,
        retained_payload_bytes: int,
        source_bytes: int,
        resource_policy: RuntimeResourcePolicy,
    ) -> str:
        result_bytes = self._joined_payload_bytes(current)
        _require_index_info_reconstruction_headroom(
            retained_payload_bytes=retained_payload_bytes,
            source_bytes=source_bytes,
            result_bytes=result_bytes,
            resource_policy=resource_policy,
        )
        output = io.StringIO()
        wrote = bool(current)
        if current:
            output.write(current)
        self._handle.seek(0)
        for _ in range(self._count):
            encoded_length = self._handle.read(4)
            if len(encoded_length) != 4:
                raise ValueError("malformed_csmar_csv_row")
            length = struct.unpack(">I", encoded_length)[0]
            if not wrote and length == 0:
                continue
            if wrote:
                output.write("\n")
            decoder = codecs.getincrementaldecoder("utf-8")("strict")
            remaining = length
            while remaining:
                payload = self._handle.read(min(io.DEFAULT_BUFFER_SIZE, remaining))
                if not payload:
                    raise ValueError("malformed_csmar_csv_row")
                remaining -= len(payload)
                output.write(decoder.decode(payload, final=remaining == 0))
            wrote = True
        return output.getvalue()

    def _joined_payload_bytes(self, current: str) -> int:
        current_bytes = _utf8_size(current)
        result_bytes = current_bytes
        wrote = bool(current)
        self._handle.seek(0)
        for _ in range(self._count):
            encoded_length = self._handle.read(4)
            if len(encoded_length) != 4:
                raise ValueError("malformed_csmar_csv_row")
            length = struct.unpack(">I", encoded_length)[0]
            if not wrote and length == 0:
                continue
            if wrote:
                result_bytes += 1
            result_bytes += length
            wrote = True
            self._handle.seek(length, 1)
        return result_bytes


@contextmanager
def _raw_export_path_session(
    root: Path,
) -> Iterator[tuple[Iterator[Path], Callable[[], None]]]:
    root = Path(root)

    def read_root_identity() -> os.stat_result:
        try:
            metadata = root.lstat()
        except OSError as error:
            raise ValueError("csmar_raw_export_identity_changed") from error
        reparse_flag = getattr(metadata, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
        )
        if not stat.S_ISDIR(metadata.st_mode) or reparse_flag:
            raise ValueError("csmar_raw_export_identity_changed")
        return metadata

    root_identity = read_root_identity()

    def verify_root() -> None:
        current = read_root_identity()
        try:
            unchanged = os.path.samestat(root_identity, current)
        except OSError as error:
            raise ValueError("csmar_raw_export_identity_changed") from error
        if not unchanged:
            raise ValueError("csmar_raw_export_identity_changed")

    def fail_walk(error: OSError) -> None:
        raise ValueError("csmar_raw_export_identity_changed") from error

    def iter_paths() -> Iterator[Path]:
        walker = iter(
            root.walk(
                follow_symlinks=False,
                on_error=fail_walk,
            )
        )
        while True:
            verify_root()
            try:
                parent, dirnames, filenames = next(walker)
            except StopIteration:
                verify_root()
                return
            verify_root()
            require_unaliased_path(parent, error="csmar_raw_export_identity_changed")
            verify_root()
            for name in dirnames:
                require_unaliased_path(
                    parent / name,
                    error="csmar_raw_export_identity_changed",
                )
                verify_root()
            dirnames.sort()
            for name in sorted(filenames):
                path = parent / name
                if path.suffix.lower() not in SUPPORTED_RAW_EXPORT_SUFFIXES:
                    continue
                require_unaliased_path(path, error="csmar_raw_export_identity_changed")
                verify_root()
                if (
                    "[DES]" in path.name
                    or "__description__" in path.name
                    or ".metadata." in path.name
                ):
                    continue
                verify_root()
                yield path
                verify_root()

    try:
        require_unaliased_path(root, error="csmar_raw_export_identity_changed")
        verify_root()
        yield iter_paths(), verify_root
    finally:
        if sys.exc_info()[0] is None:
            verify_root()
        else:
            try:
                verify_root()
            except BaseException:
                pass


def iter_raw_export_files(
    root: Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[Path]:
    """Yield supported raw exports in deterministic spill-backed order."""

    bound_policy = bind_runtime_resource_policy(resource_policy)
    raw_root = Path(root)

    def iter_files() -> Iterator[Path]:
        policy = bound_policy or active_runtime_resource_policy()
        with runtime_resource_policy_scope(policy):
            with _raw_export_path_session(raw_root) as (paths, verify_root):
                with ExactPrimaryKeyIndex(
                    primary_key=("path",),
                    timestamp_columns=(),
                    scratch_parent=raw_root.parent,
                ) as index:
                    files = _iter_indexed_raw_export_files(
                        raw_root,
                        paths=paths,
                        verify_root=verify_root,
                        index=index,
                        namespace="csmar-raw-export-paths",
                    )
                    for path, _identity in files:
                        verify_root()
                        yield path
                        verify_root()

    return iter_files()


def inspect_raw_export(
    path: Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> RawExportIntakeResult:
    """Inspect one raw export without writing metadata sidecars."""

    policy = require_raw_resource_policy(resource_policy)
    return _inspect_raw_export_with_identity(path, resource_policy=policy)


def _inspect_raw_export_with_identity(
    path: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    expected_identity: DatasetFileIdentity | None = None,
) -> RawExportIntakeResult:
    try:
        with runtime_resource_policy_scope(resource_policy):
            inspection = inspect_raw_export_file(
                path,
                resource_policy=resource_policy,
                expected_identity=expected_identity,
            )
    except Exception as exc:  # noqa: BLE001
        return RawExportIntakeResult(path=path, error=str(exc))

    return RawExportIntakeResult(path=path, inspection=inspection)


def iter_index_info_rows(
    export_path: Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
    expected_identity: DatasetFileIdentity | None = None,
    retained_payload_bytes: list[int] | None = None,
) -> Iterable[tuple[list[str], list[str], int]]:
    bound_policy = bind_runtime_resource_policy(resource_policy)

    def activate() -> Iterator[tuple[list[str], list[str], int]]:
        policy = bound_policy or active_runtime_resource_policy()
        with runtime_resource_policy_scope(policy):
            yield from _iter_index_info_rows_under_policy(
                export_path,
                resource_policy=policy,
                expected_identity=expected_identity,
                retained_payload_bytes=retained_payload_bytes,
            )

    return activate()


def _iter_index_info_rows_under_policy(
    export_path: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    expected_identity: DatasetFileIdentity | None,
    retained_payload_bytes: list[int] | None,
) -> Iterator[tuple[list[str], list[str], int]]:
    policy = resource_policy
    record_limit_bytes = logical_record_limit_bytes(policy)
    downstream_retained_bytes = (
        retained_payload_bytes if retained_payload_bytes is not None else [0]
    )
    reader_retained_bytes = [0]
    rows = iter(
        _iter_csv_records(
            export_path,
            resource_policy=policy,
            require_exact_width=False,
            expected_identity=expected_identity,
            retained_payload_bytes=reader_retained_bytes,
        )
    )
    with (
        closing_dataset_iterator(rows),
        _IndexInfoContinuationSpool(
            record_limit_bytes=record_limit_bytes,
            scratch_parent=export_path.parent,
        ) as continuations,
    ):
        pending_row: list[str] | None = None
        pending_start: int | None = None
        pending_bytes = 0
        complete_row: list[str] | None = None
        complete_bytes = 0
        row_number = 1
        while True:
            reader_retained_bytes[0] = downstream_retained_bytes[0] + (
                _decoded_rows_owner_known_bytes(
                    payload_bytes=pending_bytes + complete_bytes,
                    rows=(pending_row, complete_row),
                )
            )
            try:
                header, record, source_bytes = next(rows)
            except StopIteration:
                break
            row_number += 1
            if pending_row is not None:
                if _record_starts_index_info(record):
                    raise ValueError(f"malformed_csmar_csv_row: {pending_start}")
                remaining_columns = len(header) - len(pending_row)
                completes_row = (
                    len(pending_row) <= 6 and record.field_count >= remaining_columns
                )
                next_bytes = pending_bytes + source_bytes
                if next_bytes > record_limit_bytes:
                    raise ValueError(LOGICAL_RECORD_LIMIT_ERROR)
                pending_slot_bytes = _decoded_rows_slot_bytes((pending_row,))
                suffix_slot_bytes = (
                    remaining_columns * struct.calcsize("P") if completes_row else 0
                )
                _require_raw_conversion_headroom(
                    logical_bytes=next_bytes + pending_slot_bytes + suffix_slot_bytes,
                    retained_previous_bytes=(
                        downstream_retained_bytes[0]
                        + _decoded_rows_owner_known_bytes(
                            payload_bytes=complete_bytes,
                            rows=(complete_row,),
                        )
                    ),
                    resource_policy=policy,
                )
                suffix = continuations.append_fields(
                    record.iter_fields(),
                    field_count=record.field_count,
                    suffix_count=remaining_columns if completes_row else 0,
                    include_empty=False,
                    retained_payload_bytes=(
                        downstream_retained_bytes[0]
                        + _decoded_rows_owner_known_bytes(
                            payload_bytes=complete_bytes,
                            rows=(complete_row,),
                        )
                        + pending_slot_bytes
                        + suffix_slot_bytes
                    ),
                    source_bytes=next_bytes,
                    resource_policy=policy,
                )
                pending_bytes = next_bytes
                if completes_row:
                    pending_row[-1] = continuations.joined(
                        pending_row[-1],
                        retained_payload_bytes=(
                            downstream_retained_bytes[0]
                            + _decoded_rows_owner_known_bytes(
                                payload_bytes=complete_bytes,
                                rows=(complete_row,),
                            )
                            + pending_slot_bytes
                            + suffix_slot_bytes
                        ),
                        source_bytes=next_bytes,
                        resource_policy=policy,
                    )
                    pending_row.extend(suffix)
                    complete_row = _normalize_index_info_row(
                        pending_row,
                        len(header),
                        pending_start,
                        resource_policy=policy,
                        retained_payload_bytes=(
                            downstream_retained_bytes[0]
                            + _decoded_rows_owner_known_bytes(
                                payload_bytes=complete_bytes,
                                rows=(complete_row,),
                            )
                        ),
                        source_bytes=next_bytes,
                    )
                    complete_bytes = pending_bytes
                    continuations.clear()
                    pending_row = None
                    pending_start = None
                    pending_bytes = 0
                del record, suffix
                continue

            if record.field_count > len(header):
                if _record_starts_index_info(record):
                    row = _normalize_bounded_index_info_record(
                        record,
                        row_number,
                        continuation_spool=continuations,
                        resource_policy=policy,
                        retained_payload_bytes=(
                            downstream_retained_bytes[0]
                            + _decoded_rows_owner_known_bytes(
                                payload_bytes=complete_bytes,
                                rows=(complete_row,),
                            )
                        ),
                        source_bytes=source_bytes,
                    )
                elif complete_row is not None:
                    if complete_bytes + source_bytes > record_limit_bytes:
                        raise ValueError(LOGICAL_RECORD_LIMIT_ERROR)
                    continuations.append_fields(
                        record.iter_fields(),
                        field_count=record.field_count,
                        retained_payload_bytes=(
                            downstream_retained_bytes[0]
                            + _decoded_rows_slot_bytes((complete_row,))
                        ),
                        source_bytes=complete_bytes + source_bytes,
                        resource_policy=policy,
                    )
                    complete_bytes += source_bytes
                    del record
                    continue
                else:
                    raise ValueError(f"malformed_csmar_csv_row: {row_number}")
            else:
                row = record.materialize(max_fields=len(header))

            if len(row) < len(header):
                if _looks_like_index_info_record_start(row):
                    if complete_row is not None:
                        if continuations:
                            complete_row[-1] = continuations.joined(
                                complete_row[-1],
                                retained_payload_bytes=(
                                    downstream_retained_bytes[0]
                                    + _decoded_rows_slot_bytes((complete_row,))
                                ),
                                source_bytes=complete_bytes,
                                resource_policy=policy,
                            )
                        yield header, complete_row, complete_bytes
                        complete_row = None
                    pending_row = row
                    pending_start = row_number
                    continuations.clear()
                    pending_bytes = source_bytes
                    complete_row = None
                    complete_bytes = 0
                    del row
                    continue
                if complete_row is not None:
                    if complete_bytes + source_bytes > record_limit_bytes:
                        raise ValueError(LOGICAL_RECORD_LIMIT_ERROR)
                    continuations.append_fields(
                        row,
                        field_count=len(row),
                        retained_payload_bytes=(
                            downstream_retained_bytes[0]
                            + _decoded_rows_slot_bytes((complete_row,))
                        ),
                        source_bytes=complete_bytes + source_bytes,
                        resource_policy=policy,
                    )
                    complete_bytes += source_bytes
                    del row
                    continue
                raise ValueError(f"malformed_csmar_csv_row: {row_number}")
            if complete_row is not None:
                if continuations:
                    complete_row[-1] = continuations.joined(
                        complete_row[-1],
                        retained_payload_bytes=(
                            downstream_retained_bytes[0]
                            + _decoded_rows_slot_bytes((complete_row,))
                        ),
                        source_bytes=complete_bytes,
                        resource_policy=policy,
                    )
                yield header, complete_row, complete_bytes
                complete_row = None
            complete_row = _normalize_index_info_row(
                row,
                len(header),
                row_number,
                resource_policy=policy,
                retained_payload_bytes=(
                    downstream_retained_bytes[0]
                    + _decoded_rows_owner_known_bytes(
                        payload_bytes=complete_bytes,
                        rows=(complete_row,),
                    )
                ),
                source_bytes=source_bytes,
            )
            del row
            continuations.clear()
            complete_bytes = source_bytes
            del record
        if pending_row is not None:
            raise ValueError(f"malformed_csmar_csv_row: {pending_start}")
        if complete_row is not None:
            if continuations:
                complete_row[-1] = continuations.joined(
                    complete_row[-1],
                    retained_payload_bytes=(
                        downstream_retained_bytes[0]
                        + _decoded_rows_slot_bytes((complete_row,))
                    ),
                    source_bytes=complete_bytes,
                    resource_policy=policy,
                )
            yield header, complete_row, complete_bytes
            complete_row = None


def _normalize_index_info_row(
    row: list[str],
    column_count: int,
    row_number: int | None,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
    retained_payload_bytes: int = 0,
    source_bytes: int = 0,
) -> list[str]:
    if len(row) > column_count:
        row = _normalize_long_index_info_row(
            row,
            row_number,
            resource_policy=resource_policy,
            retained_payload_bytes=retained_payload_bytes,
            source_bytes=source_bytes,
        )
    if len(row) != column_count or not _looks_like_index_info_record_start(row):
        raise ValueError(f"malformed_csmar_csv_row: {row_number}")
    return row


def _record_starts_index_info(record: BoundedCsvRecord) -> bool:
    first = next(record.iter_fields(), "")
    return bool(INDEX_INFO_CODE_RE.fullmatch(first))


def _normalize_bounded_index_info_record(
    record: BoundedCsvRecord,
    row_number: int | None,
    *,
    continuation_spool: _IndexInfoContinuationSpool,
    resource_policy: RuntimeResourcePolicy,
    retained_payload_bytes: int,
    source_bytes: int,
) -> list[str]:
    """Normalize an over-width index record without a field-count-sized list."""

    window: list[str] = []
    candidate_start: int | None = None
    for position, value in enumerate(record.iter_fields()):
        window.append(value)
        if len(window) > 5:
            del window[0]
        start = position - 4
        if start < 6 or len(window) != 5:
            continue
        if all(
            _looks_like_source_category_code(item, allowed_values)
            for item, allowed_values in zip(
                window,
                INDEX_INFO_SOURCE_CODE_ENUMS,
                strict=True,
            )
        ):
            if candidate_start is not None:
                raise ValueError(f"malformed_csmar_csv_row: {row_number}")
            candidate_start = start
    if candidate_start is None or candidate_start + 5 >= record.field_count:
        raise ValueError(f"malformed_csmar_csv_row: {row_number}")
    fixed = list(islice(record.iter_fields(), 5))
    codes = list(islice(record.iter_fields(), candidate_start, candidate_start + 5))
    continuation_spool.clear()
    sample_field_count = candidate_start - 5
    continuation_spool.append_fields(
        islice(record.iter_fields(), 5, candidate_start),
        field_count=sample_field_count,
        retained_payload_bytes=retained_payload_bytes,
        source_bytes=source_bytes,
        resource_policy=resource_policy,
    )
    sample = continuation_spool.joined(
        "",
        retained_payload_bytes=retained_payload_bytes,
        source_bytes=source_bytes,
        resource_policy=resource_policy,
    )
    continuation_spool.clear()
    formula_field_count = record.field_count - candidate_start - 5
    continuation_spool.append_fields(
        islice(record.iter_fields(), candidate_start + 5, None),
        field_count=formula_field_count,
        retained_payload_bytes=retained_payload_bytes + _utf8_size(sample),
        source_bytes=source_bytes,
        resource_policy=resource_policy,
    )
    formula = continuation_spool.joined(
        "",
        retained_payload_bytes=retained_payload_bytes + _utf8_size(sample),
        source_bytes=source_bytes,
        resource_policy=resource_policy,
    )
    continuation_spool.clear()
    return [*fixed, sample, *codes, formula]


def _normalize_long_index_info_row(
    row: list[str],
    row_number: int | None,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
    retained_payload_bytes: int = 0,
    source_bytes: int = 0,
) -> list[str]:
    candidate_start: int | None = None
    for code_start in range(6, len(row) - 5):
        if not all(
            _looks_like_source_category_code(
                row[code_start + offset],
                allowed_values,
            )
            for offset, allowed_values in enumerate(INDEX_INFO_SOURCE_CODE_ENUMS)
        ):
            continue
        if candidate_start is not None:
            raise ValueError(f"malformed_csmar_csv_row: {row_number}")
        candidate_start = code_start
    if candidate_start is None:
        raise ValueError(f"malformed_csmar_csv_row: {row_number}")
    if resource_policy is not None:
        _require_index_info_reconstruction_headroom(
            retained_payload_bytes=retained_payload_bytes,
            source_bytes=source_bytes,
            result_bytes=source_bytes,
            resource_policy=resource_policy,
        )
    sample_scope = ",".join(row[5:candidate_start])
    formula = ",".join(row[candidate_start + 5 :])
    return [
        row[0],
        row[1],
        row[2],
        row[3],
        row[4],
        sample_scope,
        row[candidate_start],
        row[candidate_start + 1],
        row[candidate_start + 2],
        row[candidate_start + 3],
        row[candidate_start + 4],
        formula,
    ]


def _require_index_info_reconstruction_headroom(
    *,
    retained_payload_bytes: int,
    source_bytes: int,
    result_bytes: int,
    resource_policy: RuntimeResourcePolicy,
) -> None:
    retained_payload_span_bytes = retained_payload_bytes
    live_source_span_bytes = source_bytes
    string_builder_span_bytes = result_bytes
    returned_string_span_bytes = result_bytes
    if (
        retained_payload_span_bytes
        + live_source_span_bytes
        + string_builder_span_bytes
        + returned_string_span_bytes
        > resource_policy.task_memory_budget_bytes
    ):
        raise DatasetReadCapacityExceeded(
            "csmar_raw_working_set_exceeds_task_memory_budget"
        )


def _looks_like_index_info_record_start(row: list[str]) -> bool:
    return bool(row) and bool(INDEX_INFO_CODE_RE.fullmatch(row[0]))


def _looks_like_source_category_code(
    value: str,
    allowed_values: frozenset[str],
) -> bool:
    stripped = value.strip()
    return stripped == "" or stripped in allowed_values


def _clean_index_info_source_codes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame
    for column, allowed_values in zip(
        INDEX_INFO_SOURCE_CODE_COLUMNS,
        INDEX_INFO_SOURCE_CODE_ENUMS,
        strict=True,
    ):
        if column not in result.columns:
            continue
        values = result[column].astype("string")
        valid = values.isna() | values.isin(allowed_values)
        result.loc[~valid, column] = pd.NA
    return result


def iter_private_export_chunks(
    path: str | Path,
    *,
    dataset: str,
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    """Yield raw export chunks for large time-series files."""

    if type(chunksize) is not int or chunksize <= 0:
        raise ValueError("invalid_csmar_chunksize")

    bound_policy = current_runtime_resource_policy()

    def activate() -> Iterator[pd.DataFrame]:
        policy = bound_policy or active_runtime_resource_policy()
        with runtime_resource_policy_scope(policy):
            yield from iter_private_export_chunks_with_policy(
                path,
                dataset=dataset,
                chunksize=chunksize,
                resource_policy=policy,
            )

    return activate()


def iter_private_export_chunks_with_policy(
    path: str | Path,
    *,
    dataset: str,
    chunksize: int,
    resource_policy: RuntimeResourcePolicy,
    expected_identity: DatasetFileIdentity | None = None,
    retained_payload_bytes: list[int] | None = None,
) -> Iterator[pd.DataFrame]:
    require_raw_resource_policy(resource_policy)
    if type(dataset) is not str or dataset not in RAW_TABLE_CONTRACTS:
        raise ValueError("unknown_csmar_dataset")
    export_path = Path(path)
    previous_yielded_bytes = [0]
    downstream_retained_bytes = (
        retained_payload_bytes if retained_payload_bytes is not None else [0]
    )
    if dataset == "index_info":
        pending: list[list[str]] = []
        pending_bytes = 0
        pending_cell_count = 0
        batch_target_bytes = 0
        header: list[str] | None = None
        source_retained_bytes = [0]
        rows = iter(
            iter_index_info_rows(
                export_path,
                resource_policy=resource_policy,
                expected_identity=expected_identity,
                retained_payload_bytes=source_retained_bytes,
            )
        )
        with closing_dataset_iterator(rows):
            while True:
                source_retained_bytes[0] = (
                    previous_yielded_bytes[0]
                    + downstream_retained_bytes[0]
                    + _pending_decoded_owner_known_bytes(
                        payload_bytes=pending_bytes,
                        row_count=len(pending),
                        cell_count=pending_cell_count,
                    )
                )
                try:
                    header, row, row_bytes = next(rows)
                except StopIteration:
                    break
                next_pending_bytes = pending_bytes + row_bytes
                next_pending_cell_count = pending_cell_count + len(row)
                _require_raw_conversion_headroom(
                    logical_bytes=_pending_decoded_owner_known_bytes(
                        payload_bytes=next_pending_bytes,
                        row_count=len(pending) + 1,
                        cell_count=next_pending_cell_count,
                    ),
                    retained_previous_bytes=(
                        previous_yielded_bytes[0] + downstream_retained_bytes[0]
                    ),
                    resource_policy=resource_policy,
                )
                pending.append(row)
                pending_bytes = next_pending_bytes
                pending_cell_count = next_pending_cell_count
                if len(pending) == chunksize or pending_bytes >= max(
                    1, batch_target_bytes
                ):
                    logical_bytes = pending_bytes
                    _require_raw_conversion_headroom(
                        logical_bytes=logical_bytes
                        + _raw_frame_owner_known_bytes(pending, header),
                        retained_previous_bytes=(
                            previous_yielded_bytes[0] + downstream_retained_bytes[0]
                        ),
                        resource_policy=resource_policy,
                    )
                    frame = _raw_frame(pending, header)
                    pending.clear()
                    pending_bytes = 0
                    pending_cell_count = 0
                    frame_payload_bytes = pandas_frame_payload_bytes(frame)
                    yield from _yield_bounded_export_frame(
                        _clean_index_info_source_codes(frame),
                        resource_policy=resource_policy,
                        retained_previous_bytes=previous_yielded_bytes,
                    )
                    batch_target_bytes = _next_batch_target_bytes(
                        logical_bytes=logical_bytes,
                        frame_payload_bytes=frame_payload_bytes,
                        resource_policy=resource_policy,
                        retained_payload_bytes=(
                            previous_yielded_bytes[0] + downstream_retained_bytes[0]
                        ),
                    )
                    del frame
        if pending:
            _require_raw_conversion_headroom(
                logical_bytes=pending_bytes
                + _raw_frame_owner_known_bytes(pending, header),
                retained_previous_bytes=(
                    previous_yielded_bytes[0] + downstream_retained_bytes[0]
                ),
                resource_policy=resource_policy,
            )
            frame = _raw_frame(pending, header)
            pending.clear()
            pending_cell_count = 0
            yield from _yield_bounded_export_frame(
                _clean_index_info_source_codes(frame),
                resource_policy=resource_policy,
                retained_previous_bytes=previous_yielded_bytes,
            )
            del frame
        return

    pending = []
    pending_bytes = 0
    pending_cell_count = 0
    batch_target_bytes = 0
    header = None
    source_retained_bytes = [0]
    rows = iter(
        _iter_csv_records(
            export_path,
            resource_policy=resource_policy,
            require_exact_width=True,
            expected_identity=expected_identity,
            retained_payload_bytes=source_retained_bytes,
        )
    )
    with closing_dataset_iterator(rows):
        while True:
            source_retained_bytes[0] = (
                previous_yielded_bytes[0]
                + downstream_retained_bytes[0]
                + _pending_decoded_owner_known_bytes(
                    payload_bytes=pending_bytes,
                    row_count=len(pending),
                    cell_count=pending_cell_count,
                )
            )
            try:
                header, row, row_bytes = next(rows)
            except StopIteration:
                break
            next_pending_bytes = pending_bytes + row_bytes
            next_pending_cell_count = pending_cell_count + len(row)
            _require_raw_conversion_headroom(
                logical_bytes=_pending_decoded_owner_known_bytes(
                    payload_bytes=next_pending_bytes,
                    row_count=len(pending) + 1,
                    cell_count=next_pending_cell_count,
                ),
                retained_previous_bytes=(
                    previous_yielded_bytes[0] + downstream_retained_bytes[0]
                ),
                resource_policy=resource_policy,
            )
            pending.append(row)
            pending_bytes = next_pending_bytes
            pending_cell_count = next_pending_cell_count
            if len(pending) == chunksize or pending_bytes >= max(1, batch_target_bytes):
                logical_bytes = pending_bytes
                _require_raw_conversion_headroom(
                    logical_bytes=logical_bytes
                    + _raw_frame_owner_known_bytes(pending, header),
                    retained_previous_bytes=(
                        previous_yielded_bytes[0] + downstream_retained_bytes[0]
                    ),
                    resource_policy=resource_policy,
                )
                frame = _raw_frame(pending, header)
                pending.clear()
                pending_bytes = 0
                pending_cell_count = 0
                frame_payload_bytes = pandas_frame_payload_bytes(frame)
                yield from _yield_bounded_export_frame(
                    frame,
                    resource_policy=resource_policy,
                    retained_previous_bytes=previous_yielded_bytes,
                )
                batch_target_bytes = _next_batch_target_bytes(
                    logical_bytes=logical_bytes,
                    frame_payload_bytes=frame_payload_bytes,
                    resource_policy=resource_policy,
                    retained_payload_bytes=(
                        previous_yielded_bytes[0] + downstream_retained_bytes[0]
                    ),
                )
                del frame
    if pending:
        _require_raw_conversion_headroom(
            logical_bytes=pending_bytes + _raw_frame_owner_known_bytes(pending, header),
            retained_previous_bytes=(
                previous_yielded_bytes[0] + downstream_retained_bytes[0]
            ),
            resource_policy=resource_policy,
        )
        frame = _raw_frame(pending, header)
        pending.clear()
        pending_cell_count = 0
        yield from _yield_bounded_export_frame(
            frame,
            resource_policy=resource_policy,
            retained_previous_bytes=previous_yielded_bytes,
        )
        del frame


def _iter_csv_records(
    path: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    require_exact_width: bool,
    expected_identity: DatasetFileIdentity | None = None,
    retained_payload_bytes: list[int] | None = None,
) -> Iterator[tuple[list[str], list[str] | BoundedCsvRecord, int]]:
    caller_retained_bytes = (
        retained_payload_bytes if retained_payload_bytes is not None else [0]
    )
    reader_retained_bytes = [caller_retained_bytes[0]]
    rows = iter(
        iter_export_csv_records(
            path,
            resource_policy=resource_policy,
            expected_identity=expected_identity,
            retained_payload_bytes=reader_retained_bytes,
            materialize_records=require_exact_width,
            first_record_field_count_error="csmar_raw_export_schema_invalid",
        )
    )
    with closing_dataset_iterator(rows):
        header_record = next(rows, None)
        if header_record is None:
            return
        header = (
            header_record[0]
            if require_exact_width
            else materialize_header_record(header_record[0])
        )
        if require_exact_width and len(header) != len(set(header)):
            raise ValueError("duplicate_csmar_provider_column")
        del header_record
        header_owner_bytes = _pending_decoded_owner_known_bytes(
            payload_bytes=sum(_utf8_size(column) for column in header),
            row_count=1,
            cell_count=len(header),
        )
        while True:
            reader_retained_bytes[0] = caller_retained_bytes[0] + header_owner_bytes
            try:
                record, source_bytes = next(rows)
            except StopIteration:
                break
            row: list[str] | BoundedCsvRecord = record
            if require_exact_width:
                if len(record) != len(header):
                    raise ValueError("malformed_csmar_csv_row")
                row = record
                del record
            yield header, row, source_bytes
            del row
            if not require_exact_width:
                del record


def _next_batch_target_bytes(
    *,
    logical_bytes: int,
    frame_payload_bytes: int,
    resource_policy: RuntimeResourcePolicy,
    retained_payload_bytes: int = 0,
) -> int:
    if logical_bytes <= 0 or frame_payload_bytes <= 0:
        return 1
    available_bytes = max(
        1,
        resource_policy.task_memory_budget_bytes - retained_payload_bytes,
    )
    measured_total_bytes = logical_bytes + frame_payload_bytes
    return min(
        resource_policy.payload_target_bytes,
        max(1, available_bytes * logical_bytes // measured_total_bytes),
    )


def _raw_frame(rows: list[list[str]], header: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=header)
    frame.replace("", pd.NA, inplace=True)
    for column in frame.columns:
        if column in STRING_CODE_DTYPES or column in INDEX_INFO_SOURCE_CODE_COLUMNS:
            continue
        try:
            numeric = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError):
            continue
        frame[column] = numeric
    return frame


def _raw_frame_owner_known_bytes(
    rows: list[list[str]], header: list[str] | None
) -> int:
    """Bound owner-known list, frame, and numeric decode buffers before allocation."""

    if not rows or header is None:
        return 0
    pointer_bytes = struct.calcsize("P")
    cells = sum(len(row) for row in rows)
    pending_backing_bytes = len(rows) * pointer_bytes + cells * pointer_bytes
    frame_reference_bytes = (len(header) + cells) * pointer_bytes
    numeric_decode_bytes = cells * 8
    return pending_backing_bytes + frame_reference_bytes + numeric_decode_bytes


def _pending_decoded_owner_known_bytes(
    *,
    payload_bytes: int,
    row_count: int,
    cell_count: int,
) -> int:
    pointer_bytes = struct.calcsize("P")
    return payload_bytes + (row_count + cell_count) * pointer_bytes


def _decoded_rows_owner_known_bytes(
    *,
    payload_bytes: int,
    rows: Iterable[list[str] | None],
) -> int:
    return payload_bytes + _decoded_rows_slot_bytes(rows)


def _decoded_rows_slot_bytes(rows: Iterable[list[str] | None]) -> int:
    pointer_bytes = struct.calcsize("P")
    return sum((1 + len(row)) * pointer_bytes for row in rows if row is not None)


def _bounded_export_frame(
    frame: pd.DataFrame,
    *,
    resource_policy: RuntimeResourcePolicy,
) -> Iterator[pd.DataFrame]:
    try:
        yield from iter_bounded_frame_chunks(
            frame,
            target_chunk_bytes=resource_policy.payload_target_bytes,
        )
    except ValueError as exc:
        if str(exc) == "dataset_row_exceeds_target_chunk_bytes":
            raise ValueError(LOGICAL_RECORD_LIMIT_ERROR) from exc
        raise


def _yield_bounded_export_frame(
    frame: pd.DataFrame,
    *,
    resource_policy: RuntimeResourcePolicy,
    retained_previous_bytes: list[int],
) -> Iterator[pd.DataFrame]:
    parent_payload_bytes = pandas_frame_payload_bytes(frame)
    dependent_count = [0]
    owner_active = [True]
    parent_is_counted = [True]
    retained_previous_bytes[0] += parent_payload_bytes

    def release_dependent() -> None:
        dependent_count[0] -= 1
        if dependent_count[0] == 0 and not owner_active[0] and parent_is_counted[0]:
            retained_previous_bytes[0] -= parent_payload_bytes
            parent_is_counted[0] = False

    chunks = iter(_bounded_export_frame(frame, resource_policy=resource_policy))
    try:
        with closing_dataset_iterator(chunks):
            while True:
                try:
                    chunk = next(chunks)
                except StopIteration:
                    break
                finalizer = weakref.finalize(chunk, release_dependent)
                finalizer.atexit = False
                dependent_count[0] += 1
                yield chunk
                del chunk, finalizer
    finally:
        owner_active[0] = False
        if dependent_count[0] == 0 and parent_is_counted[0]:
            retained_previous_bytes[0] -= parent_payload_bytes
            parent_is_counted[0] = False


def _require_raw_conversion_headroom(
    *,
    logical_bytes: int,
    retained_previous_bytes: int,
    resource_policy: RuntimeResourcePolicy,
) -> None:
    if (
        logical_bytes + retained_previous_bytes
        > resource_policy.task_memory_budget_bytes
    ):
        raise DatasetReadCapacityExceeded(
            "csmar_raw_working_set_exceeds_task_memory_budget"
        )


def dataset_files_with_identity_by_id(
    raw_root: Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[tuple[str, Path, DatasetFileIdentity]]:
    """Yield inspected files with the identity bound during inspection."""

    bound_policy = bind_runtime_resource_policy(resource_policy)
    root = Path(raw_root)

    def iter_files() -> Iterator[tuple[str, Path, DatasetFileIdentity]]:
        policy = bound_policy or active_runtime_resource_policy()
        with runtime_resource_policy_scope(policy):
            with _raw_export_path_session(root) as (paths, verify_root):
                with ExactPrimaryKeyIndex(
                    primary_key=("dataset_path",),
                    timestamp_columns=(),
                    scratch_parent=root.parent,
                ) as index:
                    files = _iter_indexed_raw_export_files(
                        root,
                        paths=paths,
                        verify_root=verify_root,
                        index=index,
                        namespace="csmar-dataset-physical-exports",
                    )
                    for path, identity in files:
                        verify_root()
                        result = _inspect_raw_export_with_identity(
                            path,
                            resource_policy=policy,
                            expected_identity=identity,
                        )
                        verify_root()
                        if result.error is not None:
                            raise ValueError(
                                "csmar_raw_export_inspection_failed"
                            ) from None
                        if result.inspection is None:
                            raise ValueError("csmar_raw_export_inspection_failed")
                        if result.inspection.missing_required_columns:
                            raise ValueError("csmar_raw_export_schema_invalid")
                        if raw_export_file_identity(path) != identity:
                            raise ValueError("csmar_raw_export_identity_changed")
                        verify_root()
                        relative = path.relative_to(root).as_posix().encode("utf-8")
                        dataset = result.inspection.dataset.encode("utf-8")
                        identity_payload = b"\0".join(
                            str(value).encode("ascii")
                            for value in (
                                identity.device,
                                identity.inode,
                                identity.size,
                                identity.modified_ns,
                            )
                        )
                        payload = dataset + b"\0" + relative + b"\0" + identity_payload
                        verify_root()
                        index.add_identity_payloads(
                            namespace="csmar-dataset-paths",
                            payloads=((dataset + b"\0" + relative, payload),),
                        )
                        verify_root()
                    payloads = iter(
                        index.identity_payload_values(namespace="csmar-dataset-paths")
                    )
                    while True:
                        verify_root()
                        try:
                            payload = next(payloads)
                        except StopIteration:
                            verify_root()
                            return
                        verify_root()
                        dataset, relative, device, inode, size, modified_ns = (
                            payload.split(b"\0")
                        )
                        path = root / relative.decode("utf-8")
                        identity = DatasetFileIdentity(
                            device=int(device),
                            inode=int(inode),
                            size=int(size),
                            modified_ns=int(modified_ns),
                        )
                        verify_root()
                        if raw_export_file_identity(path) != identity:
                            raise ValueError("csmar_raw_export_identity_changed")
                        verify_root()
                        yield dataset.decode("utf-8"), path, identity
                        verify_root()

    return iter_files()


def _iter_indexed_raw_export_files(
    root: Path,
    *,
    paths: Iterator[Path],
    verify_root: Callable[[], None],
    index: ExactPrimaryKeyIndex,
    namespace: str,
) -> Iterator[tuple[Path, DatasetFileIdentity]]:
    path_namespace = f"{namespace}-paths"
    for path in paths:
        verify_root()
        identity = raw_export_file_identity(path)
        verify_root()
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = relative + b"\0" + _raw_export_identity_payload(identity)
        verify_root()
        inserted = index.add_identity_payloads(
            namespace=namespace,
            payloads=((_raw_export_physical_key(identity), payload),),
        )
        verify_root()
        if inserted != 1:
            raise ValueError("csmar_raw_export_identity_changed")
        index.add_identity_payloads(
            namespace=path_namespace,
            payloads=((relative, payload),),
        )
        verify_root()
    payloads = iter(index.identity_payload_values(namespace=path_namespace))
    while True:
        verify_root()
        try:
            payload = next(payloads)
        except StopIteration:
            verify_root()
            return
        verify_root()
        relative, device, inode, size, modified_ns = payload.split(b"\0")
        path = root / relative.decode("utf-8")
        identity = DatasetFileIdentity(
            device=int(device),
            inode=int(inode),
            size=int(size),
            modified_ns=int(modified_ns),
        )
        verify_root()
        if raw_export_file_identity(path) != identity:
            raise ValueError("csmar_raw_export_identity_changed")
        verify_root()
        yield path, identity
        verify_root()


def _raw_export_physical_key(identity: DatasetFileIdentity) -> bytes:
    return b"\0".join(
        str(value).encode("ascii") for value in (identity.device, identity.inode)
    )


def _raw_export_identity_payload(identity: DatasetFileIdentity) -> bytes:
    return b"\0".join(
        str(value).encode("ascii")
        for value in (
            identity.device,
            identity.inode,
            identity.size,
            identity.modified_ns,
        )
    )


def inspect_raw_export_tree(root: Path) -> Iterator[RawExportIntakeResult]:
    """Inspect supported raw exports under a private raw directory."""

    bound_policy = bind_runtime_resource_policy(None)
    raw_root = Path(root)

    def iter_inspections() -> Iterator[RawExportIntakeResult]:
        policy = bound_policy or active_runtime_resource_policy()
        with runtime_resource_policy_scope(policy):
            with _raw_export_path_session(raw_root) as (paths, verify_root):
                with ExactPrimaryKeyIndex(
                    primary_key=("path",),
                    timestamp_columns=(),
                    scratch_parent=raw_root.parent,
                ) as index:
                    files = _iter_indexed_raw_export_files(
                        raw_root,
                        paths=paths,
                        verify_root=verify_root,
                        index=index,
                        namespace="csmar-inspection-physical-exports",
                    )
                    with closing_dataset_iterator(files):
                        for path, identity in files:
                            verify_root()
                            result = _inspect_raw_export_with_identity(
                                path,
                                resource_policy=policy,
                                expected_identity=identity,
                            )
                            verify_root()
                            yield result
                            verify_root()

    return iter_inspections()


__all__ = [
    "RawExportIntakeResult",
    "STRING_CODE_DTYPES",
    "SUPPORTED_RAW_EXPORT_SUFFIXES",
    "dataset_files_with_identity_by_id",
    "inspect_raw_export",
    "inspect_raw_export_tree",
    "iter_private_export_chunks",
    "iter_raw_export_files",
]
