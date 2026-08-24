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
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import groupby, islice
from pathlib import Path

import pandas as pd

from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    MemoryLease,
    RuntimeResourcePolicy,
    _OwnedValue,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    require_free_disk_headroom,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_key_tracking import ExactPrimaryKeyIndex
from quant_research_platform.storage._dataset_paths import require_unaliased_path
from quant_research_platform.storage._dataset_physical import (
    DatasetFileIdentity,
    _dataset_file_identity_bytes,
)
from quant_research_platform.storage.errors import DatasetReadCapacityExceeded

from .field_mappings import RAW_TABLE_CONTRACTS
from .raw_schemas import (
    _CSV_RECORD_LEASE_BOUND_BYTES,
    LOGICAL_RECORD_LIMIT_ERROR,
    MAX_LOGICAL_RECORD_BYTES,
    BoundedCsvRecord,
    RawExportInspection,
    inspect_raw_export_file,
    iter_export_csv_records,
    materialize_header_record,
    raw_export_file_identity,
    require_raw_resource_policy,
)

SUPPORTED_RAW_EXPORT_SUFFIXES = {".csv"}
STRING_CODE_DTYPES = {
    "Stkcd": str,
    "Symbol": str,
    "Indexcd": str,
    "Stknme": str,
    "Listdt": str,
    "DelistedDate": str,
    "Nnindcd": str,
    "Nnindnme": str,
    "Statco": str,
    "Statdt": str,
    "Trddt": str,
    "Shrchgdt": str,
    "TradingDate": str,
    "Idxtrd01": str,
    "Idxtrd09": str,
    "Idxinfo01": str,
    "Idxinfo11": str,
    "Idxinfo02": str,
    "Idxinfo04": str,
    "Idxinfo10": str,
    "Clddt": str,
    "State": str,
}
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

    def __init__(
        self,
        *,
        record_limit_bytes: int,
        scratch_parent: Path,
    ) -> None:
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

    def __exit__(self, _exc_type, exc_value, _traceback) -> bool:  # noqa: ANN001
        try:
            self._handle.close()
        except BaseException as close_error:
            if exc_value is None:
                raise
            exc_value.add_note(
                "CSMAR continuation spool close also failed: "
                f"{type(close_error).__name__}"
            )
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
        source_bytes: int,
        resource_policy: RuntimeResourcePolicy,
        payload_lease: MemoryLease | None = None,
    ) -> list[str]:
        """Stream one continuation prefix and retain only its bounded suffix."""

        prefix_count = field_count - suffix_count
        if prefix_count < 0:
            raise ValueError("malformed_csmar_csv_row")
        _require_raw_conversion_headroom(
            logical_bytes=source_bytes + 4,
            resource_policy=resource_policy,
            payload_lease=payload_lease,
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
                source_bytes=source_bytes,
                resource_policy=resource_policy,
                payload_lease=payload_lease,
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
        source_bytes: int,
        resource_policy: RuntimeResourcePolicy,
        payload_lease: MemoryLease | None = None,
    ) -> int:
        payload_bytes = 0
        codepoint_span = max(1, io.DEFAULT_BUFFER_SIZE // 4)
        for start in range(0, len(value), codepoint_span):
            stop = min(len(value), start + codepoint_span)
            transient_bytes = (stop - start) * 8
            _require_raw_conversion_headroom(
                logical_bytes=source_bytes + transient_bytes,
                resource_policy=resource_policy,
                payload_lease=payload_lease,
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
        source_bytes: int,
        resource_policy: RuntimeResourcePolicy,
        payload_lease: MemoryLease | None = None,
    ) -> str:
        result_bytes = self._joined_payload_bytes(current)
        _require_index_info_reconstruction_headroom(
            source_bytes=source_bytes,
            result_bytes=result_bytes,
            resource_policy=resource_policy,
            payload_lease=payload_lease,
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
            try:
                parent, dirnames, filenames = next(walker)
            except StopIteration:
                return
            require_unaliased_path(parent, error="csmar_raw_export_identity_changed")
            for name in dirnames:
                require_unaliased_path(
                    parent / name,
                    error="csmar_raw_export_identity_changed",
                )
            dirnames.sort()
            for name in sorted(filenames):
                path = parent / name
                if path.suffix.lower() not in SUPPORTED_RAW_EXPORT_SUFFIXES:
                    continue
                require_unaliased_path(path, error="csmar_raw_export_identity_changed")
                if (
                    "[DES]" in path.name
                    or "__description__" in path.name
                    or ".metadata." in path.name
                ):
                    continue
                yield path

    try:
        require_unaliased_path(root, error="csmar_raw_export_identity_changed")
        yield iter_paths(), verify_root
    finally:
        primary = sys.exception()
        if primary is None:
            verify_root()
        else:
            try:
                verify_root()
            except BaseException as verify_error:
                primary.add_note(
                    "CSMAR raw export root verification also failed: "
                    f"{type(verify_error).__name__}"
                )


def _inspect_raw_export_with_identity(
    path: Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
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
    payload_lease_ref: list[MemoryLease | None] | None = None,
) -> Iterable[tuple[list[str], list[str], int]]:
    bound_policy = resource_policy

    def activate() -> Iterator[tuple[list[str], list[str], int]]:
        policy = bound_policy or current_runtime_resource_policy()
        with runtime_resource_policy_scope(policy) as policy:
            yield from _iter_index_info_rows_under_policy(
                export_path,
                resource_policy=policy,
                expected_identity=expected_identity,
                payload_lease_ref=payload_lease_ref,
            )

    return activate()


def _iter_index_info_rows_under_policy(
    export_path: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    expected_identity: DatasetFileIdentity | None,
    payload_lease_ref: list[MemoryLease | None] | None,
) -> Iterator[tuple[list[str], list[str], int]]:
    policy = resource_policy
    record_limit_bytes = MAX_LOGICAL_RECORD_BYTES
    rows = iter(
        _iter_csv_records(
            export_path,
            resource_policy=policy,
            require_exact_width=False,
            expected_identity=expected_identity,
            payload_lease_ref=payload_lease_ref,
        )
    )
    with (
        closing_iterator(rows),
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
            payload_lease = (
                payload_lease_ref[0] if payload_lease_ref is not None else None
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
                    resource_policy=policy,
                    payload_lease=payload_lease,
                )
                suffix = continuations.append_fields(
                    record.iter_fields(),
                    field_count=record.field_count,
                    suffix_count=remaining_columns if completes_row else 0,
                    include_empty=False,
                    source_bytes=next_bytes,
                    resource_policy=policy,
                    payload_lease=payload_lease,
                )
                pending_bytes = next_bytes
                if completes_row:
                    pending_row[-1] = continuations.joined(
                        pending_row[-1],
                        source_bytes=next_bytes,
                        resource_policy=policy,
                        payload_lease=payload_lease,
                    )
                    pending_row.extend(suffix)
                    complete_row = _normalize_index_info_row(
                        pending_row,
                        len(header),
                        pending_start,
                        resource_policy=policy,
                        payload_lease=payload_lease,
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
                        payload_lease=payload_lease,
                        source_bytes=source_bytes,
                    )
                elif complete_row is not None:
                    if complete_bytes + source_bytes > record_limit_bytes:
                        raise ValueError(LOGICAL_RECORD_LIMIT_ERROR)
                    continuations.append_fields(
                        record.iter_fields(),
                        field_count=record.field_count,
                        source_bytes=complete_bytes + source_bytes,
                        resource_policy=policy,
                        payload_lease=payload_lease,
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
                                source_bytes=complete_bytes,
                                resource_policy=policy,
                                payload_lease=payload_lease,
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
                        source_bytes=complete_bytes + source_bytes,
                        resource_policy=policy,
                        payload_lease=payload_lease,
                    )
                    complete_bytes += source_bytes
                    del row
                    continue
                raise ValueError(f"malformed_csmar_csv_row: {row_number}")
            if complete_row is not None:
                if continuations:
                    complete_row[-1] = continuations.joined(
                        complete_row[-1],
                        source_bytes=complete_bytes,
                        resource_policy=policy,
                        payload_lease=payload_lease,
                    )
                yield header, complete_row, complete_bytes
                complete_row = None
            complete_row = _normalize_index_info_row(
                row,
                len(header),
                row_number,
                resource_policy=policy,
                payload_lease=payload_lease,
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
                    source_bytes=complete_bytes,
                    resource_policy=policy,
                    payload_lease=payload_lease,
                )
            yield header, complete_row, complete_bytes
            complete_row = None


def _normalize_index_info_row(
    row: list[str],
    column_count: int,
    row_number: int | None,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
    payload_lease: MemoryLease | None = None,
    source_bytes: int = 0,
) -> list[str]:
    if len(row) > column_count:
        row = _normalize_long_index_info_row(
            row,
            row_number,
            resource_policy=resource_policy,
            payload_lease=payload_lease,
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
    payload_lease: MemoryLease | None = None,
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
        if start < 6 or len(window) != 5 or start + 5 >= record.field_count:
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
        source_bytes=source_bytes,
        resource_policy=resource_policy,
        payload_lease=payload_lease,
    )
    sample = continuation_spool.joined(
        "",
        source_bytes=source_bytes,
        resource_policy=resource_policy,
        payload_lease=payload_lease,
    )
    continuation_spool.clear()
    formula_field_count = record.field_count - candidate_start - 5
    continuation_spool.append_fields(
        islice(record.iter_fields(), candidate_start + 5, None),
        field_count=formula_field_count,
        source_bytes=source_bytes,
        resource_policy=resource_policy,
        payload_lease=payload_lease,
    )
    formula = continuation_spool.joined(
        "",
        source_bytes=source_bytes,
        resource_policy=resource_policy,
        payload_lease=payload_lease,
    )
    continuation_spool.clear()
    return [*fixed, sample, *codes, formula]


def _normalize_long_index_info_row(
    row: list[str],
    row_number: int | None,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
    payload_lease: MemoryLease | None = None,
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
            source_bytes=source_bytes,
            result_bytes=source_bytes,
            resource_policy=resource_policy,
            payload_lease=payload_lease,
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
    source_bytes: int,
    result_bytes: int,
    resource_policy: RuntimeResourcePolicy,
    payload_lease: MemoryLease | None = None,
) -> None:
    live_source_span_bytes = source_bytes
    string_builder_span_bytes = result_bytes
    returned_string_span_bytes = result_bytes
    if payload_lease is None:
        available_bytes = resource_policy.available_memory_bytes
    elif payload_lease.active:
        available_bytes = payload_lease.reserved_bytes
    else:
        available_bytes = 0
    if (
        live_source_span_bytes
        + string_builder_span_bytes
        + returned_string_span_bytes
        > available_bytes
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


def iter_private_export_chunks_with_policy(
    path: str | Path,
    *,
    dataset: str,
    chunksize: int,
    resource_policy: RuntimeResourcePolicy,
    expected_identity: DatasetFileIdentity | None = None,
) -> Iterator[object]:
    require_raw_resource_policy(resource_policy)
    if type(dataset) is not str or dataset not in RAW_TABLE_CONTRACTS:
        raise ValueError("unknown_csmar_dataset")
    export_path = Path(path)
    producer_bound = max(1, resource_policy.available_memory_bytes // 2)
    payload_lease_ref: list[MemoryLease | None] = [None]

    if dataset == "index_info":
        rows = iter(
            iter_index_info_rows(
                export_path,
                resource_policy=resource_policy,
                expected_identity=expected_identity,
                payload_lease_ref=payload_lease_ref,
            )
        )
    else:
        rows = iter(
            _iter_csv_records(
                export_path,
                dataset=dataset,
                resource_policy=resource_policy,
                require_exact_width=True,
                expected_identity=expected_identity,
                payload_lease_ref=payload_lease_ref,
            )
        )
    with closing_iterator(rows):
        try:
            while True:
                lease = yield producer_bound
                payload_lease_ref[0] = lease
                pending: list[object] = []
                pending_bytes = 0
                header: list[str] | None = None
                while len(pending) < chunksize:
                    if pending and (
                        pending_bytes
                        + _raw_frame_owner_known_bytes(pending, header)
                        + _CSV_RECORD_LEASE_BOUND_BYTES
                        + 64 * 1024
                        > producer_bound
                    ):
                        break
                    try:
                        next_header, row, row_bytes = next(rows)
                    except StopIteration:
                        break
                    if header is None:
                        header = next_header
                    pending.append(row)
                    pending_bytes += row_bytes
                    if (
                        pending_bytes
                        + _raw_frame_owner_known_bytes(pending, header)
                        + 64 * 1024
                        > producer_bound
                    ):
                        raise DatasetReadCapacityExceeded(
                            "csmar_raw_working_set_exceeds_task_memory_budget"
                        )
                if not pending or header is None:
                    payload_lease_ref[0] = None
                    return
                if pending_bytes > producer_bound:
                    raise DatasetReadCapacityExceeded(
                        "csmar_raw_working_set_exceeds_task_memory_budget"
                    )
                _require_raw_conversion_headroom(
                    logical_bytes=pending_bytes
                    + _raw_frame_owner_known_bytes(pending, header),
                    resource_policy=resource_policy,
                    payload_lease=lease,
                )
                if dataset == "index_info":
                    frame = _clean_index_info_source_codes(_raw_frame(pending, header))
                else:
                    frame = _raw_frame(pending, header)
                frame_bytes = pandas_frame_payload_bytes(frame)
                actual_bound = pending_bytes + 2 * frame_bytes + 128 * 1024
                if actual_bound > lease.reserved_bytes:
                    raise DatasetReadCapacityExceeded(
                        "csmar_raw_working_set_exceeds_task_memory_budget"
                    )
                lease.shrink_to(actual_bound)
                row = None
                next_header = None
                header = None
                yield _OwnedValue(frame, lease)
                del frame, pending
                payload_lease_ref[0] = None
        finally:
            payload_lease_ref[0] = None


def _iter_csv_records(
    path: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    require_exact_width: bool,
    dataset: str | None = None,
    expected_identity: DatasetFileIdentity | None = None,
    payload_lease_ref: list[MemoryLease | None] | None = None,
) -> Iterator[tuple[list[str], list[str] | BoundedCsvRecord, int]]:
    rows = iter(
        iter_export_csv_records(
            path,
            resource_policy=resource_policy,
            expected_identity=expected_identity,
            payload_lease_ref=payload_lease_ref,
            materialize_records=False,
        )
    )
    with closing_iterator(rows):
        header_record = next(rows, None)
        if header_record is None:
            return
        physical_header = materialize_header_record(header_record[0])
        physical_width = len(physical_header)
        selected_positions: frozenset[int] | None = None
        header = physical_header
        if require_exact_width and dataset is not None:
            from quant_research_platform.sources.csmar.field_mappings import (
                COLUMN_ALIASES,
                COLUMN_MAPPINGS,
                validate_raw_column_names,
            )

            validate_raw_column_names(tuple(physical_header), dataset)
            selected_columns = set(COLUMN_MAPPINGS[dataset])
            selected_columns.update(
                alias
                for aliases in COLUMN_ALIASES.get(dataset, {}).values()
                for alias in aliases
            )
            selected_positions = frozenset(
                index
                for index, column in enumerate(physical_header)
                if column in selected_columns
            )
            header = [physical_header[index] for index in sorted(selected_positions)]
            del physical_header
        del header_record
        while True:
            try:
                record, source_bytes = next(rows)
            except StopIteration:
                break
            row: list[str] | BoundedCsvRecord = record
            if require_exact_width:
                if record.field_count != physical_width:
                    raise ValueError("malformed_csmar_csv_row")
                row = [
                    value
                    for index, value in enumerate(record.iter_fields())
                    if selected_positions is None or index in selected_positions
                ]
                del record
            yield header, row, source_bytes
            del row
            if not require_exact_width:
                del record


def _raw_frame(rows: list[list[str]], header: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=header)
    frame.replace("", pd.NA, inplace=True)
    for column in frame.columns:
        if column in STRING_CODE_DTYPES or column in INDEX_INFO_SOURCE_CODE_COLUMNS:
            frame[column] = frame[column].astype("string")
            continue
        try:
            numeric = pd.to_numeric(frame[column], errors="raise").astype("float64")
        except (TypeError, ValueError):
            raise ValueError("invalid_csmar_numeric_value") from None
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
    numeric_decode_bytes = cells * 16
    return pending_backing_bytes + frame_reference_bytes + numeric_decode_bytes


def _decoded_rows_slot_bytes(rows: Iterable[list[str] | None]) -> int:
    pointer_bytes = struct.calcsize("P")
    return sum((1 + len(row)) * pointer_bytes for row in rows if row is not None)


def _require_raw_conversion_headroom(
    *,
    logical_bytes: int,
    resource_policy: RuntimeResourcePolicy,
    payload_lease: MemoryLease | None = None,
) -> None:
    if payload_lease is None:
        available_bytes = resource_policy.available_memory_bytes
    elif payload_lease.active:
        available_bytes = payload_lease.reserved_bytes
    else:
        available_bytes = 0
    if logical_bytes > available_bytes:
        raise DatasetReadCapacityExceeded(
            "csmar_raw_working_set_exceeds_task_memory_budget"
        )


def dataset_files_with_identity_by_id(
    raw_root: Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[tuple[str, Path, DatasetFileIdentity]]:
    """Yield inspected files with the identity bound during inspection."""

    bound_policy = resource_policy
    root = Path(raw_root)

    def iter_files() -> Iterator[tuple[str, Path, DatasetFileIdentity]]:
        policy = bound_policy or current_runtime_resource_policy()
        with runtime_resource_policy_scope(policy) as policy:
            with _raw_export_path_session(root) as (paths, verify_root):
                index = ExactPrimaryKeyIndex(
                    primary_key=("dataset_path",),
                    timestamp_columns=(),
                    scratch_parent=root.parent,
                )
                with (
                    policy.lease(
                        "source.csmar.raw.identity.index",
                        index.retained_memory_bytes,
                    ),
                    index,
                ):
                    files = _iter_indexed_raw_export_files(
                        root,
                        paths=paths,
                        index=index,
                        namespace="csmar-dataset-physical-exports",
                    )
                    for path, identity in files:
                        result = _inspect_raw_export_with_identity(
                            path,
                            resource_policy=policy,
                            expected_identity=identity,
                        )
                        if result.error is not None:
                            raise ValueError(
                                "csmar_raw_export_inspection_failed"
                            ) from None
                        if result.inspection is None:
                            raise ValueError("csmar_raw_export_inspection_failed")
                        if result.inspection.missing_required_columns:
                            raise ValueError("csmar_raw_export_schema_invalid")
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
                        index.add_identity_payloads(
                            namespace="csmar-dataset-paths",
                            payloads=((dataset + b"\0" + relative, payload),),
                        )
                    payloads = iter(
                        index.identity_payload_values(namespace="csmar-dataset-paths")
                    )
                    for _, dataset_payloads in groupby(
                        payloads,
                        key=lambda payload: payload.split(b"\0", 1)[0],
                    ):
                        for payload in dataset_payloads:
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
                            yield dataset.decode("utf-8"), path, identity
                        verify_root()

    return iter_files()


def _iter_indexed_raw_export_files(
    root: Path,
    *,
    paths: Iterator[Path],
    index: ExactPrimaryKeyIndex,
    namespace: str,
) -> Iterator[tuple[Path, DatasetFileIdentity]]:
    path_namespace = f"{namespace}-paths"
    for path in paths:
        identity = raw_export_file_identity(path)
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = relative + b"\0" + _dataset_file_identity_bytes(identity)
        inserted = index.add_identity_payloads(
            namespace=namespace,
            payloads=((_raw_export_physical_key(identity), payload),),
        )
        if inserted != 1:
            raise ValueError("csmar_raw_export_identity_changed")
        index.add_identity_payloads(
            namespace=path_namespace,
            payloads=((relative, payload),),
        )
    payloads = iter(index.identity_payload_values(namespace=path_namespace))
    while True:
        try:
            payload = next(payloads)
        except StopIteration:
            return
        relative, device, inode, size, modified_ns = payload.split(b"\0")
        path = root / relative.decode("utf-8")
        identity = DatasetFileIdentity(
            device=int(device),
            inode=int(inode),
            size=int(size),
            modified_ns=int(modified_ns),
        )
        yield path, identity


def _raw_export_physical_key(identity: DatasetFileIdentity) -> bytes:
    return b"\0".join(
        str(value).encode("ascii") for value in (identity.device, identity.inode)
    )
