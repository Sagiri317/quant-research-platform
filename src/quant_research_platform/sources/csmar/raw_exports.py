"""CSMAR raw export inspection and source-local IO helpers."""

from __future__ import annotations

import codecs
import io
import mmap
import os
import re
import stat
import struct
import sys
import tempfile
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack, contextmanager
from itertools import groupby, islice
from pathlib import Path

import pandas as pd

from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    RuntimeResourcePolicy,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._scratch_ownership import reserved_temporary_file
from quant_research_platform.storage.errors import DatasetReadCapacityExceeded
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    _file_identity_bytes,
    filesystem_stat,
    filesystem_walk,
    require_unaliased_path,
)

from .field_mappings import RAW_TABLE_CONTRACTS, missing_raw_column_names
from .raw_schemas import (
    LOGICAL_RECORD_LIMIT_ERROR,
    MAX_LOGICAL_RECORD_BYTES,
    BoundedCsvRecord,
    _infer_dataset_from_header,
    infer_dataset_from_native_export,
    inspect_raw_export_file,
    iter_export_csv_records,
    materialize_header_record,
    parse_raw_filename,
    raw_export_file_identity,
    read_export_header,
    require_raw_resource_policy,
)

SUPPORTED_RAW_EXPORT_SUFFIXES = {".csv"}
STRING_CODE_DTYPES = {
    "ReportID": str,
    "AnanmID": str,
    "Stkcd": str,
    "Symbol": str,
    "Indexcd": str,
    "Chgsmp01": str,
    "Chgsmp02": str,
    "Chgsmp03": str,
    "Chgsmp06": str,
    "Chgsmp07": str,
    "Stknme": str,
    "ShortName": str,
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
    "Enddt": str,
    "Constdnme": str,
    "Idxinfo01": str,
    "Idxinfo11": str,
    "Idxinfo02": str,
    "Idxinfo04": str,
    "Idxinfo10": str,
    "Clddt": str,
    "State": str,
    "IndustryClassificationID": str,
    "ImplementDate": str,
    "AbandonDate": str,
    "IndustryCode": str,
    "IndustryName": str,
    "EndDate": str,
    "IndustryCode1": str,
    "IndustryName1": str,
    "INDUSTRYCODE": str,
    "INDCLASSIFYSYSTEMCODE": str,
    "INDUSTRYNAME": str,
    "P_INDUSTRYCODE": str,
    "INDUSTRYID": str,
    "RANK": str,
    "IndClassifySystemCode": str,
    "IndClassifySystem": str,
    "InstitutionName": str,
    "DeclareDate": str,
    "Accper": str,
    "Typrep": str,
    "ChangeDate": str,
    "ChangedItem": str,
    "ValueBefore": str,
    "ValueAfter": str,
    "AnnouncementDate": str,
    "Value_Before": str,
    "Value_After": str,
    "LISTINGDATE": str,
    "LISTINGSTATE": str,
    "Reason": str,
    "Chgtype": str,
    "Annoudt": str,
    "Execudt": str,
    "Fenddt": str,
    "InstitutionID": str,
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


class _IndexInfoContinuationSpool:
    """Bounded private scratch for malformed physical-row continuations."""

    def __init__(
        self,
        *,
        resource_policy: RuntimeResourcePolicy,
    ) -> None:
        require_unaliased_path(
            Path(tempfile.gettempdir()), error="csmar_raw_scratch_parent_invalid"
        )
        self._scratch_context = reserved_temporary_file(resource_policy=resource_policy)
        self._handle, self._scratch = self._scratch_context.__enter__()
        self._count = 0

    def __enter__(self) -> _IndexInfoContinuationSpool:
        return self

    def __exit__(self, _exc_type, exc_value, _traceback) -> bool:  # noqa: ANN001
        return self._scratch_context.__exit__(_exc_type, exc_value, _traceback)

    def clear(self) -> None:
        self._handle.seek(0)
        self._handle.truncate(0)
        self._scratch.resize(0)
        self._count = 0

    def _write(self, payload: bytes) -> None:
        end = self._handle.tell() + len(payload)
        allocated = (end + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
        if allocated > self._scratch.reserved_bytes:
            self._scratch.resize(allocated)
        self._handle.write(payload)

    def append_fields(
        self,
        fields: Iterable[str],
        *,
        field_count: int,
        suffix_count: int = 0,
        include_empty: bool = True,
        payload_lease: MemoryLease,
    ) -> list[str]:
        """Stream one continuation prefix and retain only its bounded suffix."""

        prefix_count = field_count - suffix_count
        if prefix_count < 0:
            raise ValueError("malformed_csmar_csv_row")
        _require_raw_conversion_headroom(
            logical_bytes=4,
            payload_lease=payload_lease,
        )
        self._handle.seek(0, 2)
        frame_start = self._handle.tell()
        self._write(b"\0\0\0\0")
        payload_bytes = 0
        suffix: list[str] = []
        for position, value in enumerate(fields):
            if position >= prefix_count:
                suffix.append(value)
                continue
            if position:
                self._write(b",")
                payload_bytes += 1
            payload_bytes += self._write_text(
                value,
                payload_lease=payload_lease,
            )
        if payload_bytes == 0 and not include_empty:
            self._handle.seek(frame_start)
            self._handle.truncate()
            self._scratch.resize(
                (frame_start + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE
            )
            return suffix
        frame_end = self._handle.tell()
        self._handle.seek(frame_start)
        self._write(struct.pack(">I", payload_bytes))
        self._handle.seek(frame_end)
        self._count += 1
        return suffix

    def _write_text(
        self,
        value: str,
        *,
        payload_lease: MemoryLease,
    ) -> int:
        payload_bytes = 0
        codepoint_span = max(1, io.DEFAULT_BUFFER_SIZE // 4)
        for start in range(0, len(value), codepoint_span):
            stop = min(len(value), start + codepoint_span)
            transient_bytes = (stop - start) * 8
            _require_raw_conversion_headroom(
                logical_bytes=transient_bytes,
                payload_lease=payload_lease,
            )
            text_slice = value[start:stop]
            payload = text_slice.encode("utf-8")
            self._write(payload)
            payload_bytes += len(payload)
            del text_slice, payload
        return payload_bytes

    def __bool__(self) -> bool:
        return self._count > 0

    def joined(
        self,
        current: str,
        *,
        resource_policy: RuntimeResourcePolicy,
        payload_lease: MemoryLease,
    ) -> str:
        result_bytes = self._joined_payload_bytes(current)
        _grow_source_lease(
            additional_bytes=8 * result_bytes,
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
            metadata = filesystem_stat(root, follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError) as error:
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
        if not os.path.samestat(root_identity, current):
            raise ValueError("csmar_raw_export_identity_changed")

    def fail_walk(error: OSError) -> None:
        if isinstance(error, (FileNotFoundError, NotADirectoryError)):
            raise ValueError("csmar_raw_export_identity_changed") from error
        raise error

    def iter_paths() -> Iterator[Path]:
        walker = iter(
            filesystem_walk(
                root,
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
            dirnames[:] = sorted(
                name for name in dirnames if name.casefold() != "raw_only"
            )
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


def iter_index_info_rows(
    export_path: Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
    expected_identity: DatasetFileIdentity | None = None,
) -> Iterable[tuple[list[str], list[str], int]]:
    bound_policy = resource_policy

    def activate() -> Iterator[tuple[list[str], list[str], int]]:
        policy = bound_policy or current_runtime_resource_policy()
        with runtime_resource_policy_scope(policy) as policy:
            yield from _iter_index_info_rows_under_policy(
                export_path,
                resource_policy=policy,
                expected_identity=expected_identity,
            )

    return activate()


def _iter_index_info_rows_under_policy(
    export_path: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    expected_identity: DatasetFileIdentity | None,
) -> Iterator[tuple[list[str], list[str], int]]:
    policy = resource_policy
    record_limit_bytes = MAX_LOGICAL_RECORD_BYTES
    rows = iter(
        _iter_csv_records(
            export_path,
            resource_policy=policy,
            require_exact_width=False,
            expected_identity=expected_identity,
        )
    )
    with (
        closing_iterator(rows),
        policy.lease("source.csmar.index_info.rows", 0) as payload_lease,
        _IndexInfoContinuationSpool(
            resource_policy=policy,
        ) as continuations,
    ):
        pending_row: list[str] | None = None
        pending_start: int | None = None
        pending_bytes = 0
        complete_row: list[str] | None = None
        complete_bytes = 0
        row_number = 1
        header: list[str] | None = None

        def retain_rows(*values: list[str] | None) -> None:
            retained = 4 * io.DEFAULT_BUFFER_SIZE + sum(
                sys.getsizeof(value) + sum(map(sys.getsizeof, value))
                for value in (header, *values)
                if value is not None
            )
            if retained > payload_lease.reserved_bytes:
                additional = retained - payload_lease.reserved_bytes
                _grow_source_lease(
                    additional_bytes=additional,
                    resource_policy=policy,
                    payload_lease=payload_lease,
                )
            else:
                payload_lease.shrink_to(retained)

        try:
            while True:
                retain_rows(pending_row, complete_row)
                try:
                    header, record, source_bytes = next(rows)
                except StopIteration:
                    break
                row_number += 1
                # Streaming reconstruction can retain several decoded field fragments.
                _grow_source_lease(
                    additional_bytes=4 * source_bytes,
                    resource_policy=policy,
                    payload_lease=payload_lease,
                )
                if pending_row is not None:
                    if _record_starts_index_info(record):
                        raise ValueError(f"malformed_csmar_csv_row: {pending_start}")
                    remaining_columns = len(header) - len(pending_row)
                    completes_row = (
                        len(pending_row) <= 6
                        and record.field_count >= remaining_columns
                    )
                    next_bytes = pending_bytes + source_bytes
                    if next_bytes > record_limit_bytes:
                        raise ValueError(LOGICAL_RECORD_LIMIT_ERROR)
                    pending_slot_bytes = _decoded_rows_slot_bytes((pending_row,))
                    suffix_slot_bytes = (
                        remaining_columns * struct.calcsize("P") if completes_row else 0
                    )
                    _require_raw_conversion_headroom(
                        logical_bytes=pending_slot_bytes + suffix_slot_bytes,
                        payload_lease=payload_lease,
                    )
                    suffix = continuations.append_fields(
                        record.iter_fields(),
                        field_count=record.field_count,
                        suffix_count=remaining_columns if completes_row else 0,
                        include_empty=False,
                        payload_lease=payload_lease,
                    )
                    pending_bytes = next_bytes
                    if completes_row:
                        pending_row[-1] = continuations.joined(
                            pending_row[-1],
                            resource_policy=policy,
                            payload_lease=payload_lease,
                        )
                        pending_row.extend(suffix)
                        complete_row = _normalize_index_info_row(
                            pending_row,
                            len(header),
                            pending_start,
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
                        )
                    elif complete_row is not None:
                        if complete_bytes + source_bytes > record_limit_bytes:
                            raise ValueError(LOGICAL_RECORD_LIMIT_ERROR)
                        continuations.append_fields(
                            record.iter_fields(),
                            field_count=record.field_count,
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
                                    resource_policy=policy,
                                    payload_lease=payload_lease,
                                )
                            retain_rows(complete_row, row)
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
                            resource_policy=policy,
                            payload_lease=payload_lease,
                        )
                    retain_rows(complete_row, row)
                    yield header, complete_row, complete_bytes
                    complete_row = None
                complete_row = _normalize_index_info_row(
                    row,
                    len(header),
                    row_number,
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
                        resource_policy=policy,
                        payload_lease=payload_lease,
                    )
                retain_rows(complete_row)
                yield header, complete_row, complete_bytes
                complete_row = None
        finally:
            header = pending_row = complete_row = row = record = suffix = None


def _normalize_index_info_row(
    row: list[str], column_count: int, row_number: int | None
) -> list[str]:
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
    payload_lease: MemoryLease,
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
    del window, value
    fixed = list(islice(record.iter_fields(), 5))
    codes = list(islice(record.iter_fields(), candidate_start, candidate_start + 5))
    continuation_spool.clear()
    sample_field_count = candidate_start - 5
    continuation_spool.append_fields(
        islice(record.iter_fields(), 5, candidate_start),
        field_count=sample_field_count,
        payload_lease=payload_lease,
    )
    sample = continuation_spool.joined(
        "",
        resource_policy=resource_policy,
        payload_lease=payload_lease,
    )
    continuation_spool.clear()
    formula_field_count = record.field_count - candidate_start - 5
    continuation_spool.append_fields(
        islice(record.iter_fields(), candidate_start + 5, None),
        field_count=formula_field_count,
        payload_lease=payload_lease,
    )
    formula = continuation_spool.joined(
        "",
        resource_policy=resource_policy,
        payload_lease=payload_lease,
    )
    continuation_spool.clear()
    return [*fixed, sample, *codes, formula]


def _grow_source_lease(
    *,
    additional_bytes: int,
    resource_policy: RuntimeResourcePolicy,
    payload_lease: MemoryLease,
) -> None:
    if additional_bytes > resource_policy.operation_headroom_bytes:
        raise DatasetReadCapacityExceeded(
            "csmar_raw_working_set_exceeds_task_memory_budget"
        )
    try:
        payload_lease.grow(additional_bytes)
    except MemoryError:
        raise DatasetReadCapacityExceeded(
            "csmar_raw_working_set_exceeds_task_memory_budget"
        ) from None


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
    producer_bound = max(1, resource_policy.operation_headroom_bytes // 2)
    batch_target = min(
        producer_bound, max(1, resource_policy.planning_headroom_bytes // 2)
    )
    pointer_bytes = struct.calcsize("P")

    if dataset == "index_info":
        rows = iter(
            iter_index_info_rows(
                export_path,
                resource_policy=resource_policy,
                expected_identity=expected_identity,
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
            )
        )
    with closing_iterator(rows):
        while True:
            lease = yield producer_bound
            pending: list[object] = []
            pending_bytes = 0
            pending_source_bytes = 0
            header: list[str] | None = None
            while len(pending) < chunksize:
                if pending and (
                    pending_bytes
                    + 2 * pending_source_bytes
                    + _raw_frame_owner_known_bytes(len(pending), header)
                    # Bound the next logical row's UCS4 payload and conversion buffers.
                    + 6 * MAX_LOGICAL_RECORD_BYTES
                    + len(header) * (80 + 16 + 2 * pointer_bytes)
                    + pointer_bytes
                    + 64 * 1024
                    > batch_target
                ):
                    break
                try:
                    next_header, row, row_bytes = next(rows)
                except StopIteration:
                    break
                if header is None:
                    header = next_header
                pending_bytes += sum(map(sys.getsizeof, row))
                pending_source_bytes += row_bytes
                if (
                    pending_bytes
                    + 2 * pending_source_bytes
                    + _raw_frame_owner_known_bytes(len(pending) + 1, header)
                    + 64 * 1024
                    > producer_bound
                ):
                    raise DatasetReadCapacityExceeded(
                        "csmar_raw_working_set_exceeds_task_memory_budget"
                    )
                pending.append(row)
            if not pending or header is None:
                return
            _require_raw_conversion_headroom(
                logical_bytes=pending_bytes
                + 2 * pending_source_bytes
                + _raw_frame_owner_known_bytes(len(pending), header),
                payload_lease=lease,
            )
            if dataset == "index_info":
                frame = _clean_index_info_source_codes(_raw_frame(pending, header))
            else:
                frame = _raw_frame(pending, header)
            frame_bytes = pandas_frame_payload_bytes(frame)
            column_label_bytes = sum(map(sys.getsizeof, frame.columns))
            actual_bound = 2 * (frame_bytes + column_label_bytes) + 128 * 1024
            if actual_bound > lease.reserved_bytes:
                raise DatasetReadCapacityExceeded(
                    "csmar_raw_working_set_exceeds_task_memory_budget"
                )
            row = None
            next_header = None
            header = None
            del pending
            lease.shrink_to(actual_bound)
            yield OwnedValue(frame, lease)
            del frame


def _iter_csv_records(
    path: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    require_exact_width: bool,
    dataset: str | None = None,
    expected_identity: DatasetFileIdentity | None = None,
) -> Iterator[tuple[list[str], list[str] | BoundedCsvRecord, int]]:
    rows = iter(
        iter_export_csv_records(
            path,
            resource_policy=resource_policy,
            expected_identity=expected_identity,
        )
    )
    with closing_iterator(rows):
        header = physical_header = row = record = None
        try:
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
                header = [
                    physical_header[index] for index in sorted(selected_positions)
                ]
                del physical_header
            del header_record
            with resource_policy.lease("source.csmar.csv.header", 0) as header_lease:
                try:
                    _grow_source_lease(
                        additional_bytes=sys.getsizeof(header)
                        + sum(map(sys.getsizeof, header)),
                        resource_policy=resource_policy,
                        payload_lease=header_lease,
                    )
                    while True:
                        try:
                            record, source_bytes = next(rows)
                        except StopIteration:
                            break
                        row: list[str] | BoundedCsvRecord = record
                        if require_exact_width:
                            if record.field_count != physical_width:
                                raise ValueError("malformed_csmar_csv_row")
                            row = record.materialize(
                                selected_positions=selected_positions
                            )
                            del record
                        yield header, row, source_bytes
                        del row
                        if not require_exact_width:
                            del record
                finally:
                    header = physical_header = row = record = None
        finally:
            header = physical_header = row = record = None


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


def _raw_frame_owner_known_bytes(row_count: int, header: list[str] | None) -> int:
    """Bound owner-known list, frame, and numeric decode buffers before allocation."""

    if not row_count or header is None:
        return 0
    pointer_bytes = struct.calcsize("P")
    cells = row_count * len(header)
    pending_backing_bytes = row_count * pointer_bytes + cells * pointer_bytes
    frame_reference_bytes = (len(header) + cells) * pointer_bytes
    numeric_decode_bytes = cells * 16
    return (
        pending_backing_bytes
        + frame_reference_bytes
        + numeric_decode_bytes
        + sum(map(sys.getsizeof, header))
    )


def _decoded_rows_slot_bytes(rows: Iterable[list[str] | None]) -> int:
    pointer_bytes = struct.calcsize("P")
    return sum((1 + len(row)) * pointer_bytes for row in rows if row is not None)


def _require_raw_conversion_headroom(
    *,
    logical_bytes: int,
    payload_lease: MemoryLease,
) -> None:
    available_bytes = payload_lease.reserved_bytes if payload_lease.active else 0
    if logical_bytes > available_bytes:
        raise DatasetReadCapacityExceeded(
            "csmar_raw_working_set_exceeds_task_memory_budget"
        )


def dataset_files_with_identity_by_id(
    raw_root: Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
    identity_index: ExactPrimaryKeyIndex | None = None,
    requested_datasets: Iterable[str] | None = None,
) -> Iterator[tuple[str, Path, DatasetFileIdentity]]:
    """Yield inspected files with the identity bound during inspection."""

    bound_policy = resource_policy
    root = Path(raw_root)
    requested = _requested_raw_datasets(requested_datasets)

    def iter_files() -> Iterator[tuple[str, Path, DatasetFileIdentity]]:
        policy = bound_policy or current_runtime_resource_policy()
        with runtime_resource_policy_scope(policy) as policy:
            with _raw_export_path_session(root) as (paths, verify_root):
                with ExitStack() as index_owner:
                    if identity_index is None:
                        index = ExactPrimaryKeyIndex(
                            primary_key=("dataset_path",),
                        )
                        index_owner.enter_context(
                            policy.lease(
                                "source.csmar.raw.identity.index",
                                index.retained_memory_bytes,
                            )
                        )
                        index_owner.enter_context(index)
                    else:
                        index = identity_index
                    files = _iter_indexed_raw_export_files(
                        root,
                        paths=paths,
                        index=index,
                        namespace="csmar-dataset-physical-exports",
                        resource_policy=policy,
                        requested_datasets=requested,
                    )
                    for path, identity in files:
                        inspection = inspect_raw_export_file(
                            path,
                            resource_policy=policy,
                            expected_identity=identity,
                        )
                        if inspection.missing_required_columns:
                            raise ValueError("csmar_raw_export_schema_invalid")
                        relative = path.relative_to(root).as_posix().encode("utf-8")
                        dataset = inspection.dataset.encode("utf-8")
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
    resource_policy: RuntimeResourcePolicy,
    requested_datasets: frozenset[str] | None,
) -> Iterator[tuple[Path, DatasetFileIdentity]]:
    path_namespace = f"{namespace}-paths"
    for path in paths:
        identity = raw_export_file_identity(path)
        if requested_datasets is not None:
            candidate = _raw_dataset_candidate(
                path,
                resource_policy=resource_policy,
                expected_identity=identity,
            )
            if candidate not in requested_datasets:
                continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = relative + b"\0" + _file_identity_bytes(identity)
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


def _requested_raw_datasets(
    requested_datasets: Iterable[str] | None,
) -> frozenset[str] | None:
    if requested_datasets is None:
        return None
    requested = frozenset(requested_datasets)
    if not requested.issubset(RAW_TABLE_CONTRACTS):
        raise ValueError("csmar_requested_dataset_invalid")
    return requested


def _raw_dataset_candidate(
    path: Path,
    *,
    resource_policy: RuntimeResourcePolicy,
    expected_identity: DatasetFileIdentity,
) -> str | None:
    try:
        return parse_raw_filename(path).dataset
    except ValueError:
        pass
    try:
        return infer_dataset_from_native_export(path)
    except ValueError:
        pass
    try:
        columns = read_export_header(
            path,
            resource_policy=resource_policy,
            expected_identity=expected_identity,
        )
    except ValueError as exc:
        if str(exc) == "csmar_raw_export_identity_changed":
            raise
        return None
    if all(
        missing_raw_column_names(columns, dataset) for dataset in RAW_TABLE_CONTRACTS
    ):
        return None
    return _infer_dataset_from_header(columns)


def _verify_csmar_member_set(
    root: Path,
    index: ExactPrimaryKeyIndex,
    *,
    namespace: str = "csmar-dataset-physical-exports-paths",
    requested_datasets: Iterable[str] | None = None,
) -> None:
    requested = _requested_raw_datasets(requested_datasets)
    policy = current_runtime_resource_policy()
    if requested is not None and policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    with _raw_export_path_session(root) as (paths, _verify_root):
        current_count = 0
        for path in paths:
            identity = raw_export_file_identity(path)
            if (
                requested is not None
                and _raw_dataset_candidate(
                    path,
                    resource_policy=policy,
                    expected_identity=identity,
                )
                not in requested
            ):
                continue
            relative = path.relative_to(root).as_posix().encode("utf-8")
            expected = index.identity_payload_value(
                namespace=namespace,
                payload=relative,
            )
            actual = (
                relative + b"\0" + _file_identity_bytes(raw_export_file_identity(path))
            )
            if expected != actual:
                raise ValueError("csmar_raw_export_identity_changed")
            current_count += 1
    if current_count != index.identity_payload_count(namespace=namespace):
        raise ValueError("csmar_raw_export_identity_changed")


def _raw_export_physical_key(identity: DatasetFileIdentity) -> bytes:
    return b"\0".join(
        str(value).encode("ascii") for value in (identity.device, identity.inode)
    )
