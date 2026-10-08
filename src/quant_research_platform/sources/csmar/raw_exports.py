"""CSMAR raw export inspection and source-local IO helpers."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from itertools import groupby, islice
from pathlib import Path

import pandas as pd

from quant_research_platform.policy.lifecycle import closing_iterator
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
    _infer_dataset_from_header,
    infer_dataset_from_native_export,
    inspect_raw_export_file,
    iter_export_csv_records,
    materialize_header_record,
    parse_raw_filename,
    raw_export_file_identity,
    read_export_header,
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
INDEX_INFO_CODE_RE = re.compile("^[0-9A-Za-z_.-]{1,32}$")
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


def _append_continuation_fields(
    continuations: list[str],
    fields: Iterable[str],
    *,
    field_count: int,
    suffix_count: int = 0,
    include_empty: bool = True,
) -> list[str]:
    values = list(fields)
    prefix_count = field_count - suffix_count
    if prefix_count < 0:
        raise ValueError("malformed_csmar_csv_row")
    prefix = ",".join(values[:prefix_count])
    if prefix or include_empty:
        continuations.append(prefix)
    return values[prefix_count:]


def _join_continuations(current: str, continuations: list[str]) -> str:
    parts = [current] if current else []
    for value in continuations:
        if parts or value:
            parts.append(value)
    return "\n".join(parts)


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
        walker = iter(filesystem_walk(root, follow_symlinks=False, on_error=fail_walk))
        while True:
            try:
                parent, dirnames, filenames = next(walker)
            except StopIteration:
                return
            require_unaliased_path(parent, error="csmar_raw_export_identity_changed")
            for name in dirnames:
                require_unaliased_path(
                    parent / name, error="csmar_raw_export_identity_changed"
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

    primary = None
    try:
        require_unaliased_path(root, error="csmar_raw_export_identity_changed")
        yield (iter_paths(), verify_root)
    except BaseException as body_error:
        primary = body_error
        raise
    finally:
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
    export_path: Path, *, expected_identity: DatasetFileIdentity | None
) -> Iterator[tuple[list[str], list[str], int]]:
    rows = iter(
        _iter_csv_records(
            export_path, require_exact_width=False, expected_identity=expected_identity
        )
    )
    continuations: list[str] = []
    with closing_iterator(rows):
        pending_row: list[str] | None = None
        pending_start: int | None = None
        pending_bytes = 0
        complete_row: list[str] | None = None
        complete_bytes = 0
        row_number = 1
        header: list[str] | None = None
        while True:
            try:
                header, record, source_bytes = next(rows)
            except StopIteration:
                break
            row_number += 1
            if header != [
                "Indexcd",
                "Idxinfo01",
                "Idxinfo11",
                "Idxinfo02",
                "Idxinfo03",
                "Idxinfo04",
                "Idxinfo05",
                "Idxinfo06",
                "Idxinfo07",
                "Idxinfo08",
                "Idxinfo09",
                "Idxinfo10",
            ]:
                if (
                    len(record) != len(header)
                    or "Indexcd" not in header
                    or not INDEX_INFO_CODE_RE.fullmatch(record[header.index("Indexcd")])
                ):
                    raise ValueError(f"malformed_csmar_csv_row: {row_number}")
                yield header, record, source_bytes
                continue
            if pending_row is not None:
                if _record_starts_index_info(record):
                    raise ValueError(f"malformed_csmar_csv_row: {pending_start}")
                remaining_columns = len(header) - len(pending_row)
                completes_row = (
                    len(pending_row) <= 6 and len(record) >= remaining_columns
                )
                next_bytes = pending_bytes + source_bytes
                suffix = _append_continuation_fields(
                    continuations,
                    iter(record),
                    field_count=len(record),
                    suffix_count=remaining_columns if completes_row else 0,
                    include_empty=False,
                )
                pending_bytes = next_bytes
                if completes_row:
                    pending_row[-1] = _join_continuations(
                        pending_row[-1], continuations
                    )
                    pending_row.extend(suffix)
                    complete_row = _normalize_index_info_row(
                        pending_row, len(header), pending_start
                    )
                    complete_bytes = pending_bytes
                    continuations.clear()
                    pending_row = None
                    pending_start = None
                    pending_bytes = 0
                del record, suffix
                continue
            if len(record) > len(header):
                if _record_starts_index_info(record):
                    row = _normalize_index_info_record(
                        record, row_number, continuation_spool=continuations
                    )
                elif complete_row is not None:
                    _append_continuation_fields(
                        continuations, iter(record), field_count=len(record)
                    )
                    complete_bytes += source_bytes
                    del record
                    continue
                else:
                    raise ValueError(f"malformed_csmar_csv_row: {row_number}")
            else:
                row = record
            if len(row) < len(header):
                if _record_starts_index_info(row):
                    if complete_row is not None:
                        if continuations:
                            complete_row[-1] = _join_continuations(
                                complete_row[-1], continuations
                            )
                        _yield_value = (header, complete_row, complete_bytes)
                        try:
                            yield _yield_value
                        except GeneratorExit:
                            return
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
                    _append_continuation_fields(
                        continuations, row, field_count=len(row)
                    )
                    complete_bytes += source_bytes
                    del row
                    continue
                raise ValueError(f"malformed_csmar_csv_row: {row_number}")
            if complete_row is not None:
                if continuations:
                    complete_row[-1] = _join_continuations(
                        complete_row[-1], continuations
                    )
                _yield_value = (header, complete_row, complete_bytes)
                try:
                    yield _yield_value
                except GeneratorExit:
                    return
                complete_row = None
            complete_row = _normalize_index_info_row(row, len(header), row_number)
            del row
            continuations.clear()
            complete_bytes = source_bytes
            del record
        if pending_row is not None:
            raise ValueError(f"malformed_csmar_csv_row: {pending_start}")
        if complete_row is not None:
            if continuations:
                complete_row[-1] = _join_continuations(complete_row[-1], continuations)
            _yield_value = (header, complete_row, complete_bytes)
            try:
                yield _yield_value
            except GeneratorExit:
                return
            complete_row = None


def _normalize_index_info_row(
    row: list[str], column_count: int, row_number: int | None
) -> list[str]:
    if len(row) != column_count or not _record_starts_index_info(row):
        raise ValueError(f"malformed_csmar_csv_row: {row_number}")
    return row


def _record_starts_index_info(record: list[str]) -> bool:
    first = next(iter(record), "")
    return bool(INDEX_INFO_CODE_RE.fullmatch(first))


def _normalize_index_info_record(
    record: list[str], row_number: int | None, *, continuation_spool: list[str]
) -> list[str]:
    """Normalize an over-width index record while preserving text continuations."""
    window: list[str] = []
    candidate_start: int | None = None
    for position, value in enumerate(iter(record)):
        window.append(value)
        if len(window) > 5:
            del window[0]
        start = position - 4
        if start < 6 or len(window) != 5 or start + 5 >= len(record):
            continue
        if all(
            (
                _looks_like_source_category_code(item, allowed_values)
                for item, allowed_values in zip(
                    window, INDEX_INFO_SOURCE_CODE_ENUMS, strict=True
                )
            )
        ):
            if candidate_start is not None:
                raise ValueError(f"malformed_csmar_csv_row: {row_number}")
            candidate_start = start
    if candidate_start is None or candidate_start + 5 >= len(record):
        raise ValueError(f"malformed_csmar_csv_row: {row_number}")
    del window, value
    fixed = list(islice(iter(record), 5))
    codes = list(islice(iter(record), candidate_start, candidate_start + 5))
    continuation_spool.clear()
    sample_field_count = candidate_start - 5
    _append_continuation_fields(
        continuation_spool,
        islice(iter(record), 5, candidate_start),
        field_count=sample_field_count,
    )
    sample = _join_continuations("", continuation_spool)
    continuation_spool.clear()
    formula_field_count = len(record) - candidate_start - 5
    _append_continuation_fields(
        continuation_spool,
        islice(iter(record), candidate_start + 5, None),
        field_count=formula_field_count,
    )
    formula = _join_continuations("", continuation_spool)
    continuation_spool.clear()
    return [*fixed, sample, *codes, formula]


def _looks_like_source_category_code(
    value: str, allowed_values: frozenset[str]
) -> bool:
    stripped = value.strip()
    return stripped == "" or stripped in allowed_values


def _clean_index_info_source_codes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame
    for column, allowed_values in zip(
        INDEX_INFO_SOURCE_CODE_COLUMNS, INDEX_INFO_SOURCE_CODE_ENUMS, strict=True
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
    expected_identity: DatasetFileIdentity | None = None,
) -> Iterator[pd.DataFrame]:
    if type(dataset) is not str or dataset not in RAW_TABLE_CONTRACTS:
        raise ValueError("unknown_csmar_dataset")
    if type(chunksize) is not int or chunksize <= 0:
        raise ValueError("invalid_csmar_chunksize")
    export_path = Path(path)
    rows = (
        iter_index_info_rows(export_path, expected_identity=expected_identity)
        if dataset == "index_info"
        else _iter_csv_records(
            export_path,
            dataset=dataset,
            require_exact_width=True,
            expected_identity=expected_identity,
        )
    )
    with closing_iterator(rows):
        while True:
            pending = []
            header = None
            while len(pending) < chunksize:
                try:
                    header, row, _row_bytes = next(rows)
                except StopIteration:
                    break
                pending.append(row)
            if not pending:
                return
            frame = _raw_frame(pending, header)
            if dataset == "index_info":
                frame = _clean_index_info_source_codes(frame)
            try:
                yield frame
            except GeneratorExit:
                return


def _iter_csv_records(
    path: Path,
    *,
    require_exact_width: bool,
    dataset: str | None = None,
    expected_identity: DatasetFileIdentity | None = None,
) -> Iterator[tuple[list[str], list[str], int]]:
    rows = iter(iter_export_csv_records(path, expected_identity=expected_identity))
    with closing_iterator(rows):
        header = physical_header = row = record = None
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
                (
                    index
                    for index, column in enumerate(physical_header)
                    if column in selected_columns
                )
            )
            header = [physical_header[index] for index in sorted(selected_positions)]
            del physical_header
        del header_record
        while True:
            try:
                record, source_bytes = next(rows)
            except StopIteration:
                break
            row: list[str] = record
            if require_exact_width:
                if len(record) != physical_width:
                    raise ValueError("malformed_csmar_csv_row")
                row = [
                    value
                    for position, value in enumerate(record)
                    if position in selected_positions
                ]
                del record
            _yield_value = (header, row, source_bytes)
            try:
                yield _yield_value
            except GeneratorExit:
                return
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


def dataset_files_with_identity_by_id(
    raw_root: Path,
    *,
    identity_index: ExactPrimaryKeyIndex | None = None,
    requested_datasets: Iterable[str] | None = None,
) -> Iterator[tuple[str, Path, DatasetFileIdentity]]:
    """Yield inspected files with the identity bound during inspection."""
    root = Path(raw_root)
    requested = _requested_raw_datasets(requested_datasets)

    def iter_files() -> Iterator[tuple[str, Path, DatasetFileIdentity]]:
        with _raw_export_path_session(root) as (paths, verify_root):
            if identity_index is None:
                index = ExactPrimaryKeyIndex(primary_key=("dataset_path",))
            else:
                index = identity_index
            files = _iter_indexed_raw_export_files(
                root,
                paths=paths,
                index=index,
                namespace="csmar-dataset-physical-exports",
                requested_datasets=requested,
            )
            for path, identity in files:
                inspection = inspect_raw_export_file(path, expected_identity=identity)
                if inspection.missing_required_columns:
                    raise ValueError("csmar_raw_export_schema_invalid")
                relative = path.relative_to(root).as_posix().encode("utf-8")
                dataset = inspection.dataset.encode("utf-8")
                identity_payload = b"\x00".join(
                    str(value).encode("ascii")
                    for value in (
                        identity.device,
                        identity.inode,
                        identity.size,
                        identity.modified_ns,
                    )
                )
                payload = dataset + b"\x00" + relative + b"\x00" + identity_payload
                index.add_identity_payloads(
                    namespace="csmar-dataset-paths",
                    payloads=((dataset + b"\x00" + relative, payload),),
                )
            with closing_iterator(
                _iter_dataset_file_identities(root, index)
            ) as records:
                for _, group in groupby(records, key=lambda item: item[0]):
                    for _yield_value in group:
                        try:
                            yield _yield_value
                        except GeneratorExit:
                            return
                    verify_root()

    return iter_files()


def _iter_dataset_file_identities(
    root: Path, index: ExactPrimaryKeyIndex, *, data_id: str | None = None
) -> Iterator[tuple[str, Path, DatasetFileIdentity]]:
    """Replay the inspected inventory without retaining a second member list."""
    with closing_iterator(
        index.identity_payload_values(namespace="csmar-dataset-paths")
    ) as payloads:
        for payload in payloads:
            dataset, relative, device, inode, size, modified_ns = payload.split(b"\x00")
            dataset = dataset.decode("utf-8")
            if data_id is None or dataset == data_id:
                _yield_value = (
                    dataset,
                    root / relative.decode("utf-8"),
                    DatasetFileIdentity(
                        device=int(device),
                        inode=int(inode),
                        size=int(size),
                        modified_ns=int(modified_ns),
                    ),
                )
                try:
                    yield _yield_value
                except GeneratorExit:
                    return


def _iter_indexed_raw_export_files(
    root: Path,
    *,
    paths: Iterator[Path],
    index: ExactPrimaryKeyIndex,
    namespace: str,
    requested_datasets: frozenset[str] | None,
) -> Iterator[tuple[Path, DatasetFileIdentity]]:
    path_namespace = f"{namespace}-paths"
    for path in paths:
        identity = raw_export_file_identity(path)
        if requested_datasets is not None:
            candidate = _raw_dataset_candidate(path, expected_identity=identity)
            if candidate not in requested_datasets:
                continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = relative + b"\x00" + _file_identity_bytes(identity)
        inserted = index.add_identity_payloads(
            namespace=namespace,
            payloads=((_raw_export_physical_key(identity), payload),),
        )
        if inserted != 1:
            raise ValueError("csmar_raw_export_identity_changed")
        index.add_identity_payloads(
            namespace=path_namespace, payloads=((relative, payload),)
        )
    payloads = iter(index.identity_payload_values(namespace=path_namespace))
    while True:
        try:
            payload = next(payloads)
        except StopIteration:
            return
        relative, device, inode, size, modified_ns = payload.split(b"\x00")
        path = root / relative.decode("utf-8")
        identity = DatasetFileIdentity(
            device=int(device),
            inode=int(inode),
            size=int(size),
            modified_ns=int(modified_ns),
        )
        yield (path, identity)


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
    path: Path, *, expected_identity: DatasetFileIdentity
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
        columns = read_export_header(path, expected_identity=expected_identity)
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
    with _raw_export_path_session(root) as (paths, _verify_root):
        current_count = 0
        for path in paths:
            identity = raw_export_file_identity(path)
            if (
                requested is not None
                and _raw_dataset_candidate(path, expected_identity=identity)
                not in requested
            ):
                continue
            relative = path.relative_to(root).as_posix().encode("utf-8")
            expected = index.identity_payload_value(
                namespace=namespace, payload=relative
            )
            actual = (
                relative
                + b"\x00"
                + _file_identity_bytes(raw_export_file_identity(path))
            )
            if expected != actual:
                raise ValueError("csmar_raw_export_identity_changed")
            current_count += 1
    if current_count != index.identity_payload_count(namespace=namespace):
        raise ValueError("csmar_raw_export_identity_changed")


def _raw_export_physical_key(identity: DatasetFileIdentity) -> bytes:
    return b"\x00".join(
        str(value).encode("ascii") for value in (identity.device, identity.inode)
    )
