"""CSMAR raw export inspection and source-local IO helpers."""

from __future__ import annotations

import re
import stat
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    filesystem_stat,
    filesystem_walk,
    require_unaliased_path,
)

from .field_mappings import RAW_TABLE_CONTRACTS, missing_raw_column_names
from .raw_schemas import (
    _infer_dataset_from_header,
    infer_dataset_from_native_export,
    iter_export_csv_records,
    parse_raw_filename,
    raw_export_file_identity,
    read_export_header,
    validate_export_header,
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
    "DataSources": str,
    "SubjectCode": str,
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


def _iter_raw_export_paths(root: Path) -> Iterator[Path]:
    root = require_unaliased_path(Path(root), error="csmar_raw_export_identity_changed")
    metadata = filesystem_stat(root, follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("csmar_raw_export_identity_changed")

    def fail_walk(error: OSError) -> None:
        if isinstance(error, (FileNotFoundError, NotADirectoryError)):
            raise ValueError("csmar_raw_export_identity_changed") from error
        raise error

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
        dirnames[:] = sorted(name for name in dirnames if name.casefold() != "raw_only")
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


def iter_index_info_rows(
    export_path: Path,
    *,
    expected_identity: DatasetFileIdentity | None,
    content_hashes: list[str] | None = None,
) -> Iterator[tuple[list[str], list[str] | None]]:
    rows = iter(
        _iter_csv_records(
            export_path,
            dataset="index_info",
            require_exact_width=False,
            expected_identity=expected_identity,
            content_hashes=content_hashes,
        )
    )
    continuations: list[str] = []
    with closing_iterator(rows):
        pending_row: list[str] | None = None
        pending_start: int | None = None
        complete_row: list[str] | None = None
        row_number = 1
        header: list[str] | None = None
        while True:
            try:
                header, record = next(rows)
            except StopIteration:
                break
            if record is None:
                yield header, None
                continue
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
                yield header, record
                continue
            if pending_row is not None:
                if _record_starts_index_info(record):
                    raise ValueError(f"malformed_csmar_csv_row: {pending_start}")
                remaining_columns = len(header) - len(pending_row)
                completes_row = (
                    len(pending_row) <= 6 and len(record) >= remaining_columns
                )
                suffix = _append_continuation_fields(
                    continuations,
                    iter(record),
                    field_count=len(record),
                    suffix_count=remaining_columns if completes_row else 0,
                    include_empty=False,
                )
                if completes_row:
                    pending_row[-1] = _join_continuations(
                        pending_row[-1], continuations
                    )
                    pending_row.extend(suffix)
                    complete_row = _normalize_index_info_row(
                        pending_row, len(header), pending_start
                    )
                    continuations.clear()
                    pending_row = None
                    pending_start = None
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
                        _yield_value = (header, complete_row)
                        try:
                            yield _yield_value
                        except GeneratorExit:
                            return
                        complete_row = None
                    pending_row = row
                    pending_start = row_number
                    continuations.clear()
                    complete_row = None
                    del row
                    continue
                if complete_row is not None:
                    _append_continuation_fields(
                        continuations, row, field_count=len(row)
                    )
                    del row
                    continue
                raise ValueError(f"malformed_csmar_csv_row: {row_number}")
            if complete_row is not None:
                if continuations:
                    complete_row[-1] = _join_continuations(
                        complete_row[-1], continuations
                    )
                _yield_value = (header, complete_row)
                try:
                    yield _yield_value
                except GeneratorExit:
                    return
                complete_row = None
            complete_row = _normalize_index_info_row(row, len(header), row_number)
            del row
            continuations.clear()
            del record
        if pending_row is not None:
            raise ValueError(f"malformed_csmar_csv_row: {pending_start}")
        if complete_row is not None:
            if continuations:
                complete_row[-1] = _join_continuations(complete_row[-1], continuations)
            _yield_value = (header, complete_row)
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


def iter_private_export_chunks(
    path: str | Path,
    *,
    dataset: str,
    chunksize: int,
    expected_identity: DatasetFileIdentity | None = None,
    content_hashes: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Iterator[pd.DataFrame]:
    if type(dataset) is not str or dataset not in RAW_TABLE_CONTRACTS:
        raise ValueError("unknown_csmar_dataset")
    if type(chunksize) is not int or chunksize <= 0:
        raise ValueError("invalid_csmar_chunksize")
    export_path = Path(path)
    rows = (
        iter_index_info_rows(
            export_path,
            expected_identity=expected_identity,
            content_hashes=content_hashes,
        )
        if dataset == "index_info"
        else _iter_csv_records(
            export_path,
            dataset=dataset,
            require_exact_width=True,
            expected_identity=expected_identity,
            content_hashes=content_hashes,
        )
    )
    emitted = False
    with closing_iterator(rows):
        header = None
        while True:
            pending = []
            while len(pending) < chunksize:
                try:
                    header, row = next(rows)
                except StopIteration:
                    break
                if row is not None:
                    pending.append(row)
            if not pending:
                if not emitted:
                    if header is None:
                        raise ValueError("malformed_csmar_csv_row")
                    yield _raw_frame([], header, dataset=dataset)
                return
            if dataset == "index_info":
                from .field_mappings import COLUMN_MAPPINGS, validate_raw_column_names

                validate_raw_column_names(tuple(header), dataset)
                positions = [
                    i
                    for i, name in enumerate(header)
                    if name in COLUMN_MAPPINGS[dataset]
                ]
                header = [header[i] for i in positions]
                pending = [[row[i] for i in positions] for row in pending]
            emitted = True
            frame = _raw_frame(
                pending,
                header,
                dataset=dataset,
                start_date=start_date,
                end_date=end_date,
            )
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
    content_hashes: list[str] | None = None,
) -> Iterator[tuple[list[str], list[str] | None]]:
    rows = iter(
        iter_export_csv_records(
            path, expected_identity=expected_identity, content_hashes=content_hashes
        )
    )
    with closing_iterator(rows):
        header = physical_header = row = record = None
        header_record = next(rows, None)
        if header_record is None:
            return
        physical_header = validate_export_header(header_record)
        physical_width = len(physical_header)
        selected_positions: frozenset[int] | None = None
        header = physical_header
        if dataset is not None:
            from quant_research_platform.sources.csmar.field_mappings import (
                COLUMN_ALIASES,
                COLUMN_MAPPINGS,
                validate_raw_column_names,
            )

            matches = [
                name
                for name in RAW_TABLE_CONTRACTS
                if not missing_raw_column_names(tuple(physical_header), name)
            ]
            if matches and dataset not in matches:
                raise ValueError("csmar_raw_export_table_identity_conflict")
            validate_raw_column_names(tuple(physical_header), dataset)
        if require_exact_width and dataset is not None:
            selected_columns = set(COLUMN_MAPPINGS[dataset])
            selected_columns.update(RAW_TABLE_CONTRACTS[dataset].required_columns)
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
        yield header, None
        while True:
            try:
                record = next(rows)
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
            _yield_value = (header, row)
            try:
                yield _yield_value
            except GeneratorExit:
                return
            del row
            if not require_exact_width:
                del record


def _raw_frame(
    rows: list[list[str]],
    header: list[str],
    *,
    dataset: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    from .field_mappings import COLUMN_ALIASES, COLUMN_MAPPINGS

    frame = pd.DataFrame(rows, columns=header)
    frame.replace("", pd.NA, inplace=True)
    date_column = {
        "market_bars_1d": "Trddt",
        "adjustment": "TradingDate",
        "market_cap": "Shrchgdt",
    }.get(dataset)
    if date_column is not None and (start_date is not None or end_date is not None):
        date_column = next(
            (
                column
                for column in (
                    date_column,
                    *COLUMN_ALIASES.get(dataset, {}).get(date_column, ()),
                )
                if column in frame
            ),
            date_column,
        )
        if date_column not in frame:
            missing = missing_raw_column_names(tuple(header), dataset)
            raise ValueError(
                f"{dataset} raw export is missing CSMAR columns: {missing}"
            )
        timestamps = strict_timestamp_series(
            frame[date_column], error=f"invalid_{dataset}_source_timestamp"
        )
        mask = pd.Series(True, index=frame.index)
        if start_date is not None and dataset != "market_cap":
            mask &= timestamps >= pd.Timestamp(start_date)
        if end_date is not None:
            mask &= timestamps <= pd.Timestamp(end_date)
        frame = frame.loc[mask].copy()
    consumed = set(COLUMN_MAPPINGS[dataset])
    consumed.update(
        alias
        for aliases in COLUMN_ALIASES.get(dataset, {}).values()
        for alias in aliases
    )
    for column in frame.columns:
        if (
            dataset == "corporate_actions"
            or column not in consumed
            or column in STRING_CODE_DTYPES
            or column in INDEX_INFO_SOURCE_CODE_COLUMNS
        ):
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
    requested_datasets: Iterable[str] | None = None,
) -> Iterator[tuple[str, Path, DatasetFileIdentity]]:
    """Select source members; validate schema during actual CSV consumption."""
    root = Path(raw_root)
    requested = _requested_raw_datasets(requested_datasets)
    records, physical_members = [], set()
    with closing_iterator(_iter_raw_export_paths(root)) as paths:
        for path in paths:
            identity = raw_export_file_identity(path)
            dataset = _raw_dataset_candidate(path, expected_identity=identity)
            if dataset is None or requested is not None and dataset not in requested:
                continue
            physical_key = (identity.device, identity.inode)
            if physical_key in physical_members:
                raise ValueError("csmar_raw_export_identity_changed")
            physical_members.add(physical_key)
            records.append((dataset, path, identity))
    yield from sorted(
        records, key=lambda item: (item[0], item[1].relative_to(root).as_posix())
    )


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
