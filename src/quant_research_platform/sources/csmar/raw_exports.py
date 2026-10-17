"""CSMAR raw export inspection and source-local IO helpers."""

from __future__ import annotations

import re
import stat
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

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
    rows = _iter_csv_records(
        export_path,
        dataset=dataset,
        expected_identity=expected_identity,
        content_hashes=content_hashes,
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
    dataset: str,
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
        from .field_mappings import (
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
        selected_columns = set(COLUMN_MAPPINGS[dataset])
        selected_columns.update(RAW_TABLE_CONTRACTS[dataset].required_columns)
        selected_columns.update(
            alias
            for aliases in COLUMN_ALIASES.get(dataset, {}).values()
            for alias in aliases
        )
        selected_positions = [
            index
            for index, column in enumerate(physical_header)
            if column in selected_columns
        ]
        header = [physical_header[index] for index in selected_positions]
        del physical_header
        del header_record
        yield header, None
        while True:
            try:
                record = next(rows)
            except StopIteration:
                break
            if len(record) != physical_width:
                raise ValueError("malformed_csmar_csv_row")
            row = [record[position] for position in selected_positions]
            del record
            _yield_value = (header, row)
            try:
                yield _yield_value
            except GeneratorExit:
                return
            del row


def _raw_frame(
    rows: list[list[str]],
    header: list[str],
    *,
    dataset: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=header)
    frame.replace("", pd.NA, inplace=True)
    # Raw IO owns readable structure, never business numeric interpretation.
    frame = frame.astype("string")
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
