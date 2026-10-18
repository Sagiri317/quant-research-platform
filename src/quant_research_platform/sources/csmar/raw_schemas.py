"""Internal provider raw-file parsing helpers."""

from __future__ import annotations

import csv
import io
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from quant_research_platform.sources.csmar import field_mappings as _field_mappings
from quant_research_platform.storage.identity import (
    _DigestingReader,
    open_filesystem_file,
)


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
    """Infer the dataset only from an explicit native CSMAR table file name."""

    export_path = Path(path)
    raw_only = re.fullmatch(
        r"raw__(?P<table>[A-Za-z0-9_]+)__part-\d+__observed-\d{8}\.csv",
        export_path.name,
    )
    stem = (
        raw_only.group("table")
        if raw_only is not None
        else re.sub(r"\d+$", "", export_path.stem)
    )
    for dataset, contract in _field_mappings.RAW_TABLE_CONTRACTS.items():
        if contract.table_code == stem:
            return dataset

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


class _CsvRecords:
    """Parse strict provider CSV records and hash the consumed bytes."""

    def __init__(self, path, *, content_hashes=None):
        self._handle = open_filesystem_file(path, "rb")
        try:
            self._content_hashes = content_hashes
            self._exhausted = False
            self._bytes = _DigestingReader(self._handle)
            self._text = io.TextIOWrapper(self._bytes, encoding="utf-8-sig", newline="")
            # The provider format permits fields above Python's default limit.
            csv.field_size_limit(sys.maxsize)
            self._reader = csv.reader(self._text, strict=True)
        except BaseException as primary:
            try:
                self._handle.close()
            except BaseException as error:
                primary.add_note(
                    "CSMAR source close also failed: " + type(error).__name__
                )
            raise

    def __iter__(self):
        return self

    def __next__(self):
        try:
            fields = next(self._reader)
        except StopIteration:
            self._exhausted = True
            raise
        except (UnicodeError, csv.Error) as exc:
            raise ValueError("malformed_csmar_csv_row") from exc
        return fields

    def close(self):
        self._text.close()
        if self._exhausted and self._content_hashes is not None:
            self._content_hashes.append(self._bytes.hexdigest())


def iter_export_csv_records(
    path: str | Path,
    *,
    content_hashes: list[str] | None = None,
) -> Iterator[list[str]]:

    def activate() -> Iterator[list[str]]:
        records: _CsvRecords | None = None
        primary = None
        try:
            records = _CsvRecords(
                Path(path),
                content_hashes=content_hashes,
            )
            for record in records:
                try:
                    yield record
                except GeneratorExit:
                    return
                del record
        except BaseException as body_error:
            primary = body_error
            raise
        finally:
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


def read_export_header(path: str | Path) -> tuple[str, ...]:
    """Read the header row from a supported CSV raw export."""

    export_path = Path(path)
    suffix = export_path.suffix.lower()
    if suffix != ".csv":
        raise ValueError(f"Unsupported CSMAR export extension: {suffix}")
    rows = iter_export_csv_records(export_path)
    primary = None
    try:
        record = next(rows, None)
        columns = tuple(validate_export_header(record)) if record is not None else ()
        if not columns:
            raise ValueError("malformed_csmar_csv_row")
    except BaseException as body_error:
        primary = body_error
        raise
    finally:
        try:
            rows.close()
        except BaseException as close_error:
            if primary is None:
                raise
            primary.add_note(
                f"CSMAR CSV row cleanup also failed: {type(close_error).__name__}"
            )
    return columns


def validate_export_header(record: list[str]) -> list[str]:
    """Materialize a legal header while preserving duplicate-header errors."""

    columns = record
    if len(columns) != len(set(columns)):
        raise ValueError("duplicate_csmar_provider_column")
    return columns
