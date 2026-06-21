"""CSMAR raw export inspection and source-local IO helpers."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .raw_schemas import RawExportInspection, inspect_raw_export_file

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


@dataclass(frozen=True)
class RawExportIntakeResult:
    """Record one raw export inspection result."""

    path: Path
    inspection: RawExportInspection | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Return whether inspection succeeded."""

        return self.error is None and self.inspection is not None


def is_raw_export_file(path: Path) -> bool:
    """Return whether the path looks like a supported raw export file."""

    return (
        path.is_file()
        and path.suffix.lower() in SUPPORTED_RAW_EXPORT_SUFFIXES
        and "[DES]" not in path.name
        and "__description__" not in path.name
        and ".metadata." not in path.name
    )


def iter_raw_export_files(root: Path) -> list[Path]:
    """Return supported raw export files under a private raw directory."""

    return sorted(path for path in root.rglob("*") if is_raw_export_file(path))


def inspect_raw_export(path: Path) -> RawExportIntakeResult:
    """Inspect one raw export without writing metadata sidecars."""

    try:
        inspection = inspect_raw_export_file(path)
    except Exception as exc:  # noqa: BLE001
        return RawExportIntakeResult(path=path, error=str(exc))

    return RawExportIntakeResult(path=path, inspection=inspection)


def read_private_export_file(path: str | Path, dataset: str) -> pd.DataFrame:
    """Read one private CSMAR CSV export and handle known CSV edge cases."""

    export_path = Path(path)
    if dataset != "index_info":
        return pd.read_csv(export_path, dtype=STRING_CODE_DTYPES)

    with export_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows: list[list[str]] = []
        pending_row: list[str] | None = None
        pending_start: int | None = None
        for row_number, row in enumerate(reader, start=2):
            if pending_row is not None:
                if _looks_like_index_info_record_start(row):
                    raise ValueError(f"malformed_csmar_csv_row: {pending_start}")
                pending_row = _merge_index_info_continuation(
                    pending_row,
                    row,
                    len(header),
                )
                if len(pending_row) >= len(header):
                    rows.append(
                        _normalize_index_info_row(
                            pending_row,
                            len(header),
                            pending_start,
                        )
                    )
                    pending_row = None
                    pending_start = None
                continue

            if len(row) < len(header):
                if _looks_like_index_info_record_start(row):
                    pending_row = list(row)
                    pending_start = row_number
                    continue
                if rows:
                    rows[-1][-1] = _append_text_continuation(rows[-1][-1], row)
                    continue
                raise ValueError(f"malformed_csmar_csv_row: {row_number}")
            rows.append(_normalize_index_info_row(row, len(header), row_number))
        if pending_row is not None:
            raise ValueError(f"malformed_csmar_csv_row: {pending_start}")
    frame = pd.DataFrame(rows, columns=header).replace("", pd.NA)
    return _clean_index_info_source_codes(frame)


def _normalize_index_info_row(
    row: list[str],
    column_count: int,
    row_number: int | None,
) -> list[str]:
    if len(row) > column_count:
        row = _normalize_long_index_info_row(row, row_number)
    if len(row) != column_count or not _looks_like_index_info_record_start(row):
        raise ValueError(f"malformed_csmar_csv_row: {row_number}")
    return row


def _normalize_long_index_info_row(
    row: list[str],
    row_number: int | None,
) -> list[str]:
    candidates: list[list[str]] = []
    for code_start in range(6, len(row) - 5):
        codes = row[code_start : code_start + 5]
        if not all(
            _looks_like_source_category_code(value, allowed_values)
            for value, allowed_values in zip(
                codes,
                INDEX_INFO_SOURCE_CODE_ENUMS,
                strict=True,
            )
        ):
            continue
        sample_scope = ",".join(row[5:code_start])
        formula = ",".join(row[code_start + 5 :])
        candidates.append([*row[:5], sample_scope, *codes, formula])
    if len(candidates) != 1:
        raise ValueError(f"malformed_csmar_csv_row: {row_number}")
    return candidates[0]


def _merge_index_info_continuation(
    pending_row: list[str],
    row: list[str],
    column_count: int,
) -> list[str]:
    merged = list(pending_row)
    if len(merged) <= 6:
        remaining_columns = column_count - len(merged)
        if len(row) >= remaining_columns:
            text_fields = row[: len(row) - remaining_columns]
            if text_fields:
                merged[-1] = _append_text_continuation(merged[-1], text_fields)
            merged.extend(row[len(row) - remaining_columns :])
            return merged
        merged[-1] = _append_text_continuation(merged[-1], row)
        return merged
    merged[-1] = _append_text_continuation(merged[-1], row)
    return merged


def _append_text_continuation(current: str, row: list[str]) -> str:
    continuation = ",".join(row)
    return f"{current}\n{continuation}" if current else continuation


def _looks_like_index_info_record_start(row: list[str]) -> bool:
    return bool(row) and bool(INDEX_INFO_CODE_RE.fullmatch(row[0]))


def _looks_like_source_category_code(
    value: str,
    allowed_values: frozenset[str],
) -> bool:
    stripped = value.strip()
    return stripped == "" or stripped in allowed_values


def _clean_index_info_source_codes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
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

    export_path = Path(path)
    if dataset == "index_info":
        yield read_private_export_file(export_path, dataset)
        return
    yield from pd.read_csv(export_path, chunksize=chunksize, dtype=STRING_CODE_DTYPES)


def find_dataset_files(raw_root: Path, dataset: str) -> list[Path]:
    """Find raw CSMAR CSV files for one normalized dataset id."""

    return list(dataset_files_by_id(raw_root).get(dataset, ()))


def dataset_files_by_id(raw_root: Path) -> dict[str, tuple[Path, ...]]:
    """Group raw CSMAR CSV files by inspected normalized dataset id."""

    matches: dict[str, list[Path]] = {}
    for result in inspect_raw_export_tree(raw_root):
        if not result.ok or result.inspection is None:
            continue
        matches.setdefault(result.inspection.dataset, []).append(result.path)
    return {
        data_id: tuple(sorted(paths))
        for data_id, paths in sorted(matches.items())
    }


def inspect_raw_export_tree(root: Path) -> list[RawExportIntakeResult]:
    """Inspect supported raw exports under a private raw directory."""

    return [inspect_raw_export(path) for path in iter_raw_export_files(root)]


__all__ = [
    "RawExportIntakeResult",
    "STRING_CODE_DTYPES",
    "SUPPORTED_RAW_EXPORT_SUFFIXES",
    "dataset_files_by_id",
    "find_dataset_files",
    "inspect_raw_export",
    "inspect_raw_export_tree",
    "is_raw_export_file",
    "iter_private_export_chunks",
    "iter_raw_export_files",
    "read_private_export_file",
]
