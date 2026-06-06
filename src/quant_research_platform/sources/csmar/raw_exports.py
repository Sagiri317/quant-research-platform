"""CSMAR raw export inspection and source-local IO helpers."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .raw_schemas import RawExportInspection, inspect_raw_export_file

SUPPORTED_RAW_EXPORT_SUFFIXES = {".csv", ".xlsx", ".txt"}
STRING_CODE_DTYPES = {"Stkcd": str, "Symbol": str, "Indexcd": str}


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
        rows = []
        for row in reader:
            if len(row) > len(header):
                row = row[: len(header) - 1] + [",".join(row[len(header) - 1 :])]
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            rows.append(row)
    return pd.DataFrame(rows, columns=header).replace("", pd.NA)


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

    if dataset == "index_info":
        return sorted(
            path
            for path in raw_root.rglob("csmar__index_info__*.csv")
            if path.is_file() and "__description__" not in path.name
        )

    return sorted(
        path
        for path in (raw_root / dataset).rglob(f"csmar__{dataset}__*.csv")
        if path.is_file() and "__description__" not in path.name
    )


def inspect_raw_export_tree(root: Path) -> list[RawExportIntakeResult]:
    """Inspect supported raw exports under a private raw directory."""

    return [inspect_raw_export(path) for path in iter_raw_export_files(root)]


__all__ = [
    "RawExportIntakeResult",
    "STRING_CODE_DTYPES",
    "SUPPORTED_RAW_EXPORT_SUFFIXES",
    "find_dataset_files",
    "inspect_raw_export",
    "inspect_raw_export_tree",
    "is_raw_export_file",
    "iter_private_export_chunks",
    "iter_raw_export_files",
    "read_private_export_file",
]
