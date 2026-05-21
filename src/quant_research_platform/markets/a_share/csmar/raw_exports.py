"""Raw CSMAR export inspection and metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_research_platform.markets.a_share.csmar.build_interim import (
    write_raw_export_metadata,
)
from quant_research_platform.markets.a_share.csmar.contracts import (
    RawExportInspection,
    inspect_raw_export_file,
)

SUPPORTED_RAW_EXPORT_SUFFIXES = {".csv", ".xlsx", ".txt"}


@dataclass(frozen=True)
class RawExportIntakeResult:
    """One raw export inspection result, optionally with metadata output."""

    path: Path
    inspection: RawExportInspection | None = None
    metadata_path: Path | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether inspection and optional metadata generation succeeded."""

        return self.error is None and self.inspection is not None


def is_raw_export_file(path: Path) -> bool:
    """Return whether a path looks like a raw CSMAR export data file."""

    return (
        path.is_file()
        and path.suffix.lower() in SUPPORTED_RAW_EXPORT_SUFFIXES
        and "[DES]" not in path.name
        and "__description__" not in path.name
        and ".metadata." not in path.name
    )


def iter_raw_export_files(root: Path) -> list[Path]:
    """Return raw CSMAR export files under a private raw directory."""

    return sorted(path for path in root.rglob("*") if is_raw_export_file(path))


def inspect_raw_export(
    path: Path,
    *,
    write_metadata: bool = False,
) -> RawExportIntakeResult:
    """Inspect one raw export and optionally write a metadata sidecar."""

    try:
        metadata_path = write_raw_export_metadata(path) if write_metadata else None
        inspection = inspect_raw_export_file(path)
    except Exception as exc:  # noqa: BLE001
        return RawExportIntakeResult(path=path, error=str(exc))

    return RawExportIntakeResult(
        path=path,
        inspection=inspection,
        metadata_path=metadata_path,
    )


def inspect_raw_export_tree(
    root: Path,
    *,
    write_metadata: bool = False,
) -> list[RawExportIntakeResult]:
    """Inspect all supported raw exports below a private raw directory."""

    return [
        inspect_raw_export(path, write_metadata=write_metadata)
        for path in iter_raw_export_files(root)
    ]
