"""提供 raw CSMAR export 检查和 metadata helpers。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .build_interim import (
    write_raw_export_metadata,
)
from .contracts import (
    RawExportInspection,
    inspect_raw_export_file,
)

SUPPORTED_RAW_EXPORT_SUFFIXES = {".csv", ".xlsx", ".txt"}


@dataclass(frozen=True)
class RawExportIntakeResult:
    """记录单个 raw export 检查结果，可附带 metadata output。"""

    path: Path
    inspection: RawExportInspection | None = None
    metadata_path: Path | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """返回检查和可选 metadata 生成是否成功。"""

        return self.error is None and self.inspection is not None


def is_raw_export_file(path: Path) -> bool:
    """判断路径是否像原始 CSMAR export 数据文件。"""

    return (
        path.is_file()
        and path.suffix.lower() in SUPPORTED_RAW_EXPORT_SUFFIXES
        and "[DES]" not in path.name
        and "__description__" not in path.name
        and ".metadata." not in path.name
    )


def iter_raw_export_files(root: Path) -> list[Path]:
    """返回私有 raw directory 下的原始 CSMAR export files。"""

    return sorted(path for path in root.rglob("*") if is_raw_export_file(path))


def inspect_raw_export(
    path: Path,
    *,
    write_metadata: bool = False,
) -> RawExportIntakeResult:
    """检查单个 raw export，并可选写出 metadata sidecar。"""

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
    """检查私有 raw directory 下所有受支持的 raw exports。"""

    return [
        inspect_raw_export(path, write_metadata=write_metadata)
        for path in iter_raw_export_files(root)
    ]
