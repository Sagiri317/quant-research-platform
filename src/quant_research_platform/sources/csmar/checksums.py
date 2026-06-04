"""Internal provider raw-export checksum helpers."""

from pathlib import Path

from quant_research_platform.sources.csmar.raw_schemas import build_raw_export_metadata
from quant_research_platform.storage.checksums import sha256_file


def raw_export_sha256(path: str | Path) -> str:
    return sha256_file(Path(path))


__all__ = ["build_raw_export_metadata", "raw_export_sha256"]
