"""Dataset IO public and internal types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class DatasetReadMode(StrEnum):
    MATERIALIZED = "materialized"
    CHUNKED = "chunked"


@dataclass(frozen=True)
class DatasetInspection:
    root: Path
    data_root: Path
    spec: dict[str, Any]
    columns: tuple[str, ...]
    schema_hash: str
    file_count: int
    row_count: int
    partitions: tuple[str, ...]
    partition_row_counts: dict[str, int]
    date_min: str | None
    date_max: str | None
    manifest: dict[str, Any]


@dataclass(frozen=True)
class DatasetFingerprint:
    schema_hash: str
    row_count: int
    key_columns: tuple[str, ...]
    content_hash: str
    partition_fingerprints: tuple[dict[str, Any], ...]
    file_count: int
    column_names: tuple[str, ...]


@dataclass(frozen=True)
class DatasetComparison:
    status: str
    matches: bool
    summary: dict[str, Any]
    layout_delta: dict[str, Any]


__all__ = [
    "DatasetReadMode",
    "DatasetInspection",
    "DatasetFingerprint",
    "DatasetComparison",
]
