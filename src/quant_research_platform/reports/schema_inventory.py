"""Sanitized schema inventory for private local data roots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    inspect_dataset,
    read_dataset,
)

IMPORTANT_NULL_COUNT_COLUMNS = frozenset(
    {
        "eligible",
        "can_enter",
        "can_exit",
        "is_st",
        "is_pt",
        "limit_status",
        "forward_return_1d",
        "forward_return_5d",
        "forward_return_10d",
        "forward_return_20d",
        "forward_return_60d",
        "forward_return_120d",
    }
)


def build_schema_inventory(project_root: Path) -> list[dict[str, Any]]:
    """Build sanitized aggregate schema metadata for private dataset artifacts."""

    root = Path(project_root)
    items = [
        _inventory_item(root, artifact_root)
        for artifact_root in _iter_dataset_artifact_roots(root)
    ]
    if not items:
        raise ValueError("schema_inventory_no_datasets_found")
    return sorted(items, key=lambda item: item["logical_relative_root"])


def _iter_dataset_artifact_roots(project_root: Path) -> Iterable[Path]:
    allowed_roots = (
        project_root / "data" / "interim",
        project_root / "data" / "processed",
    )
    for layer_root in allowed_roots:
        if not layer_root.exists():
            continue
        for spec_path in sorted(layer_root.rglob("spec.json")):
            yield spec_path.parent


def _inventory_item(project_root: Path, artifact_root: Path) -> dict[str, Any]:
    inspection = inspect_dataset(artifact_root)
    logical_root = artifact_root.relative_to(project_root).as_posix()
    spec_payload = inspection.spec
    warnings: list[str] = []
    columns = inspection.columns
    primary_key = tuple(str(value) for value in spec_payload.get("primary_key", ()))
    partition_columns = tuple(
        str(value) for value in spec_payload.get("partition_columns", ())
    )
    if not primary_key:
        primary_key = _infer_primary_key(columns)
        warnings.append(
            "primary_key_inferred" if primary_key else "primary_key_missing"
        )
    stats = _aggregate_dataset_stats(artifact_root, columns, primary_key)
    layer, category, artifact_id = _logical_identifiers(logical_root)
    return {
        "artifact_id": artifact_id,
        "layer": layer,
        "category": category,
        "logical_relative_root": logical_root,
        "columns": list(columns),
        "dtypes": stats["dtypes"],
        "primary_key": list(primary_key),
        "partition_columns": list(partition_columns),
        "file_count": inspection.file_count,
        "row_count": stats["row_count"],
        "row_count_by_year": stats["row_count_by_year"],
        "min_timestamp": stats["min_timestamp"],
        "max_timestamp": stats["max_timestamp"],
        "duplicate_primary_key_count": stats["duplicate_primary_key_count"],
        "null_key_counts": stats["null_key_counts"],
        "selected_important_column_null_counts": (
            stats["selected_important_column_null_counts"]
        ),
        "schema_hash": _schema_hash(
            columns=columns,
            dtypes=stats["dtypes"],
            primary_key=primary_key,
            partition_columns=partition_columns,
        ),
        "warnings": sorted(set(warnings)),
    }


def _aggregate_dataset_stats(
    artifact_root: Path,
    columns: tuple[str, ...],
    primary_key: tuple[str, ...],
) -> dict[str, Any]:
    dtypes: dict[str, str] = {}
    row_count = 0
    row_count_by_year: dict[str, int] = {}
    null_key_counts = {column: 0 for column in primary_key}
    duplicate_primary_key_count = 0
    seen_key_hashes: set[int] = set()
    important_columns = tuple(
        column for column in columns if column in IMPORTANT_NULL_COUNT_COLUMNS
    )
    important_null_counts = {column: 0 for column in important_columns}
    min_timestamp: pd.Timestamp | None = None
    max_timestamp: pd.Timestamp | None = None

    chunks = read_dataset(artifact_root, mode=DatasetReadMode.CHUNKED)
    for frame in chunks:
        row_count += int(len(frame))
        for column in frame.columns:
            dtypes.setdefault(str(column), str(frame[column].dtype))
        if "timestamp" in frame.columns:
            timestamps = pd.to_datetime(frame["timestamp"], errors="coerce").dropna()
            if not timestamps.empty:
                year_counts = timestamps.dt.year.astype(str).value_counts()
                for year, count in year_counts.items():
                    row_count_by_year[year] = row_count_by_year.get(year, 0) + int(
                        count
                    )
                chunk_min = timestamps.min()
                chunk_max = timestamps.max()
                min_timestamp = (
                    chunk_min
                    if min_timestamp is None
                    else min(min_timestamp, chunk_min)
                )
                max_timestamp = (
                    chunk_max
                    if max_timestamp is None
                    else max(max_timestamp, chunk_max)
                )
        for column in primary_key:
            null_key_counts[column] += int(frame[column].isna().sum())
        for column in important_columns:
            important_null_counts[column] += int(frame[column].isna().sum())
        if primary_key:
            key_hashes = pd.util.hash_pandas_object(
                frame.loc[:, list(primary_key)],
                index=False,
            ).to_numpy(dtype="uint64")
            for key_hash in key_hashes:
                value = int(key_hash)
                if value in seen_key_hashes:
                    duplicate_primary_key_count += 1
                else:
                    seen_key_hashes.add(value)

    return {
        "dtypes": {column: dtypes.get(column, "") for column in columns},
        "row_count": row_count,
        "row_count_by_year": dict(sorted(row_count_by_year.items())),
        "min_timestamp": _format_timestamp(min_timestamp),
        "max_timestamp": _format_timestamp(max_timestamp),
        "duplicate_primary_key_count": duplicate_primary_key_count,
        "null_key_counts": dict(sorted(null_key_counts.items())),
        "selected_important_column_null_counts": dict(
            sorted(important_null_counts.items())
        ),
    }


def _logical_identifiers(logical_root: str) -> tuple[str, str, str]:
    parts = logical_root.split("/")
    if len(parts) < 4 or parts[0] != "data":
        raise ValueError("invalid_inventory_root")
    layer = parts[1]
    if layer == "interim":
        return layer, parts[2], parts[-1]
    if layer == "processed":
        category_parts = parts[3:-1]
        category = "/".join(category_parts) if category_parts else parts[2]
        return layer, category, parts[-1]
    return layer, "", parts[-1]


def _infer_primary_key(columns: tuple[str, ...]) -> tuple[str, ...]:
    reference_timestamp_key = (
        "market_id",
        "asset_id",
        "source_id",
        "source_asset_id",
        "timestamp",
    )
    canonical_key = ("market_id", "asset_id", "timestamp")
    reference_key = ("market_id", "asset_id", "source_id", "source_asset_id")
    calendar_key = ("market_id", "source_id", "timestamp")
    for candidate in (
        reference_timestamp_key,
        canonical_key,
        reference_key,
        calendar_key,
    ):
        if all(column in columns for column in candidate):
            return candidate
    return ()


def _format_timestamp(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return value.date().isoformat()


def _schema_hash(
    *,
    columns: tuple[str, ...],
    dtypes: dict[str, str],
    primary_key: tuple[str, ...],
    partition_columns: tuple[str, ...],
) -> str:
    encoded = json.dumps(
        {
            "columns": list(columns),
            "dtypes": dtypes,
            "primary_key": list(primary_key),
            "partition_columns": list(partition_columns),
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["build_schema_inventory"]
