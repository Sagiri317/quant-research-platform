"""Sanitized schema inventory for private local data roots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

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
    """Build sanitized aggregate schema metadata for private local data roots."""

    root = Path(project_root)
    items = [_inventory_item(root, data_root) for data_root in _iter_data_roots(root)]
    if not items:
        raise ValueError("schema_inventory_no_datasets_found")
    return sorted(items, key=lambda item: item["logical_relative_root"])


def _iter_data_roots(project_root: Path) -> Iterable[Path]:
    allowed_roots = (
        project_root / "data" / "interim",
        project_root / "data" / "processed",
    )
    for layer_root in allowed_roots:
        if not layer_root.exists():
            continue
        for data_root in layer_root.rglob("data"):
            if data_root.is_dir() and any(data_root.rglob("*.parquet")):
                yield data_root


def _inventory_item(project_root: Path, data_root: Path) -> dict[str, Any]:
    artifact_root = data_root.parent
    logical_root = data_root.relative_to(project_root).as_posix()
    spec_payload, warnings = _read_spec_payload(artifact_root)
    files = tuple(sorted(data_root.rglob("*.parquet")))
    columns, dtypes = _merged_schema(files)
    primary_key = tuple(str(value) for value in spec_payload.get("primary_key", ()))
    partition_columns = tuple(
        str(value) for value in spec_payload.get("partition_columns", ())
    )
    if not primary_key:
        primary_key = _infer_primary_key(columns)
        warnings.append(
            "primary_key_inferred" if primary_key else "primary_key_missing"
        )
    duplicate_count, null_key_counts = _key_stats(files, primary_key)
    min_timestamp, max_timestamp = _timestamp_bounds(files)
    layer, category, artifact_id = _logical_identifiers(logical_root)
    return {
        "artifact_id": artifact_id,
        "layer": layer,
        "category": category,
        "logical_relative_root": logical_root,
        "columns": list(columns),
        "dtypes": dtypes,
        "primary_key": list(primary_key),
        "partition_columns": list(partition_columns),
        "file_count": len(files),
        "row_count": _metadata_row_count(files),
        "row_count_by_year": _row_count_by_year(files),
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "duplicate_primary_key_count": duplicate_count,
        "null_key_counts": null_key_counts,
        "selected_important_column_null_counts": _important_null_counts(
            files,
            columns,
        ),
        "schema_hash": _schema_hash(
            columns=columns,
            dtypes=dtypes,
            primary_key=primary_key,
            partition_columns=partition_columns,
        ),
        "warnings": sorted(set(warnings)),
    }


def _read_spec_payload(artifact_root: Path) -> tuple[dict[str, Any], list[str]]:
    spec_path = artifact_root / "spec.json"
    if not spec_path.exists():
        return {}, ["spec_missing"]
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, ["spec_unreadable"]
    if not isinstance(payload, dict):
        return {}, ["spec_invalid"]
    return payload, []


def _merged_schema(files: tuple[Path, ...]) -> tuple[tuple[str, ...], dict[str, str]]:
    columns: set[str] = set()
    dtypes: dict[str, str] = {}
    for file in files:
        schema = pq.ParquetFile(file).schema_arrow
        for field in schema:
            name = str(field.name)
            columns.add(name)
            dtypes.setdefault(name, str(field.type))
    ordered = tuple(sorted(columns))
    return ordered, {column: dtypes[column] for column in ordered}


def _metadata_row_count(files: tuple[Path, ...]) -> int:
    return int(sum(pq.ParquetFile(file).metadata.num_rows for file in files))


def _row_count_by_year(files: tuple[Path, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for file in files:
        year = _partition_year(file)
        if year is None:
            continue
        counts[year] = counts.get(year, 0) + int(pq.ParquetFile(file).metadata.num_rows)
    return dict(sorted(counts.items()))


def _partition_year(file: Path) -> str | None:
    for part in file.parts:
        if part.startswith("year="):
            return part.split("=", 1)[1]
    return None


def _key_stats(
    files: tuple[Path, ...],
    primary_key: tuple[str, ...],
) -> tuple[int, dict[str, int]]:
    if not primary_key:
        return 0, {}
    frames: list[pd.DataFrame] = []
    null_counts = {column: 0 for column in primary_key}
    for file in files:
        frame = pd.read_parquet(file, columns=list(primary_key), engine="pyarrow")
        for column in primary_key:
            null_counts[column] += int(frame[column].isna().sum())
        frames.append(frame)
    if not frames:
        return 0, null_counts
    keys = pd.concat(frames, ignore_index=True)
    return int(keys.duplicated(list(primary_key)).sum()), dict(
        sorted(null_counts.items())
    )


def _timestamp_bounds(files: tuple[Path, ...]) -> tuple[str | None, str | None]:
    min_timestamp: pd.Timestamp | None = None
    max_timestamp: pd.Timestamp | None = None
    for file in files:
        if "timestamp" not in pq.ParquetFile(file).schema.names:
            continue
        timestamps = pd.to_datetime(
            pd.read_parquet(file, columns=["timestamp"], engine="pyarrow")[
                "timestamp"
            ],
            errors="coerce",
        ).dropna()
        if timestamps.empty:
            continue
        file_min = timestamps.min()
        file_max = timestamps.max()
        min_timestamp = (
            file_min if min_timestamp is None else min(min_timestamp, file_min)
        )
        max_timestamp = (
            file_max if max_timestamp is None else max(max_timestamp, file_max)
        )
    return _format_timestamp(min_timestamp), _format_timestamp(max_timestamp)


def _important_null_counts(
    files: tuple[Path, ...],
    columns: tuple[str, ...],
) -> dict[str, int]:
    selected = tuple(
        column for column in columns if column in IMPORTANT_NULL_COUNT_COLUMNS
    )
    if not selected:
        return {}
    counts = {column: 0 for column in selected}
    for file in files:
        available = [
            column for column in selected if column in pq.ParquetFile(file).schema.names
        ]
        if not available:
            continue
        frame = pd.read_parquet(file, columns=available, engine="pyarrow")
        for column in available:
            counts[column] += int(frame[column].isna().sum())
    return dict(sorted(counts.items()))


def _logical_identifiers(logical_root: str) -> tuple[str, str, str]:
    parts = logical_root.split("/")
    if len(parts) < 4 or parts[0] != "data":
        raise ValueError("invalid_inventory_root")
    layer = parts[1]
    if layer == "interim":
        return layer, parts[2], parts[-2]
    if layer == "processed":
        category_parts = parts[3:-2]
        category = "/".join(category_parts) if category_parts else parts[2]
        return layer, category, parts[-2]
    return layer, "", parts[-2]


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
