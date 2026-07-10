"""Workflow-owned schema inventory payload builder."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.runtime.resources import (
    active_runtime_resource_policy,
    require_free_disk_headroom,
)
from quant_research_platform.storage._dataset_key_tracking import ExactPrimaryKeyIndex
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    inspect_dataset,
    read_dataset,
)


def build_schema_inventory_payload(
    project_root: Path,
    *,
    important_null_count_columns: Collection[str] | None = None,
) -> dict[str, Any]:
    """Build sanitized aggregate schema metadata for private dataset artifacts."""

    root = Path(project_root)
    important_columns = frozenset(str(column) for column in (
        important_null_count_columns or ()
    ))
    items = [
        _inventory_item(
            root,
            artifact_root,
            important_null_count_columns=important_columns,
        )
        for artifact_root in _iter_dataset_artifact_roots(root)
    ]
    if not items:
        raise ValueError("schema_inventory_no_datasets_found")
    return {
        "schema_id": "qrp-schema-inventory",
        "datasets": sorted(items, key=lambda item: item["logical_relative_root"]),
    }


def _iter_dataset_artifact_roots(project_root: Path) -> Iterable[Path]:
    allowed_roots = (
        project_root / "data" / "interim",
        project_root / "data" / "processed",
        project_root / "data" / "research",
    )
    for layer_root in allowed_roots:
        if not layer_root.exists():
            continue
        for spec_path in sorted(layer_root.rglob("spec.json")):
            yield spec_path.parent


def _inventory_item(
    project_root: Path,
    artifact_root: Path,
    *,
    important_null_count_columns: frozenset[str],
) -> dict[str, Any]:
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
    stats = _aggregate_dataset_stats(
        artifact_root,
        columns,
        primary_key,
        important_null_count_columns=important_null_count_columns,
        exact_duplicate_count=(
            int(inspection.manifest["duplicate_primary_key_count"])
            if "duplicate_primary_key_count" in inspection.manifest
            else None
        ),
    )
    layer, kind, artifact_id = _logical_identifiers(logical_root)
    return {
        "artifact_id": artifact_id,
        "layer": layer,
        "kind": kind,
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
    *,
    important_null_count_columns: frozenset[str],
    exact_duplicate_count: int | None,
) -> dict[str, Any]:
    dtypes: dict[str, str] = {}
    row_count = 0
    row_count_by_year: dict[str, int] = {}
    null_key_counts = {column: 0 for column in primary_key}
    important_columns = tuple(
        column for column in columns if column in important_null_count_columns
    )
    important_null_counts = {column: 0 for column in important_columns}
    min_timestamp: pd.Timestamp | None = None
    max_timestamp: pd.Timestamp | None = None
    duplicate_primary_key_count = (
        0 if exact_duplicate_count is None else exact_duplicate_count
    )

    chunks = read_dataset(artifact_root, mode=DatasetReadMode.CHUNKED)
    with ExitStack() as stack:
        key_index: ExactPrimaryKeyIndex | None = None
        policy = None
        if exact_duplicate_count is None and primary_key:
            policy = active_runtime_resource_policy()
            require_free_disk_headroom(
                policy,
                artifact_root.parent,
                reserve_bytes=policy.effective_chunk_bytes * 2,
            )
            key_index = stack.enter_context(
                ExactPrimaryKeyIndex(
                    primary_key=primary_key,
                    timestamp_columns=tuple(
                        column for column in primary_key if column == "timestamp"
                    ),
                    scratch_parent=artifact_root.parent,
                )
            )
        for frame in chunks:
            row_count += int(len(frame))
            for column in frame.columns:
                dtypes.setdefault(str(column), str(frame[column].dtype))
            if "timestamp" in frame.columns:
                timestamps = pd.to_datetime(
                    frame["timestamp"], errors="coerce"
                ).dropna()
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
            if key_index is not None:
                key_frame = frame.loc[:, list(primary_key)]
                key_bytes = int(
                    key_frame.memory_usage(index=True, deep=True).sum()
                )
                assert policy is not None
                require_free_disk_headroom(
                    policy,
                    artifact_root.parent,
                    reserve_bytes=policy.effective_chunk_bytes + 4 * key_bytes,
                )
                duplicate_primary_key_count += key_index.add_frame(key_frame)

    return {
        "dtypes": {column: dtypes.get(column, "") for column in columns},
        "row_count": row_count,
        "row_count_by_year": dict(sorted(row_count_by_year.items())),
        "min_timestamp": _format_timestamp(min_timestamp),
        "max_timestamp": _format_timestamp(max_timestamp),
        "null_key_counts": dict(sorted(null_key_counts.items())),
        "duplicate_primary_key_count": duplicate_primary_key_count,
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
        return layer, "interim", parts[-1]
    if layer == "processed":
        kind_parts = parts[3:-1]
        kind = "/".join(kind_parts) if kind_parts else parts[2]
        return layer, kind, parts[-1]
    if layer == "research":
        return layer, "research_dataset", parts[-1]
    return layer, "", parts[-1]


def _infer_primary_key(columns: tuple[str, ...]) -> tuple[str, ...]:
    canonical_key = ("market_id", "asset_id", "timestamp")
    source_asset_timestamp_key = ("source_asset_id", "timestamp")
    source_asset_key = ("source_asset_id",)
    calendar_key = ("timestamp",)
    for candidate in (
        canonical_key,
        source_asset_timestamp_key,
        source_asset_key,
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


__all__ = ["build_schema_inventory_payload"]
