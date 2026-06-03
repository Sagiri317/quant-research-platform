"""Spec-driven parquet storage helpers for QRP 0.1."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Protocol

import pandas as pd
import pyarrow.parquet as pq

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataGrain,
    DataSpec,
    InterimDataSpec,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import write_manifest
from quant_research_platform.storage.partitions import add_year_partition
from quant_research_platform.storage.privacy import is_private_boundary_path


class DataSpecCatalog(Protocol):
    def get(self, data_id: str) -> DataSpec: ...


def write_interim_data(
    frame: pd.DataFrame,
    *,
    layout: StorageLayout,
    catalog: DataSpecCatalog,
    interim_data_id: str,
) -> Path:
    spec = catalog.get(interim_data_id)
    if not isinstance(spec, InterimDataSpec):
        raise ValueError("expected_interim_data_spec")
    data_root = layout.interim_data_root(spec.source_id, spec.interim_data_id) / "data"
    write_partitioned_parquet_dataset(
        frame,
        data_root,
        catalog=catalog,
        data_id=interim_data_id,
    )
    _write_data_manifest(data_root.parent / "_manifest" / "manifest.json", spec)
    return data_root


def write_processed_data(
    frame: pd.DataFrame,
    *,
    layout: StorageLayout,
    catalog: DataSpecCatalog,
    processed_data_id: str,
) -> Path:
    spec = catalog.get(processed_data_id)
    if not isinstance(spec, ProcessedDataSpec):
        raise ValueError("expected_processed_data_spec")
    data_root = (
        layout.processed_data_root(
            spec.market_id,
            spec.category,
            spec.processed_data_id,
        )
        / "data"
    )
    write_partitioned_parquet_dataset(
        frame,
        data_root,
        catalog=catalog,
        data_id=processed_data_id,
    )
    _write_data_manifest(data_root.parent / "_manifest" / "manifest.json", spec)
    return data_root


def write_research_dataset(
    frame: pd.DataFrame,
    *,
    layout: StorageLayout,
    catalog: DataSpecCatalog,
    dataset_id: str,
) -> Path:
    spec = catalog.get(dataset_id)
    if not isinstance(spec, ResearchDatasetSpec):
        raise ValueError("expected_research_dataset_spec")
    data_root = layout.research_dataset_root(spec.market_id, spec.dataset_id) / "data"
    write_partitioned_parquet_dataset(
        frame,
        data_root,
        catalog=catalog,
        data_id=dataset_id,
    )
    _write_research_dataset_spec(data_root.parent / "spec.json", spec)
    _write_data_manifest(data_root.parent / "_manifest" / "manifest.json", spec)
    return data_root


def write_partitioned_parquet_dataset(
    frame: pd.DataFrame,
    root: Path,
    *,
    catalog: DataSpecCatalog,
    data_id: str,
) -> Path:
    spec = catalog.get(data_id)
    timestamp_column = validate_parquet_dataset_write(frame, root, spec)
    root.mkdir(parents=True, exist_ok=True)
    partitioned = add_year_partition(frame, timestamp_column=timestamp_column)
    for year, year_frame in partitioned.groupby("year", sort=True):
        partition_root = root / f"year={int(year)}"
        _write_parquet_file(
            year_frame.drop(columns=["year"]),
            partition_root / "part-000.parquet",
        )
    return root


def read_partitioned_parquet_dataset(
    root: Path,
    *,
    years: tuple[int, ...] | None = None,
    columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Read a partitioned parquet dataset written by QRP storage helpers."""

    paths = _parquet_part_paths(root, years=years)
    if not paths:
        raise FileNotFoundError(root)
    return pd.concat(
        [
            pd.read_parquet(path, columns=columns, engine="pyarrow")
            for path in paths
        ],
        ignore_index=True,
    )


def parquet_dataset_columns(root: Path) -> tuple[str, ...]:
    """Return columns from the first parquet part in a partitioned dataset."""

    paths = _parquet_part_paths(root, years=None)
    if not paths:
        raise FileNotFoundError(root)
    return tuple(pq.ParquetFile(paths[0]).schema.names)


def _parquet_part_paths(root: Path, *, years: tuple[int, ...] | None) -> list[Path]:
    data_root = Path(root)
    if years is None:
        return sorted(data_root.glob("year=*/*.parquet"))
    paths: list[Path] = []
    for year in years:
        paths.extend(sorted((data_root / f"year={year}").glob("*.parquet")))
    return paths


def validate_parquet_dataset_write(
    frame: pd.DataFrame,
    root: Path,
    spec: DataSpec,
) -> str:
    if spec.schema_version != "0.1":
        raise ValueError("unsupported_schema_version")
    if not isinstance(spec.grain, DataGrain):
        raise ValueError("invalid_grain")
    if not spec.primary_key:
        raise ValueError("missing_primary_key")
    if spec.partition_columns != ("year",):
        raise ValueError("unsupported_partitioning")
    if not is_private_boundary_path(root):
        raise ValueError("outside_private_boundary")
    _validate_allowed_output_root(root, spec)
    _validate_column_roles(spec)

    unregistered = sorted(set(frame.columns).difference(spec.column_roles))
    if unregistered:
        raise ValueError(f"unregistered_output_columns: {', '.join(unregistered)}")
    missing = sorted(set(spec.required_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_required_columns: {', '.join(missing)}")
    null_key_columns = [
        column for column in spec.primary_key if bool(frame[column].isna().any())
    ]
    if null_key_columns:
        raise ValueError(f"null_primary_key_columns: {', '.join(null_key_columns)}")
    duplicate_keys = frame.duplicated(list(spec.primary_key))
    if bool(duplicate_keys.any()):
        raise ValueError("duplicate_primary_key")
    timestamp_columns = [
        column
        for column in spec.primary_key
        if spec.column_roles[column] is ColumnRole.TIMESTAMP
    ]
    if len(timestamp_columns) != 1:
        raise ValueError("invalid_timestamp_primary_key")
    return timestamp_columns[0]


def _validate_column_roles(spec: DataSpec) -> None:
    for column, role in spec.column_roles.items():
        if not isinstance(role, ColumnRole):
            raise ValueError(f"invalid_column_role: {column}")
    missing_key_roles = [
        column
        for column in spec.primary_key
        if spec.column_roles.get(column) not in {ColumnRole.KEY, ColumnRole.TIMESTAMP}
    ]
    if missing_key_roles:
        raise ValueError(f"invalid_primary_key_roles: {', '.join(missing_key_roles)}")


def _validate_allowed_output_root(root: Path, spec: DataSpec) -> None:
    parts = _relative_parts(root)
    if isinstance(spec, InterimDataSpec):
        if parts[:2] != ("data", "interim"):
            raise ValueError("invalid_interim_output_root")
        return
    if isinstance(spec, ProcessedDataSpec):
        if parts[:2] != ("data", "processed"):
            raise ValueError("invalid_processed_output_root")
        return
    if isinstance(spec, ResearchDatasetSpec):
        if parts[:2] != ("data", "processed"):
            raise ValueError("invalid_research_dataset_output_root")
        if "research_datasets" not in parts:
            raise ValueError("invalid_research_dataset_output_root")
        return
    raise ValueError("unsupported_data_spec_type")


def _relative_parts(path: Path) -> tuple[str, ...]:
    parts = tuple(part for part in PurePosixPath(path.as_posix()).parts if part != "/")
    if "data" not in parts:
        return parts
    return parts[parts.index("data") :]


def _write_parquet_file(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


def _write_data_manifest(path: Path, spec: DataSpec) -> Path:
    payload = {
        "schema_version": spec.schema_version,
        "market_id": str(spec.market_id),
        "frequency": spec.frequency.value,
        "grain": spec.grain.value,
        "primary_key": list(spec.primary_key),
        "partition_columns": list(spec.partition_columns),
        "required_columns": list(spec.required_columns),
        "column_roles": {
            column: role.value for column, role in sorted(spec.column_roles.items())
        },
    }
    if isinstance(spec, InterimDataSpec):
        payload.update(
            {
                "source_id": str(spec.source_id),
                "interim_data_id": spec.interim_data_id,
            }
        )
    if isinstance(spec, ProcessedDataSpec):
        payload.update(
            {
                "category": spec.category,
                "processed_data_id": spec.processed_data_id,
            }
        )
    if isinstance(spec, ResearchDatasetSpec):
        payload.update(
            {
                "dataset_id": spec.dataset_id,
                "processed_data_ids": list(spec.processed_data_ids),
                "feature_columns": list(spec.feature_columns),
                "label_columns": list(spec.label_columns),
                "filter_columns": list(spec.filter_columns),
                "context_columns": list(spec.context_columns),
                "source_columns": list(spec.source_columns),
                "universe_id": spec.universe_id,
                "mask_ids": list(spec.mask_ids),
            }
        )
    return write_manifest(path, payload)


def _write_research_dataset_spec(path: Path, spec: ResearchDatasetSpec) -> Path:
    return write_manifest(
        path,
        {
            "schema_version": spec.schema_version,
            "dataset_id": spec.dataset_id,
            "market_id": str(spec.market_id),
            "processed_data_ids": list(spec.processed_data_ids),
            "feature_columns": list(spec.feature_columns),
            "label_columns": list(spec.label_columns),
            "filter_columns": list(spec.filter_columns),
            "context_columns": list(spec.context_columns),
            "source_columns": list(spec.source_columns),
            "universe_id": spec.universe_id,
            "mask_ids": list(spec.mask_ids),
            "frequency": spec.frequency.value,
            "grain": spec.grain.value,
            "primary_key": list(spec.primary_key),
            "partition_columns": list(spec.partition_columns),
            "column_roles": {
                column: role.value for column, role in sorted(spec.column_roles.items())
            },
        },
    )


__all__ = [
    "DataSpecCatalog",
    "parquet_dataset_columns",
    "read_partitioned_parquet_dataset",
    "validate_parquet_dataset_write",
    "write_interim_data",
    "write_partitioned_parquet_dataset",
    "write_processed_data",
    "write_research_dataset",
]
