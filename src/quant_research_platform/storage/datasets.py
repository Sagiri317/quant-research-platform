"""Unified dataset IO, manifests, fingerprints, and comparison helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

import numpy as np
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
from quant_research_platform.runtime.privacy import is_private_boundary_path
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.manifests import write_internal_artifact_manifest
from quant_research_platform.storage.partitions import add_year_partition


class DataSpecProvider(Protocol):
    def get(self, data_id: str) -> DataSpec: ...


class DatasetAccessMode(StrEnum):
    MATERIALIZED = "materialized"
    STREAMING = "streaming"


@dataclass(frozen=True)
class DatasetFingerprint:
    fingerprint_schema_version: str
    schema_hash: str
    row_count: int
    key_columns: tuple[str, ...]
    content_hash: str
    partition_fingerprints: tuple[dict[str, Any], ...]
    file_count: int
    column_names: tuple[str, ...]


@dataclass(frozen=True)
class DatasetComparison:
    comparison_schema_version: str
    matches: bool
    schema_matches: bool
    row_count_matches: bool
    key_columns_match: bool
    content_matches: bool
    partition_mismatch_count: int
    unexplained_delta: int
    summary: dict[str, Any]


@dataclass
class DatasetManifestAccumulator:
    """Incrementally collect manifest stats over materialized or chunked writes."""

    spec: DataSpec
    data_root: Path
    row_count: int = 0
    file_count: int = 0
    file_size_bytes: int = 0
    partitions: set[str] = field(default_factory=set)
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    null_count_summary: dict[str, int] = field(default_factory=dict)
    non_finite_count_summary: dict[str, int] = field(default_factory=dict)
    factor_nan_profile: dict[str, int] = field(default_factory=dict)
    label_nan_profile: dict[str, int] = field(default_factory=dict)
    duplicate_primary_key_count: int = 0
    _seen_key_hashes: set[int] = field(default_factory=set)

    def update_frame(self, frame: pd.DataFrame) -> None:
        self.row_count += int(len(frame))
        self._update_keys(frame)
        self._update_dates(frame)
        self._update_numeric_profiles(frame)

    def record_file(self, path: Path) -> None:
        self.file_count += 1
        self.file_size_bytes += int(path.stat().st_size)
        self.partitions.add(path.parent.as_posix())

    def payload(self) -> dict[str, Any]:
        validation_failed = (
            self.duplicate_primary_key_count > 0
            or any(value > 0 for value in self.null_count_summary.values())
            or any(value > 0 for value in self.non_finite_count_summary.values())
        )
        payload = _data_spec_payload(self.spec)
        payload.update(
            {
                "row_count": int(self.row_count),
                "file_count": int(self.file_count),
                "file_size_bytes": int(self.file_size_bytes),
                "partition_count": len(self.partitions),
                "date_min": _date_string(self.date_min),
                "date_max": _date_string(self.date_max),
                "null_count_summary": dict(sorted(self.null_count_summary.items())),
                "non_finite_count_summary": dict(
                    sorted(self.non_finite_count_summary.items())
                ),
                "duplicate_primary_key_count": int(
                    self.duplicate_primary_key_count
                ),
                "factor_nan_profile": dict(sorted(self.factor_nan_profile.items())),
                "label_nan_profile": dict(sorted(self.label_nan_profile.items())),
                "validation_status": "fail" if validation_failed else "pass",
            }
        )
        return payload

    def _update_keys(self, frame: pd.DataFrame) -> None:
        for column in self.spec.primary_key:
            self.null_count_summary.setdefault(column, 0)
            self.null_count_summary[column] += int(frame[column].isna().sum())
        if frame.empty:
            return
        keys = frame.loc[:, list(self.spec.primary_key)]
        local_duplicates = int(keys.duplicated(list(self.spec.primary_key)).sum())
        hashes = pd.util.hash_pandas_object(keys, index=False).to_numpy("uint64")
        cross_duplicates = 0
        for value in hashes:
            key_hash = int(value)
            if key_hash in self._seen_key_hashes:
                cross_duplicates += 1
            else:
                self._seen_key_hashes.add(key_hash)
        self.duplicate_primary_key_count += local_duplicates + cross_duplicates

    def _update_dates(self, frame: pd.DataFrame) -> None:
        timestamp_column = _timestamp_primary_key_column(self.spec)
        if timestamp_column is None or timestamp_column not in frame or frame.empty:
            return
        values = pd.to_datetime(frame[timestamp_column], errors="coerce").dropna()
        if values.empty:
            return
        current_min = values.min()
        current_max = values.max()
        self.date_min = (
            current_min if self.date_min is None else min(self.date_min, current_min)
        )
        self.date_max = (
            current_max if self.date_max is None else max(self.date_max, current_max)
        )

    def _update_numeric_profiles(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        for column, role in self.spec.column_roles.items():
            if column in self.spec.primary_key or column not in frame:
                continue
            series = frame[column]
            nan_count = int(series.isna().sum())
            if role is ColumnRole.FACTOR and nan_count:
                self.factor_nan_profile[column] = (
                    self.factor_nan_profile.get(column, 0) + nan_count
                )
            if role is ColumnRole.LABEL and nan_count:
                self.label_nan_profile[column] = (
                    self.label_nan_profile.get(column, 0) + nan_count
                )
            if not pd.api.types.is_numeric_dtype(series):
                continue
            numeric = pd.to_numeric(series, errors="coerce")
            values = numeric.to_numpy(dtype="float64", na_value=np.nan)
            inf_count = int(np.isinf(values).sum())
            if inf_count:
                self.non_finite_count_summary[column] = (
                    self.non_finite_count_summary.get(column, 0) + inf_count
                )


def write_dataset(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    root: Path,
    *,
    catalog: DataSpecProvider,
    data_id: str,
    access_mode: DatasetAccessMode = DatasetAccessMode.MATERIALIZED,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Path:
    spec = catalog.get(data_id)
    final_root = Path(root)
    policy = resource_policy or RuntimeResourcePolicy.from_environment()
    policy.require_minimum_free_disk(final_root.parent)
    _validate_dataset_root(final_root, spec)
    chunks = _iter_input_chunks(data, access_mode=access_mode)

    def _write(temp_root: Path) -> Path:
        _write_dataset_contents(chunks, temp_root, spec=spec)
        return final_root

    atomic_write_directory(final_root, _write)
    return final_root


def write_dataset_artifact(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    artifact_root: Path,
    *,
    catalog: DataSpecProvider,
    data_id: str,
    access_mode: DatasetAccessMode = DatasetAccessMode.MATERIALIZED,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Path:
    final_root = Path(artifact_root)
    policy = resource_policy or RuntimeResourcePolicy.from_environment()
    policy.require_minimum_free_disk(final_root.parent)
    spec = catalog.get(data_id)

    def _write(temp_root: Path) -> None:
        accumulator = write_dataset_contents(
            data,
            temp_root / "data",
            catalog=catalog,
            data_id=data_id,
            access_mode=access_mode,
        )
        write_internal_artifact_manifest(
            temp_root / "spec.json",
            _data_spec_payload(spec),
        )
        write_internal_artifact_manifest(
            temp_root / "_manifest" / "manifest.json",
            accumulator.payload(),
        )

    atomic_write_directory(final_root, _write)
    return final_root / "data"


def write_dataset_contents(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    root: Path,
    *,
    catalog: DataSpecProvider,
    data_id: str,
    access_mode: DatasetAccessMode = DatasetAccessMode.MATERIALIZED,
) -> DatasetManifestAccumulator:
    spec = catalog.get(data_id)
    _validate_dataset_root(Path(root), spec)
    return _write_dataset_contents(
        _iter_input_chunks(data, access_mode=access_mode),
        Path(root),
        spec=spec,
    )


def read_dataset(
    root: Path,
    *,
    catalog: DataSpecProvider | None = None,
    data_id: str | None = None,
    access_mode: DatasetAccessMode = DatasetAccessMode.MATERIALIZED,
    years: tuple[int, ...] | None = None,
    columns: tuple[str, ...] | list[str] | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    del resource_policy
    spec = catalog.get(data_id) if catalog is not None and data_id is not None else None
    if access_mode is DatasetAccessMode.STREAMING:
        return _read_dataset_streaming(
            Path(root),
            years=years,
            columns=columns,
            spec=spec,
        )
    frame = pd.concat(
        list(
            _read_dataset_streaming(
                Path(root),
                years=years,
                columns=columns,
                spec=spec,
            )
        ),
        ignore_index=True,
    )
    return _sort_by_primary_key(frame, spec)


def fingerprint_dataset(
    root: Path,
    *,
    catalog: DataSpecProvider,
    data_id: str,
    years: tuple[int, ...] | None = None,
    columns: tuple[str, ...] | list[str] | None = None,
) -> DatasetFingerprint:
    spec = catalog.get(data_id)
    paths = _parquet_part_paths(Path(root), years=years)
    if not paths:
        raise FileNotFoundError(root)
    schema_hash = _schema_hash(paths[0])
    row_count = 0
    row_hash_sum = 0
    row_hash_xor = 0
    partition_rows: dict[str, int] = {}
    partition_hash_sum: dict[str, int] = {}
    partition_hash_xor: dict[str, int] = {}
    file_count = 0
    for chunk in _read_dataset_streaming(
        Path(root),
        years=years,
        columns=columns,
        spec=spec,
    ):
        file_count += 1
        if chunk.empty:
            continue
        row_count += int(len(chunk))
        partition = _partition_label(chunk, spec)
        hashes = pd.util.hash_pandas_object(
            chunk.loc[:, list(chunk.columns)],
            index=False,
        ).to_numpy("uint64")
        chunk_sum = int(hashes.sum(dtype=np.uint64))
        chunk_xor = int(np.bitwise_xor.reduce(hashes, initial=np.uint64(0)))
        row_hash_sum = (row_hash_sum + chunk_sum) % 2**64
        row_hash_xor ^= chunk_xor
        partition_rows[partition] = partition_rows.get(partition, 0) + len(chunk)
        partition_hash_sum[partition] = (
            partition_hash_sum.get(partition, 0) + chunk_sum
        ) % 2**64
        partition_hash_xor[partition] = partition_hash_xor.get(partition, 0) ^ chunk_xor
    content_hash = _sha256_json(
        {
            "row_hash_sum": row_hash_sum,
            "row_hash_xor": row_hash_xor,
            "row_count": row_count,
            "columns": list(columns or spec.required_columns),
        }
    )
    partition_fingerprints = tuple(
        {
            "partition": partition,
            "row_count": int(partition_rows[partition]),
            "content_hash": _sha256_json(
                {
                    "row_hash_sum": int(partition_hash_sum[partition]),
                    "row_hash_xor": int(partition_hash_xor[partition]),
                    "row_count": int(partition_rows[partition]),
                }
            ),
        }
        for partition in sorted(partition_rows)
    )
    return DatasetFingerprint(
        fingerprint_schema_version="0.1",
        schema_hash=schema_hash,
        row_count=row_count,
        key_columns=tuple(spec.primary_key),
        content_hash=content_hash,
        partition_fingerprints=partition_fingerprints,
        file_count=file_count,
        column_names=tuple(columns or spec.required_columns),
    )


def compare_dataset_fingerprints(
    left: DatasetFingerprint,
    right: DatasetFingerprint,
) -> DatasetComparison:
    schema_matches = left.schema_hash == right.schema_hash
    row_count_matches = left.row_count == right.row_count
    key_columns_match = left.key_columns == right.key_columns
    content_matches = left.content_hash == right.content_hash
    left_partitions = {
        str(item["partition"]): item for item in left.partition_fingerprints
    }
    right_partitions = {
        str(item["partition"]): item for item in right.partition_fingerprints
    }
    partition_keys = set(left_partitions).union(right_partitions)
    partition_mismatches = [
        key
        for key in sorted(partition_keys)
        if left_partitions.get(key) != right_partitions.get(key)
    ]
    matches = (
        schema_matches
        and row_count_matches
        and key_columns_match
        and content_matches
        and not partition_mismatches
    )
    return DatasetComparison(
        comparison_schema_version="0.1",
        matches=matches,
        schema_matches=schema_matches,
        row_count_matches=row_count_matches,
        key_columns_match=key_columns_match,
        content_matches=content_matches,
        partition_mismatch_count=len(partition_mismatches),
        unexplained_delta=0 if matches else len(partition_mismatches) + 1,
        summary={
            "schema_hash": {"left": left.schema_hash, "right": right.schema_hash},
            "row_count": {"left": left.row_count, "right": right.row_count},
            "key_columns": {"left": left.key_columns, "right": right.key_columns},
            "content_hash": {
                "left": left.content_hash,
                "right": right.content_hash,
            },
            "partition_mismatches": partition_mismatches,
        },
    )


def parquet_dataset_columns(root: Path) -> tuple[str, ...]:
    paths = _parquet_part_paths(root, years=None)
    if not paths:
        raise FileNotFoundError(root)
    return tuple(pq.ParquetFile(paths[0]).schema.names)


def _write_dataset_contents(
    chunks: Iterator[pd.DataFrame],
    root: Path,
    *,
    spec: DataSpec,
) -> DatasetManifestAccumulator:
    timestamp_column = _validate_dataset_spec(spec)
    accumulator = DatasetManifestAccumulator(spec=spec, data_root=root)
    part_indices: dict[str, int] = {}
    for chunk in chunks:
        frame = _validate_frame(chunk, root, spec)
        accumulator.update_frame(frame)
        _raise_accumulated_failures(accumulator)
        for partition, partition_frame in _partition_frames(
            frame,
            timestamp_column=timestamp_column,
            partition_columns=spec.partition_columns,
        ):
            part_index = part_indices.get(partition, 0)
            part_indices[partition] = part_index + 1
            output_path = root / partition / f"part-{part_index:03d}.parquet"
            _write_parquet_file(partition_frame, output_path)
            accumulator.record_file(output_path)
    if accumulator.row_count == 0:
        raise ValueError("empty_dataset")
    return accumulator


def _iter_input_chunks(
    data: pd.DataFrame | Iterable[pd.DataFrame],
    *,
    access_mode: DatasetAccessMode,
) -> Iterator[pd.DataFrame]:
    if access_mode is DatasetAccessMode.MATERIALIZED:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("materialized_dataset_requires_dataframe")
        yield data
        return
    if access_mode is not DatasetAccessMode.STREAMING:
        raise ValueError("unsupported_dataset_access_mode")
    if isinstance(data, pd.DataFrame):
        raise TypeError("streaming_dataset_requires_iterable")
    yield from data


def _read_dataset_streaming(
    root: Path,
    *,
    years: tuple[int, ...] | None,
    columns: tuple[str, ...] | list[str] | None,
    spec: DataSpec | None,
) -> Iterator[pd.DataFrame]:
    paths = _parquet_part_paths(root, years=years)
    if not paths:
        raise FileNotFoundError(root)
    for path in paths:
        frame = pd.read_parquet(path, columns=columns, engine="pyarrow")
        yield _sort_by_primary_key(frame, spec)


def _sort_by_primary_key(frame: pd.DataFrame, spec: DataSpec | None) -> pd.DataFrame:
    if spec is None or not set(spec.primary_key).issubset(frame.columns):
        return frame.reset_index(drop=True)
    return frame.sort_values(list(spec.primary_key), kind="mergesort").reset_index(
        drop=True
    )


def _partition_frames(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None,
    partition_columns: tuple[str, ...],
) -> Iterator[tuple[str, pd.DataFrame]]:
    if partition_columns == ():
        yield ".", frame
        return
    if timestamp_column is None:
        raise ValueError("missing_partition_timestamp_column")
    partitioned = add_year_partition(frame, timestamp_column=timestamp_column)
    for year, year_frame in partitioned.groupby("year", sort=True):
        yield f"year={int(year)}", year_frame.drop(columns=["year"])


def _validate_frame(frame: pd.DataFrame, root: Path, spec: DataSpec) -> pd.DataFrame:
    del root
    unregistered = sorted(set(frame.columns).difference(spec.column_roles))
    if unregistered:
        raise ValueError(f"unregistered_output_columns: {', '.join(unregistered)}")
    missing = sorted(set(spec.required_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_required_columns: {', '.join(missing)}")
    output = frame.loc[:, list(spec.required_columns)].copy()
    null_key_columns = [
        column for column in spec.primary_key if bool(output[column].isna().any())
    ]
    if null_key_columns:
        raise ValueError(f"null_primary_key_columns: {', '.join(null_key_columns)}")
    duplicates = output.duplicated(list(spec.primary_key))
    if bool(duplicates.any()):
        raise ValueError("duplicate_primary_key")
    inf_columns = _non_finite_columns(output, spec)
    if inf_columns:
        raise ValueError(f"non_finite_columns: {', '.join(inf_columns)}")
    return output


def _raise_accumulated_failures(accumulator: DatasetManifestAccumulator) -> None:
    if accumulator.duplicate_primary_key_count:
        raise ValueError("duplicate_primary_key")
    null_columns = [
        column
        for column, count in accumulator.null_count_summary.items()
        if int(count) > 0
    ]
    if null_columns:
        raise ValueError(f"null_primary_key_columns: {', '.join(null_columns)}")
    non_finite = [
        column
        for column, count in accumulator.non_finite_count_summary.items()
        if int(count) > 0
    ]
    if non_finite:
        raise ValueError(f"non_finite_columns: {', '.join(non_finite)}")


def _non_finite_columns(frame: pd.DataFrame, spec: DataSpec) -> list[str]:
    columns: list[str] = []
    for column, role in spec.column_roles.items():
        if column in spec.primary_key or role is ColumnRole.TIMESTAMP:
            continue
        if column not in frame or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype="float64",
            na_value=np.nan,
        )
        if bool(np.isinf(values).any()):
            columns.append(column)
    return columns


def _validate_dataset_spec(spec: DataSpec) -> str | None:
    if spec.schema_version != "0.1":
        raise ValueError("unsupported_schema_version")
    if not isinstance(spec.grain, DataGrain):
        raise ValueError("invalid_grain")
    if not spec.primary_key:
        raise ValueError("missing_primary_key")
    if spec.partition_columns not in {("year",), ()}:
        raise ValueError("unsupported_partitioning")
    for column, role in spec.column_roles.items():
        if not isinstance(role, ColumnRole):
            raise ValueError(f"invalid_column_role: {column}")
    invalid_key_roles = [
        column
        for column in spec.primary_key
        if spec.column_roles.get(column) not in {ColumnRole.KEY, ColumnRole.TIMESTAMP}
    ]
    if invalid_key_roles:
        raise ValueError(f"invalid_primary_key_roles: {', '.join(invalid_key_roles)}")
    timestamp_columns = [
        column
        for column in spec.primary_key
        if spec.column_roles[column] is ColumnRole.TIMESTAMP
    ]
    if spec.partition_columns == ():
        if timestamp_columns:
            raise ValueError("unpartitioned_data_must_not_use_timestamp_key")
        return None
    if len(timestamp_columns) != 1:
        raise ValueError("invalid_timestamp_primary_key")
    return timestamp_columns[0]


def _validate_dataset_root(root: Path, spec: DataSpec) -> None:
    if not is_private_boundary_path(root):
        raise ValueError("outside_private_boundary")
    path_parts = _path_parts(root)
    if isinstance(spec, InterimDataSpec):
        if _unique_matching_suffix(
            path_parts,
            lambda candidate: _is_interim_output_suffix(candidate, spec),
        ) is None:
            raise ValueError("invalid_interim_output_root")
        return
    if isinstance(spec, ProcessedDataSpec):
        if _unique_matching_suffix(
            path_parts,
            lambda candidate: _is_processed_output_suffix(candidate, spec),
        ) is None:
            raise ValueError("invalid_processed_output_root")
        return
    if isinstance(spec, ResearchDatasetSpec):
        if _unique_matching_suffix(
            path_parts,
            lambda candidate: _is_research_dataset_output_suffix(candidate, spec),
        ) is None:
            raise ValueError("invalid_research_dataset_output_root")
        return
    raise ValueError("unsupported_data_spec_type")


def _path_parts(path: Path) -> tuple[str, ...]:
    return tuple(part for part in PurePosixPath(path.as_posix()).parts if part != "/")


def _unique_matching_suffix(
    parts: tuple[str, ...],
    predicate: Any,
) -> tuple[str, ...] | None:
    candidates = [
        parts[index:] for index, part in enumerate(parts) if part == "data"
    ]
    matches = [candidate for candidate in candidates if predicate(candidate)]
    storage_like_invalid = [
        candidate
        for candidate in candidates
        if _starts_storage_layout(candidate) and not predicate(candidate)
    ]
    if len(matches) != 1 or storage_like_invalid:
        return None
    return cast(tuple[str, ...], matches[0])


def _starts_storage_layout(parts: tuple[str, ...]) -> bool:
    return len(parts) >= 2 and parts[0] == "data" and parts[1] in {
        "interim",
        "processed",
    }


def _is_interim_output_suffix(
    parts: tuple[str, ...],
    spec: InterimDataSpec,
) -> bool:
    return (
        len(parts) == 5
        and parts[:3] == ("data", "interim", str(spec.source_id))
        and _matches_artifact_segment(parts[3], spec.interim_data_id)
        and parts[4] == "data"
    )


def _is_processed_output_suffix(
    parts: tuple[str, ...],
    spec: ProcessedDataSpec,
) -> bool:
    category_parts = tuple(
        part for part in PurePosixPath(spec.category).parts if part != "/"
    )
    expected_prefix = (
        "data",
        "processed",
        str(spec.market_id),
        *category_parts,
    )
    return (
        len(parts) == len(expected_prefix) + 2
        and parts[: len(expected_prefix)] == expected_prefix
        and _matches_artifact_segment(
            parts[len(expected_prefix)], spec.processed_data_id
        )
        and parts[-1] == "data"
    )


def _is_research_dataset_output_suffix(
    parts: tuple[str, ...],
    spec: ResearchDatasetSpec,
) -> bool:
    return (
        len(parts) == 6
        and parts[:4] == ("data", "processed", str(spec.market_id), "research_datasets")
        and _matches_artifact_segment(parts[4], spec.dataset_id)
        and parts[5] == "data"
    )


def _matches_artifact_segment(segment: str, expected: str) -> bool:
    return (
        segment == expected
        or segment.startswith(f".{expected}.tmp-")
        or segment.startswith(".tmp-")
    )


def _parquet_part_paths(root: Path, *, years: tuple[int, ...] | None) -> list[Path]:
    data_root = Path(root)
    unpartitioned = data_root / "part-000.parquet"
    if unpartitioned.exists():
        if years is not None:
            return []
        return [unpartitioned]
    if years is None:
        return sorted(data_root.glob("year=*/*.parquet"))
    paths: list[Path] = []
    for year in years:
        paths.extend(sorted((data_root / f"year={year}").glob("*.parquet")))
    return paths


def _write_parquet_file(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


def _timestamp_primary_key_column(spec: DataSpec) -> str | None:
    timestamp_columns = [
        column
        for column in spec.primary_key
        if spec.column_roles.get(column) is ColumnRole.TIMESTAMP
    ]
    return timestamp_columns[0] if timestamp_columns else None


def _partition_label(frame: pd.DataFrame, spec: DataSpec) -> str:
    timestamp = _timestamp_primary_key_column(spec)
    if timestamp is None or timestamp not in frame or frame.empty:
        return "unpartitioned"
    years = pd.to_datetime(frame[timestamp], errors="coerce").dt.year.dropna().unique()
    if len(years) == 1:
        return f"year={int(years[0])}"
    return "mixed"


def _schema_hash(path: Path) -> str:
    schema = pq.ParquetFile(path).schema_arrow
    fields = [(field.name, str(field.type)) for field in schema]
    return _sha256_json(fields)


def _data_spec_payload(spec: DataSpec) -> dict[str, object]:
    payload: dict[str, object] = {
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
            {"source_id": str(spec.source_id), "interim_data_id": spec.interim_data_id}
        )
    if isinstance(spec, ProcessedDataSpec):
        payload.update(
            {"category": spec.category, "processed_data_id": spec.processed_data_id}
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
    return payload


def _date_string(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).date().isoformat()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DataSpecProvider",
    "DatasetAccessMode",
    "DatasetComparison",
    "DatasetFingerprint",
    "DatasetManifestAccumulator",
    "compare_dataset_fingerprints",
    "fingerprint_dataset",
    "parquet_dataset_columns",
    "read_dataset",
    "write_dataset",
    "write_dataset_artifact",
    "write_dataset_contents",
]
