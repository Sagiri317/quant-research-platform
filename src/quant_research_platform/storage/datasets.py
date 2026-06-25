"""Dataset artifact IO, inspection, fingerprints, and comparisons."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

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
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.share_safety import is_local_only_path
from quant_research_platform.storage.manifests import (
    read_manifest,
    write_internal_artifact_manifest,
)


class DatasetReadMode(StrEnum):
    MATERIALIZED = "materialized"
    CHUNKED = "chunked"


_DATASET_INSTANCE_METADATA_ALLOWED_KEYS = frozenset({"input_fingerprints"})
_DATASET_MANIFEST_RESERVED_KEYS = frozenset(
    {
        "schema_version",
        "data_id",
        "frequency",
        "grain",
        "primary_key",
        "partition_columns",
        "required_columns",
        "column_roles",
        "source_id",
        "market_id",
        "kind",
        "source_input_data_ids",
        "processed_input_data_ids",
        "input_data_ids",
        "base_input_data_id",
        "factor_columns",
        "feature_columns",
        "label_columns",
        "filter_columns",
        "context_columns",
        "key_coverage",
        "row_count",
        "file_count",
        "file_size_bytes",
        "columns",
        "schema_hash",
        "partition_row_counts",
        "partition_count",
        "partitions",
        "date_min",
        "date_max",
        "null_count_summary",
        "non_finite_count_summary",
        "duplicate_primary_key_count",
        "factor_nan_profile",
        "label_nan_profile",
        "validation_status",
    }
)
_DATASET_INPUT_FINGERPRINT_KEYS = frozenset(
    {
        "fingerprint_schema_version",
        "schema_hash",
        "row_count",
        "key_columns",
        "content_hash",
        "partition_fingerprints",
        "file_count",
        "column_names",
    }
)
_DATASET_PARTITION_FINGERPRINT_KEYS = frozenset(
    {"partition", "row_count", "content_hash"}
)


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
    status: str
    matches: bool
    summary: dict[str, Any]
    layout_delta: dict[str, Any]


@dataclass(frozen=True)
class _DatasetReadContext:
    root: Path
    data_root: Path
    spec: dict[str, Any]
    manifest: dict[str, Any]
    all_paths: tuple[Path, ...]
    read_paths: tuple[Path, ...]
    logical_columns: tuple[str, ...]
    physical_columns: tuple[str, ...]
    timestamp_column: str | None
    time_window: TimeWindow | None
    schema: tuple[tuple[str, str], ...]
    schema_hash: str
    parquet_filters: list[tuple[str, str, Any]] | None


@dataclass
class _WriteStats:
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
    partition_row_counts: dict[str, int] = field(default_factory=dict)
    columns: tuple[str, ...] | None = None
    schema_hash: str | None = None
    _seen_key_hashes: set[int] = field(default_factory=set)
    _schema: tuple[tuple[str, str], ...] | None = None

    def update_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        unregistered = sorted(set(frame.columns).difference(self.spec.column_roles))
        if unregistered:
            raise ValueError(f"unregistered_output_columns: {', '.join(unregistered)}")
        missing = sorted(set(self.spec.required_columns).difference(frame.columns))
        if missing:
            raise ValueError(f"missing_required_columns: {', '.join(missing)}")
        output = frame.loc[:, list(self.spec.required_columns)].copy()
        self.row_count += int(len(output))
        self._update_keys(output)
        self._update_dates(output)
        self._update_numeric_profiles(output)
        _raise_accumulated_failures(self)
        return output

    def record_file(self, path: Path, *, row_count: int) -> None:
        parquet_file = pq.ParquetFile(path)
        actual_row_count = int(parquet_file.metadata.num_rows)
        if actual_row_count != int(row_count):
            raise ValueError("dataset_row_count_mismatch")
        schema = _canonical_schema_from_file(parquet_file)
        if self._schema is None:
            self._schema = schema
            self.columns = tuple(name for name, _ in schema)
            self.schema_hash = _sha256_json(schema)
        elif schema != self._schema:
            raise ValueError("dataset_schema_mismatch")
        partition = _relative_partition_label(self.data_root, path)
        self.file_count += 1
        self.file_size_bytes += int(path.stat().st_size)
        self.partitions.add(partition)
        self.partition_row_counts[partition] = (
            self.partition_row_counts.get(partition, 0) + actual_row_count
        )

    def payload(self) -> dict[str, Any]:
        if self.columns is None or self.schema_hash is None:
            raise ValueError("dataset_schema_missing")
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
                "columns": list(self.columns),
                "schema_hash": self.schema_hash,
                "partition_row_counts": {
                    key: int(value)
                    for key, value in sorted(self.partition_row_counts.items())
                },
                "partition_count": len(self.partitions),
                "partitions": sorted(self.partitions),
                "date_min": _date_string(self.date_min),
                "date_max": _date_string(self.date_max),
                "null_count_summary": dict(sorted(self.null_count_summary.items())),
                "non_finite_count_summary": dict(
                    sorted(self.non_finite_count_summary.items())
                ),
                "duplicate_primary_key_count": int(self.duplicate_primary_key_count),
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
        timestamp_column = _timestamp_primary_key_column_from_spec(self.spec)
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
    dataset_root: Path,
    data: pd.DataFrame | Iterable[pd.DataFrame],
    *,
    spec: DataSpec,
    instance_metadata: Mapping[str, Any] | None = None,
) -> None:
    final_root = Path(dataset_root)
    _validate_dataset_spec(spec)
    _validate_dataset_artifact_root(final_root, spec)
    if final_root.exists():
        raise ValueError("dataset_exists")

    def _writer(temp_root: Path) -> None:
        accumulator = _write_contents(
            _iter_input_chunks(data),
            temp_root / "data",
            spec=spec,
        )
        write_internal_artifact_manifest(
            temp_root / "spec.json",
            _data_spec_payload(spec),
        )
        write_internal_artifact_manifest(
            temp_root / "_manifest" / "manifest.json",
            _merge_dataset_instance_metadata(
                accumulator.payload(),
                instance_metadata,
            ),
        )
        _validate_written_artifact(temp_root, accumulator)

    _staged_write_directory(final_root, _writer)


def read_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
    mode: DatasetReadMode = DatasetReadMode.MATERIALIZED,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    context = _build_dataset_read_context(
        dataset_root,
        columns=columns,
        time_window=time_window,
    )

    chunks = _read_dataset_chunks(context)
    if mode is DatasetReadMode.CHUNKED:
        return chunks
    if mode is not DatasetReadMode.MATERIALIZED:
        raise ValueError("unsupported_dataset_read_mode")
    frames = list(chunks)
    if not frames:
        return pd.DataFrame(columns=list(context.logical_columns))
    return pd.concat(frames, ignore_index=True)


def inspect_dataset(
    dataset_root: Path,
    *,
    time_window: TimeWindow | None = None,
) -> DatasetInspection:
    context = _build_dataset_read_context(
        dataset_root,
        columns=None,
        time_window=time_window,
    )
    runtime = _inspection_runtime_metadata(
        context,
    )
    return DatasetInspection(
        root=context.root,
        data_root=context.data_root,
        spec=context.spec,
        columns=tuple(name for name, _ in context.schema),
        schema_hash=context.schema_hash,
        file_count=int(runtime["file_count"]),
        row_count=int(runtime["row_count"]),
        partitions=tuple(runtime["partitions"]),
        partition_row_counts=dict(runtime["partition_row_counts"]),
        date_min=runtime["date_min"],
        date_max=runtime["date_max"],
        manifest=context.manifest,
    )


def fingerprint_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
) -> DatasetFingerprint:
    context = _build_dataset_read_context(
        dataset_root,
        columns=columns,
        time_window=time_window,
    )
    key_columns = tuple(str(column) for column in context.spec["primary_key"])
    hash_columns = tuple(dict.fromkeys((*key_columns, *context.logical_columns)))
    hash_physical_columns = _read_columns_for_time_window(
        hash_columns,
        spec=context.spec,
        time_window=context.time_window,
    )
    output_columns = tuple(context.logical_columns)
    row_count = 0
    row_hash_sum = 0
    row_hash_xor = 0
    partition_rows: dict[str, int] = {}
    partition_hash_sum: dict[str, int] = {}
    partition_hash_xor: dict[str, int] = {}
    file_count = 0

    for path in context.read_paths:
        chunk = pd.read_parquet(
            path,
            columns=list(hash_physical_columns),
            engine="pyarrow",
            filters=context.parquet_filters,
        )
        chunk = _filter_frame_by_time_window(
            chunk,
            spec=context.spec,
            time_window=context.time_window,
        )
        chunk = chunk.loc[:, list(hash_columns)]
        if chunk.empty:
            continue
        file_count += 1
        hashes = pd.util.hash_pandas_object(
            chunk.loc[:, list(hash_columns)],
            index=False,
        ).to_numpy("uint64")
        chunk_sum = int(hashes.sum(dtype=np.uint64))
        chunk_xor = int(np.bitwise_xor.reduce(hashes, initial=np.uint64(0)))
        partition = _relative_partition_label(context.data_root, path)
        row_count += int(len(chunk))
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
            "columns": list(output_columns),
            "hash_columns": list(hash_columns),
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
                    "hash_columns": list(hash_columns),
                }
            ),
        }
        for partition in sorted(partition_rows)
    )
    return DatasetFingerprint(
        fingerprint_schema_version="0.1",
        schema_hash=context.schema_hash,
        row_count=row_count,
        key_columns=key_columns,
        content_hash=content_hash,
        partition_fingerprints=partition_fingerprints,
        file_count=file_count,
        column_names=output_columns,
    )


def compare_datasets(
    left_root: Path,
    right_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
) -> DatasetComparison:
    left_path = Path(left_root)
    right_path = Path(right_root)
    if not left_path.exists():
        return _comparison("MISSING_LEFT")
    if not right_path.exists():
        return _comparison("MISSING_RIGHT")
    try:
        left = fingerprint_dataset(
            left_path,
            columns=columns,
            time_window=time_window,
        )
        right = fingerprint_dataset(
            right_path,
            columns=columns,
            time_window=time_window,
        )
    except Exception as exc:
        return _comparison("INSPECTION_FAILED", summary={"error": str(exc)})

    layout_delta = {
        "file_count": {"left": left.file_count, "right": right.file_count},
    }
    summary = {
        "schema_hash": {"left": left.schema_hash, "right": right.schema_hash},
        "row_count": {"left": left.row_count, "right": right.row_count},
        "key_columns": {"left": left.key_columns, "right": right.key_columns},
        "content_hash": {"left": left.content_hash, "right": right.content_hash},
        "partitions": {
            "left": left.partition_fingerprints,
            "right": right.partition_fingerprints,
        },
    }
    if left.schema_hash != right.schema_hash or left.column_names != right.column_names:
        status = "SCHEMA_MISMATCH"
    elif left.row_count != right.row_count:
        status = "ROW_COUNT_MISMATCH"
    elif left.key_columns != right.key_columns:
        status = "KEY_MISMATCH"
    elif _partition_keys(left) != _partition_keys(right):
        status = "PARTITION_MISMATCH"
    elif left.content_hash != right.content_hash:
        status = "CONTENT_MISMATCH"
    else:
        status = "MATCH"
    return _comparison(status, summary=summary, layout_delta=layout_delta)


def _comparison(
    status: str,
    *,
    summary: dict[str, Any] | None = None,
    layout_delta: dict[str, Any] | None = None,
) -> DatasetComparison:
    return DatasetComparison(
        comparison_schema_version="0.1",
        status=status,
        matches=status == "MATCH",
        summary={} if summary is None else summary,
        layout_delta={} if layout_delta is None else layout_delta,
    )


def _merge_dataset_instance_metadata(
    payload: Mapping[str, Any],
    instance_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(payload)
    if instance_metadata is None:
        return result
    if not isinstance(instance_metadata, Mapping):
        raise TypeError("invalid_dataset_instance_metadata")
    metadata: dict[str, Any] = {}
    for key, value in instance_metadata.items():
        if not isinstance(key, str):
            raise ValueError("unsupported_dataset_instance_metadata_key")
        if key in _DATASET_MANIFEST_RESERVED_KEYS or key in result:
            raise ValueError(f"reserved_dataset_instance_metadata_key: {key}")
        if key not in _DATASET_INSTANCE_METADATA_ALLOWED_KEYS:
            raise ValueError(f"unsupported_dataset_instance_metadata_key: {key}")
        if key == "input_fingerprints":
            metadata[key] = _validate_input_fingerprints_metadata(value)
    _require_json_serializable(metadata)
    result.update(metadata)
    return result


def _require_json_serializable(value: Mapping[str, Any]) -> None:
    try:
        json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)
    except TypeError as exc:
        raise ValueError("dataset_instance_metadata_not_json_serializable") from exc
    except ValueError as exc:
        raise ValueError("dataset_instance_metadata_not_json_serializable") from exc


def _validate_input_fingerprints_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("invalid_input_fingerprints_metadata")
    fingerprints: dict[str, Any] = {}
    for data_id, fingerprint in value.items():
        if not isinstance(data_id, str) or not data_id:
            raise ValueError("invalid_input_fingerprints_metadata")
        fingerprints[data_id] = _validate_input_fingerprint_payload(fingerprint)
    return fingerprints


def _validate_input_fingerprint_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("invalid_input_fingerprints_metadata")
    keys = set(value)
    if keys != _DATASET_INPUT_FINGERPRINT_KEYS:
        raise ValueError("invalid_input_fingerprints_metadata")
    return {
        "fingerprint_schema_version": _require_string(
            value["fingerprint_schema_version"],
        ),
        "schema_hash": _require_string(value["schema_hash"]),
        "row_count": _require_non_negative_int(value["row_count"]),
        "key_columns": _require_string_list(value["key_columns"]),
        "content_hash": _require_string(value["content_hash"]),
        "partition_fingerprints": [
            _validate_partition_fingerprint_payload(item)
            for item in _require_list(value["partition_fingerprints"])
        ],
        "file_count": _require_non_negative_int(value["file_count"]),
        "column_names": _require_string_list(value["column_names"]),
    }


def _validate_partition_fingerprint_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("invalid_input_fingerprints_metadata")
    keys = set(value)
    if keys != _DATASET_PARTITION_FINGERPRINT_KEYS:
        raise ValueError("invalid_input_fingerprints_metadata")
    return {
        "partition": _require_string(value["partition"]),
        "row_count": _require_non_negative_int(value["row_count"]),
        "content_hash": _require_string(value["content_hash"]),
    }


def _require_string(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid_input_fingerprints_metadata")
    return value


def _require_string_list(value: Any) -> list[str]:
    values = _require_list(value)
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError("invalid_input_fingerprints_metadata")
    return list(values)


def _require_list(value: Any) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("invalid_input_fingerprints_metadata")
    return list(value)


def _require_non_negative_int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("invalid_input_fingerprints_metadata")
    return int(value)


def _staged_write_directory(
    final_root: Path,
    writer: Any,
) -> None:
    final_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = final_root.with_name(f".tmp-{uuid4().hex[:12]}")
    backup_root = final_root.with_name(f".replace-{uuid4().hex[:12]}")
    try:
        temp_root.mkdir()
        writer(temp_root)
        if final_root.exists():
            raise ValueError("dataset_exists")
        temp_root.rename(final_root)
    except Exception:
        _remove_tree(temp_root)
        if backup_root.exists() and not final_root.exists():
            backup_root.rename(final_root)
        raise


def _remove_tree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _write_contents(
    chunks: Iterator[pd.DataFrame],
    root: Path,
    *,
    spec: DataSpec,
) -> _WriteStats:
    timestamp_column = _validate_dataset_spec(spec)
    accumulator = _WriteStats(spec=spec, data_root=root)
    part_indices: dict[str, int] = {}
    empty_schema_frame: pd.DataFrame | None = None
    for chunk in chunks:
        frame = accumulator.update_frame(chunk)
        if frame.empty:
            if empty_schema_frame is None:
                empty_schema_frame = frame
            continue
        for partition, partition_frame in _partition_frames(
            frame,
            timestamp_column=timestamp_column,
            partition_columns=spec.partition_columns,
        ):
            part_index = part_indices.get(partition, 0)
            part_indices[partition] = part_index + 1
            output_path = _partition_part_path(root, partition, part_index)
            _write_parquet_file(partition_frame, output_path)
            accumulator.record_file(output_path, row_count=len(partition_frame))
    if accumulator.row_count == 0:
        if empty_schema_frame is None:
            raise ValueError("empty_dataset_schema_missing")
        output_path = _partition_part_path(root, "", 0)
        _write_parquet_file(empty_schema_frame, output_path)
        accumulator.record_file(output_path, row_count=0)
    return accumulator


def _iter_input_chunks(
    data: pd.DataFrame | Iterable[pd.DataFrame],
) -> Iterator[pd.DataFrame]:
    if isinstance(data, pd.DataFrame):
        yield data
        return
    for chunk in data:
        if not isinstance(chunk, pd.DataFrame):
            raise TypeError("invalid_dataset_chunk")
        yield chunk


def _build_dataset_read_context(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
) -> _DatasetReadContext:
    root = _require_artifact_root(Path(dataset_root))
    spec_path = root / "spec.json"
    data_root = root / "data"
    manifest_path = root / "_manifest" / "manifest.json"
    if not spec_path.is_file():
        raise ValueError("dataset_spec_missing")
    if not data_root.is_dir():
        raise ValueError("dataset_data_dir_missing")
    spec_payload = read_manifest(spec_path)
    if not manifest_path.is_file():
        raise ValueError("dataset_manifest_missing")
    manifest = read_manifest(manifest_path)
    if not _manifest_has_runtime_metadata(manifest):
        raise ValueError("manifest_runtime_metadata_missing")
    all_paths = tuple(_candidate_data_files(data_root, spec=spec_payload))
    if not all_paths:
        raise ValueError("dataset_data_dir_missing")
    schema = _validated_schema(list(all_paths))
    schema_columns = tuple(name for name, _ in schema)
    schema_hash = _sha256_json(schema)
    _validate_manifest_runtime_metadata(
        manifest,
        data_root=data_root,
        paths=list(all_paths),
        validate_row_counts=True,
    )
    if tuple(str(column) for column in manifest["columns"]) != schema_columns:
        raise ValueError("manifest_columns_mismatch")
    if str(manifest["schema_hash"]) != schema_hash:
        raise ValueError("manifest_schema_hash_mismatch")
    logical_columns = _resolve_columns_from_schema(
        schema_columns,
        spec=spec_payload,
        columns=columns,
    )
    timestamp_column = _timestamp_column_from_spec_payload(spec_payload)
    physical_columns = _read_columns_for_time_window(
        logical_columns,
        spec=spec_payload,
        time_window=time_window,
    )
    return _DatasetReadContext(
        root=root,
        data_root=data_root,
        spec=spec_payload,
        manifest=manifest,
        all_paths=all_paths,
        read_paths=tuple(
            _prune_paths_for_time_window(
                data_root,
                all_paths,
                spec=spec_payload,
                time_window=time_window,
            )
        ),
        logical_columns=logical_columns,
        physical_columns=physical_columns,
        timestamp_column=timestamp_column,
        time_window=time_window,
        schema=schema,
        schema_hash=schema_hash,
        parquet_filters=_parquet_filters_for_time_window(
            timestamp_column=timestamp_column,
            time_window=time_window,
        ),
    )


def _read_dataset_chunks(
    context: _DatasetReadContext,
) -> Iterator[pd.DataFrame]:
    if not context.read_paths:
        yield pd.DataFrame(columns=list(context.logical_columns))
        return
    for path in context.read_paths:
        frame = pd.read_parquet(
            path,
            columns=list(context.physical_columns),
            engine="pyarrow",
            filters=context.parquet_filters,
        )
        filtered = _filter_frame_by_time_window(
            frame,
            spec=context.spec,
            time_window=context.time_window,
        )
        yield filtered.loc[:, list(context.logical_columns)]


def _inspection_runtime_metadata(
    context: _DatasetReadContext,
) -> dict[str, Any]:
    if context.time_window is None:
        partition_row_counts = _manifest_partition_row_counts(context.manifest)
        return {
            "file_count": int(context.manifest["file_count"]),
            "row_count": int(context.manifest["row_count"]),
            "partitions": tuple(sorted(partition_row_counts)),
            "partition_row_counts": partition_row_counts,
            "date_min": context.manifest.get("date_min"),
            "date_max": context.manifest.get("date_max"),
        }
    timestamp_column = context.timestamp_column
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    row_count = 0
    file_count = 0
    partition_row_counts: dict[str, int] = {}
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    for path in context.read_paths:
        frame = pd.read_parquet(
            path,
            columns=[timestamp_column],
            engine="pyarrow",
            filters=context.parquet_filters,
        )
        filtered = _filter_frame_by_time_window(
            frame,
            spec=context.spec,
            time_window=context.time_window,
        )
        if filtered.empty:
            continue
        file_count += 1
        partition = _relative_partition_label(context.data_root, path)
        partition_row_counts[partition] = partition_row_counts.get(partition, 0) + len(
            filtered
        )
        row_count += len(filtered)
        values = pd.to_datetime(filtered[timestamp_column], errors="coerce").dropna()
        if not values.empty:
            current_min = values.min()
            current_max = values.max()
            date_min = current_min if date_min is None else min(date_min, current_min)
            date_max = current_max if date_max is None else max(date_max, current_max)
    return {
        "file_count": file_count,
        "row_count": int(row_count),
        "partitions": tuple(sorted(partition_row_counts)),
        "partition_row_counts": dict(sorted(partition_row_counts.items())),
        "date_min": _date_string(date_min),
        "date_max": _date_string(date_max),
    }


def _filter_frame_by_time_window(
    frame: pd.DataFrame,
    *,
    spec: Mapping[str, Any],
    time_window: TimeWindow | None,
) -> pd.DataFrame:
    if time_window is None:
        return frame
    timestamp_column = _timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    if timestamp_column not in frame:
        raise ValueError("time_window_timestamp_column_not_selected")
    values = pd.to_datetime(frame[timestamp_column], errors="coerce")
    dates = values.dt.date
    mask = (dates >= time_window.start) & (dates <= time_window.end)
    return frame.loc[mask.fillna(False)].copy()


def _prune_paths_for_time_window(
    data_root: Path,
    paths: tuple[Path, ...],
    *,
    spec: Mapping[str, Any],
    time_window: TimeWindow | None,
) -> tuple[Path, ...]:
    if time_window is None:
        return paths
    timestamp_column = _timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    partition_columns = tuple(
        str(column) for column in spec.get("partition_columns", ())
    )
    if partition_columns != ("year",):
        return paths
    years = {
        f"year={year}"
        for year in range(int(time_window.start.year), int(time_window.end.year) + 1)
    }
    return tuple(
        path for path in paths if _relative_partition_label(data_root, path) in years
    )


def _parquet_filters_for_time_window(
    *,
    timestamp_column: str | None,
    time_window: TimeWindow | None,
) -> list[tuple[str, str, Any]] | None:
    if time_window is None:
        return None
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    start = pd.Timestamp(time_window.start)
    exclusive_end = pd.Timestamp(time_window.end) + pd.Timedelta(days=1)
    return [
        (timestamp_column, ">=", start),
        (timestamp_column, "<", exclusive_end),
    ]


def _resolve_columns(
    inspection: DatasetInspection,
    columns: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if columns is None:
        requested = tuple(str(column) for column in inspection.spec["required_columns"])
    else:
        requested = tuple(str(column) for column in columns)
    missing = [column for column in requested if column not in inspection.columns]
    if missing:
        raise ValueError("dataset_column_not_found")
    return requested


def _resolve_columns_from_schema(
    schema_columns: tuple[str, ...],
    *,
    spec: Mapping[str, Any],
    columns: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if columns is None:
        requested = tuple(str(column) for column in spec["required_columns"])
    else:
        requested = tuple(str(column) for column in columns)
    missing = [column for column in requested if column not in schema_columns]
    if missing:
        raise ValueError("dataset_column_not_found")
    return requested


def _read_columns_for_time_window(
    columns: tuple[str, ...],
    *,
    spec: Mapping[str, Any],
    time_window: TimeWindow | None,
) -> tuple[str, ...]:
    if time_window is None:
        return columns
    timestamp_column = _timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    return tuple(dict.fromkeys((*columns, timestamp_column)))


def _add_partition_columns(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
) -> pd.DataFrame:
    if timestamp_column not in frame:
        raise ValueError("missing_partition_timestamp_column")
    result = frame.copy()
    result["year"] = pd.to_datetime(
        result[timestamp_column],
        errors="raise",
    ).dt.year.astype("int64")
    return result


def _partition_frames(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None,
    partition_columns: tuple[str, ...],
) -> Iterator[tuple[str, pd.DataFrame]]:
    if frame.empty:
        return
    if partition_columns == ():
        yield "", frame
        return
    if partition_columns != ("year",):
        raise ValueError("unsupported_partitioning")
    if timestamp_column is None:
        raise ValueError("missing_partition_timestamp_column")
    partitioned = _add_partition_columns(frame, timestamp_column=timestamp_column)
    for year, year_frame in partitioned.groupby("year", sort=True):
        yield f"year={int(year)}", year_frame.drop(columns=["year"])


def _partition_part_path(root: Path, partition: str, part_index: int) -> Path:
    if partition:
        return root / partition / f"part-{part_index:03d}.parquet"
    return root / f"part-{part_index:03d}.parquet"


def _raise_accumulated_failures(accumulator: _WriteStats) -> None:
    if accumulator.duplicate_primary_key_count:
        raise ValueError("duplicate_dataset_primary_key")
    if any(int(count) > 0 for count in accumulator.null_count_summary.values()):
        raise ValueError("dataset_primary_key_null")
    if any(int(count) > 0 for count in accumulator.non_finite_count_summary.values()):
        raise ValueError("dataset_infinite_value")


def _validate_written_artifact(root: Path, accumulator: _WriteStats) -> None:
    spec_path = root / "spec.json"
    data_root = root / "data"
    manifest_path = root / "_manifest" / "manifest.json"
    if not spec_path.is_file():
        raise ValueError("dataset_spec_missing")
    if not data_root.is_dir():
        raise ValueError("dataset_data_dir_missing")
    if not manifest_path.is_file():
        raise ValueError("dataset_manifest_missing")
    spec_payload = read_manifest(spec_path)
    manifest = read_manifest(manifest_path)
    if not _manifest_has_runtime_metadata(manifest):
        raise ValueError("manifest_runtime_metadata_missing")
    paths = _candidate_data_files(data_root, spec=spec_payload)
    if len(paths) != accumulator.file_count:
        raise ValueError("written_file_count_mismatch")
    actual_partitions = sorted(
        {_relative_partition_label(data_root, path) for path in paths}
    )
    if actual_partitions != sorted(accumulator.partition_row_counts):
        raise ValueError("written_partition_mismatch")
    _validate_manifest_matches_accumulator(manifest, accumulator)


def _validate_manifest_matches_accumulator(
    manifest: Mapping[str, Any],
    accumulator: _WriteStats,
) -> None:
    if int(manifest["row_count"]) != int(accumulator.row_count):
        raise ValueError("manifest_row_count_mismatch")
    if int(manifest["file_count"]) != int(accumulator.file_count):
        raise ValueError("manifest_file_count_mismatch")
    if tuple(str(column) for column in manifest["columns"]) != accumulator.columns:
        raise ValueError("manifest_columns_mismatch")
    if str(manifest["schema_hash"]) != accumulator.schema_hash:
        raise ValueError("manifest_schema_hash_mismatch")
    if _manifest_partition_row_counts(manifest) != accumulator.partition_row_counts:
        raise ValueError("manifest_partition_row_counts_mismatch")
    if sum(_manifest_partition_row_counts(manifest).values()) != accumulator.row_count:
        raise ValueError("manifest_row_count_mismatch")


def _manifest_has_runtime_metadata(manifest: Mapping[str, Any]) -> bool:
    return all(
        key in manifest
        for key in (
            "columns",
            "schema_hash",
            "partition_row_counts",
            "file_count",
            "row_count",
        )
    )


def _validate_manifest_runtime_metadata(
    manifest: Mapping[str, Any],
    *,
    data_root: Path,
    paths: list[Path],
    validate_row_counts: bool,
) -> None:
    manifest_file_count, _, partition_row_counts = (
        _validate_manifest_runtime_metadata_shape(manifest)
    )
    if manifest_file_count != len(paths):
        raise ValueError("manifest_file_count_mismatch")
    actual_partitions = {_relative_partition_label(data_root, path) for path in paths}
    if set(partition_row_counts) != actual_partitions:
        raise ValueError("manifest_partition_row_counts_mismatch")
    if validate_row_counts:
        actual_row_counts = _parquet_row_counts_by_partition(data_root, paths)
        if actual_row_counts != partition_row_counts:
            raise ValueError("manifest_partition_row_counts_mismatch")


def _validate_manifest_runtime_metadata_shape(
    manifest: Mapping[str, Any],
) -> tuple[int, int, dict[str, int]]:
    columns = manifest.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(column, str) for column in columns)
    ):
        raise ValueError("manifest_columns_invalid")
    schema_hash = manifest.get("schema_hash")
    if not isinstance(schema_hash, str) or not schema_hash:
        raise ValueError("manifest_schema_hash_invalid")
    try:
        manifest_file_count = int(manifest["file_count"])
        manifest_row_count = int(manifest["row_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest_counts_invalid") from exc
    if manifest_file_count < 0 or manifest_row_count < 0:
        raise ValueError("manifest_counts_invalid")
    partition_row_counts = _manifest_partition_row_counts(manifest)
    if sum(partition_row_counts.values()) != manifest_row_count:
        raise ValueError("manifest_row_count_mismatch")
    return manifest_file_count, manifest_row_count, partition_row_counts


def _manifest_partition_row_counts(manifest: Mapping[str, Any]) -> dict[str, int]:
    value = manifest.get("partition_row_counts")
    if not isinstance(value, dict) or not value:
        raise ValueError("manifest_partition_row_counts_invalid")
    counts: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("manifest_partition_row_counts_invalid")
        try:
            count = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError("manifest_partition_row_counts_invalid") from exc
        if count < 0:
            raise ValueError("manifest_partition_row_counts_invalid")
        counts[key] = count
    return dict(sorted(counts.items()))


def _parquet_row_counts_by_partition(
    data_root: Path,
    paths: list[Path],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in paths:
        partition = _relative_partition_label(data_root, path)
        row_count = int(pq.ParquetFile(path).metadata.num_rows)
        counts[partition] = counts.get(partition, 0) + row_count
    return dict(sorted(counts.items()))


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


def _validate_dataset_artifact_root(root: Path, spec: DataSpec) -> None:
    if not is_local_only_path(root):
        raise ValueError("outside_local_only_boundary")
    path_parts = _path_parts(root)
    if isinstance(spec, InterimDataSpec):
        if (
            _unique_matching_suffix(
                path_parts,
                lambda candidate: _is_interim_output_suffix(candidate, spec),
            )
            is None
        ):
            raise ValueError("invalid_interim_output_root")
        return
    if isinstance(spec, ProcessedDataSpec):
        if (
            _unique_matching_suffix(
                path_parts,
                lambda candidate: _is_processed_output_suffix(candidate, spec),
            )
            is None
        ):
            raise ValueError("invalid_processed_output_root")
        return
    if isinstance(spec, ResearchDatasetSpec):
        if (
            _unique_matching_suffix(
                path_parts,
                lambda candidate: _is_research_dataset_output_suffix(candidate, spec),
            )
            is None
        ):
            raise ValueError("invalid_research_dataset_output_root")
        return
    raise ValueError("unsupported_data_spec_type")


def _require_artifact_root(root: Path) -> Path:
    candidate = Path(root)
    if candidate.name == "data":
        raise ValueError("dataset_root_must_be_artifact_root")
    return candidate


def _path_parts(path: Path) -> tuple[str, ...]:
    return tuple(part for part in PurePosixPath(path.as_posix()).parts if part != "/")


def _unique_matching_suffix(
    parts: tuple[str, ...],
    predicate: Any,
) -> tuple[str, ...] | None:
    candidates = [parts[index:] for index, part in enumerate(parts) if part == "data"]
    matches = [candidate for candidate in candidates if predicate(candidate)]
    storage_like_invalid = [
        candidate
        for candidate in candidates
        if _starts_storage_layout(candidate) and not predicate(candidate)
    ]
    if len(matches) != 1 or storage_like_invalid:
        return None
    return matches[0]


def _starts_storage_layout(parts: tuple[str, ...]) -> bool:
    return (
        len(parts) >= 2
        and parts[0] == "data"
        and parts[1]
        in {
            "interim",
            "processed",
            "research",
        }
    )


def _is_interim_output_suffix(parts: tuple[str, ...], spec: InterimDataSpec) -> bool:
    return (
        len(parts) == 4
        and parts[:3] == ("data", "interim", str(spec.source_id))
        and _matches_artifact_segment(parts[3], spec.data_id)
    )


def _is_processed_output_suffix(
    parts: tuple[str, ...],
    spec: ProcessedDataSpec,
) -> bool:
    expected_prefix = (
        "data",
        "processed",
        str(spec.market_id),
        spec.kind.value,
    )
    return (
        len(parts) == len(expected_prefix) + 1
        and parts[: len(expected_prefix)] == expected_prefix
        and _matches_artifact_segment(
            parts[len(expected_prefix)],
            spec.data_id,
        )
    )


def _is_research_dataset_output_suffix(
    parts: tuple[str, ...],
    spec: ResearchDatasetSpec,
) -> bool:
    return (
        len(parts) == 4
        and parts[:3] == ("data", "research", str(spec.market_id))
        and _matches_artifact_segment(parts[3], spec.data_id)
    )


def _matches_artifact_segment(segment: str, expected: str) -> bool:
    return segment == expected or segment.startswith(f".{expected}.tmp-")


def _candidate_data_files(
    data_root: Path,
    *,
    spec: Mapping[str, Any],
) -> list[Path]:
    root = Path(data_root)
    partition_columns = tuple(
        str(column) for column in spec.get("partition_columns", ())
    )
    unpartitioned = sorted(root.glob("part-*.parquet"))
    if unpartitioned:
        return unpartitioned
    if partition_columns == ():
        return []
    if partition_columns != ("year",):
        raise ValueError("unsupported_partitioning")
    available_years = {
        path.name.split("=", 1)[1]
        for path in root.glob("year=*")
        if path.is_dir() and list(path.glob("*.parquet"))
    }
    paths: list[Path] = []
    for year in sorted(available_years, key=lambda value: int(value)):
        paths.extend(sorted((root / f"year={year}").glob("*.parquet")))
    return paths


def _validated_schema(paths: list[Path]) -> tuple[tuple[str, str], ...]:
    if not paths:
        raise ValueError("dataset_data_dir_missing")
    schemas = [_canonical_schema(path) for path in paths]
    first = schemas[0]
    if any(schema != first for schema in schemas[1:]):
        raise ValueError("dataset_schema_mismatch")
    return first


def _canonical_schema(path: Path) -> tuple[tuple[str, str], ...]:
    schema = pq.ParquetFile(path).schema_arrow
    return _canonical_schema_from_schema(schema)


def _canonical_schema_from_file(
    parquet_file: pq.ParquetFile,
) -> tuple[tuple[str, str], ...]:
    return _canonical_schema_from_schema(parquet_file.schema_arrow)


def _canonical_schema_from_schema(schema: Any) -> tuple[tuple[str, str], ...]:
    return tuple((field.name, str(field.type)) for field in schema)


def _write_parquet_file(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


def _timestamp_primary_key_column_from_spec(spec: DataSpec) -> str | None:
    timestamp_columns = [
        column
        for column in spec.primary_key
        if spec.column_roles.get(column) is ColumnRole.TIMESTAMP
    ]
    return timestamp_columns[0] if timestamp_columns else None


def _timestamp_column_from_spec_payload(spec: Mapping[str, Any]) -> str | None:
    roles = spec.get("column_roles", {})
    if not isinstance(roles, Mapping):
        raise ValueError("invalid_column_roles")
    primary_key = tuple(str(column) for column in spec.get("primary_key", ()))
    timestamp_columns = [
        column
        for column in primary_key
        if str(roles.get(column)) == ColumnRole.TIMESTAMP.value
    ]
    if len(timestamp_columns) > 1:
        raise ValueError("invalid_timestamp_primary_key")
    return timestamp_columns[0] if timestamp_columns else None


def _relative_partition_label(data_root: Path, path: Path) -> str:
    relative_parent = path.parent.relative_to(data_root)
    label = relative_parent.as_posix()
    return "unpartitioned" if label == "." else label


def _partition_keys(fingerprint: DatasetFingerprint) -> tuple[str, ...]:
    return tuple(str(item["partition"]) for item in fingerprint.partition_fingerprints)


def _data_spec_payload(spec: DataSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": spec.schema_version,
        "data_id": spec.data_id,
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
        payload.update({"source_id": str(spec.source_id)})
    if isinstance(spec, ProcessedDataSpec):
        payload.update(
            {
                "market_id": str(spec.market_id),
                "kind": spec.kind.value,
                "source_input_data_ids": list(spec.source_input_data_ids),
                "processed_input_data_ids": list(spec.processed_input_data_ids),
            }
        )
    if isinstance(spec, ResearchDatasetSpec):
        payload.update(
            {
                "market_id": str(spec.market_id),
                "input_data_ids": list(spec.input_data_ids),
                "base_input_data_id": spec.base_input_data_id,
                "factor_columns": list(spec.factor_columns),
                "feature_columns": list(spec.feature_columns),
                "label_columns": list(spec.label_columns),
                "filter_columns": list(spec.filter_columns),
                "context_columns": list(spec.context_columns),
                "key_coverage": dict(spec.key_coverage),
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
    "DatasetReadMode",
    "DatasetInspection",
    "DatasetFingerprint",
    "DatasetComparison",
    "write_dataset",
    "read_dataset",
    "inspect_dataset",
    "fingerprint_dataset",
    "compare_datasets",
]
