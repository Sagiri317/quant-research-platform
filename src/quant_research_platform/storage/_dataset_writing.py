"""Dataset writing implementation."""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.storage._dataset_key_tracking import (
    ExactPrimaryKeyIndex,
)
from quant_research_platform.storage._dataset_manifest_validation import (
    validate_written_artifact,
)
from quant_research_platform.storage._dataset_paths import (
    partition_frames,
    partition_part_path,
    relative_partition_label,
    timestamp_primary_key_column_from_spec,
    validate_dataset_artifact_root,
)
from quant_research_platform.storage._dataset_schema import (
    canonical_schema_from_file,
    data_spec_payload,
    date_string,
    merge_dataset_instance_metadata,
    sha256_json,
    validate_dataset_spec,
)
from quant_research_platform.storage._scratch_ownership import (
    owned_scratch_name,
    scratch_owner_is_stale,
)
from quant_research_platform.storage.atomic import publish_no_replace
from quant_research_platform.storage.manifests import write_internal_artifact_manifest


@dataclass
class _WriteStats:
    spec: DataSpec
    data_root: Path
    key_tracker: ExactPrimaryKeyIndex
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
    _schema: tuple[tuple[str, str], ...] | None = None
    _arrow_schema: pa.Schema | None = None
    _ordered_categorical_domains: dict[str, pd.Index] = field(default_factory=dict)

    def update_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        unregistered = sorted(set(frame.columns).difference(self.spec.column_roles))
        if unregistered:
            raise ValueError(f"unregistered_output_columns: {', '.join(unregistered)}")
        missing = sorted(set(self.spec.required_columns).difference(frame.columns))
        if missing:
            raise ValueError(f"missing_required_columns: {', '.join(missing)}")
        output = frame.loc[:, list(self.spec.required_columns)].copy()
        for column in self.spec.primary_key:
            if self.spec.column_roles.get(column) is ColumnRole.TIMESTAMP:
                output[column] = strict_timestamp_series(
                    output[column],
                    error="invalid_dataset_timestamp_value",
                    allow_null=True,
                )
        table = pa.Table.from_pandas(output, preserve_index=False)
        if self._arrow_schema is None:
            self._arrow_schema = table.schema.remove_metadata()
        else:
            try:
                table = table.cast(self._arrow_schema, safe=True)
            except (
                pa.ArrowInvalid,
                pa.ArrowNotImplementedError,
                pa.ArrowTypeError,
            ) as exc:
                raise ValueError("dataset_schema_mismatch") from exc
        output = table.to_pandas(use_threads=False)
        self._validate_ordered_categorical_domains(output)
        self.row_count += int(len(output))
        self._update_keys(output, table=table)
        self._update_dates(output)
        self._update_numeric_profiles(output)
        _raise_accumulated_failures(self)
        return output

    def _validate_ordered_categorical_domains(self, frame: pd.DataFrame) -> None:
        for column in frame.columns:
            dtype = frame[column].dtype
            if not isinstance(dtype, pd.CategoricalDtype) or not dtype.ordered:
                continue
            categories = frame[column].cat.categories
            expected = self._ordered_categorical_domains.get(column)
            if expected is None:
                self._ordered_categorical_domains[column] = categories.copy()
            elif not categories.equals(expected):
                raise ValueError("dataset_ordered_categorical_mismatch")

    def record_file(self, path: Path, *, row_count: int) -> None:
        parquet_file = pq.ParquetFile(path)
        actual_row_count = int(parquet_file.metadata.num_rows)
        if actual_row_count != int(row_count):
            raise ValueError("dataset_row_count_mismatch")
        schema = canonical_schema_from_file(parquet_file)
        if self._schema is None:
            self._schema = schema
            self.columns = tuple(name for name, _ in schema)
            self.schema_hash = sha256_json(schema)
        elif schema != self._schema:
            raise ValueError("dataset_schema_mismatch")
        partition = relative_partition_label(self.data_root, path)
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
        payload = data_spec_payload(self.spec)
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
                "date_min": date_string(self.date_min),
                "date_max": date_string(self.date_max),
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

    def _update_keys(self, frame: pd.DataFrame, *, table: pa.Table) -> None:
        has_null_key = False
        for column in self.spec.primary_key:
            self.null_count_summary.setdefault(column, 0)
            null_count = int(frame[column].isna().sum())
            self.null_count_summary[column] += null_count
            has_null_key = has_null_key or null_count > 0
        if frame.empty or has_null_key:
            return
        self.duplicate_primary_key_count += self.key_tracker.add_table(
            table.select(self.spec.primary_key)
        )

    def _update_dates(self, frame: pd.DataFrame) -> None:
        timestamp_column = timestamp_primary_key_column_from_spec(self.spec)
        if timestamp_column is None or timestamp_column not in frame or frame.empty:
            return
        values = strict_timestamp_series(
            frame[timestamp_column],
            error="invalid_dataset_timestamp_value",
            allow_null=True,
        ).dropna()
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
    validate_dataset_spec(spec)
    validate_dataset_artifact_root(final_root, spec)

    def _writer(temp_root: Path) -> None:
        accumulator = _write_contents(
            _iter_input_chunks(data),
            temp_root / "data",
            spec=spec,
        )
        write_internal_artifact_manifest(
            temp_root / "spec.json",
            data_spec_payload(spec),
        )
        write_internal_artifact_manifest(
            temp_root / "_manifest" / "manifest.json",
            merge_dataset_instance_metadata(
                {
                    **accumulator.payload(),
                    "schema_id": "qrp-dataset-manifest",
                },
                instance_metadata,
            ),
        )
        validate_written_artifact(temp_root, accumulator)

    _staged_write_directory(final_root, _writer)


def _staged_write_directory(
    final_root: Path,
    writer: Any,
) -> None:
    final_root.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_staged_scratch(final_root)
    if final_root.exists():
        raise ValueError("dataset_exists")
    scratch_prefix = f".{final_root.name}.tmp-"
    temp_root = final_root.with_name(
        owned_scratch_name(scratch_prefix, uuid4().hex[:6])
    )
    claimed_temp = False
    try:
        temp_root.mkdir()
        claimed_temp = True
        writer(temp_root)
        if final_root.exists():
            raise ValueError("dataset_exists")
        publish_no_replace(temp_root, final_root)
    except Exception:
        if claimed_temp:
            _remove_tree(temp_root)
        raise


def _remove_tree(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _cleanup_stale_staged_scratch(final_root: Path) -> None:
    scratch_prefix = f".{final_root.name}.tmp-"
    for staged_root in final_root.parent.glob(f"{scratch_prefix}*"):
        if not staged_root.is_dir():
            continue
        if not scratch_owner_is_stale(staged_root, prefix=scratch_prefix):
            continue
        _remove_tree(staged_root)


def _write_contents(
    chunks: Iterator[pd.DataFrame],
    root: Path,
    *,
    spec: DataSpec,
) -> _WriteStats:
    timestamp_column = validate_dataset_spec(spec)
    timestamp_columns = tuple(
        column
        for column in spec.primary_key
        if spec.column_roles.get(column) is ColumnRole.TIMESTAMP
    )
    with ExactPrimaryKeyIndex(
        primary_key=spec.primary_key,
        timestamp_columns=timestamp_columns,
        scratch_parent=root.parent,
    ) as key_tracker:
        accumulator = _WriteStats(
            spec=spec,
            data_root=root,
            key_tracker=key_tracker,
        )
        part_indices: dict[str, int] = {}
        empty_schema_frame: pd.DataFrame | None = None
        for chunk in chunks:
            frame = accumulator.update_frame(chunk)
            if frame.empty:
                if empty_schema_frame is None:
                    empty_schema_frame = frame
                continue
            for partition, partition_frame in partition_frames(
                frame,
                timestamp_column=timestamp_column,
                partition_columns=spec.partition_columns,
            ):
                part_index = part_indices.get(partition, 0)
                part_indices[partition] = part_index + 1
                output_path = partition_part_path(root, partition, part_index)
                write_parquet_file(
                    partition_frame,
                    output_path,
                    schema=accumulator._arrow_schema,
                )
                accumulator.record_file(output_path, row_count=len(partition_frame))
        if accumulator.row_count == 0:
            if empty_schema_frame is None:
                raise ValueError("empty_dataset_schema_missing")
            output_path = partition_part_path(root, "", 0)
            write_parquet_file(
                empty_schema_frame,
                output_path,
                schema=accumulator._arrow_schema,
            )
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


def _raise_accumulated_failures(accumulator: _WriteStats) -> None:
    if accumulator.duplicate_primary_key_count:
        raise ValueError("duplicate_dataset_primary_key")
    if any(int(count) > 0 for count in accumulator.null_count_summary.values()):
        raise ValueError("dataset_primary_key_null")
    if any(int(count) > 0 for count in accumulator.non_finite_count_summary.values()):
        raise ValueError("dataset_infinite_value")


def write_parquet_file(
    frame: pd.DataFrame,
    path: Path,
    *,
    schema: pa.Schema | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if schema is not None:
        table = table.cast(schema, safe=True)
    pq.write_table(table, path, compression="zstd")
    return path
