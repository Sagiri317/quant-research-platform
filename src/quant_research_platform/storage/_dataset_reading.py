"""Dataset reading implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from quant_research_platform.contracts._json import (
    canonical_json_hash,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_date_bounds,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage._dataset_manifest_validation import (
    manifest_has_runtime_metadata,
    validate_manifest_runtime_metadata,
)
from quant_research_platform.storage._dataset_paths import (
    candidate_data_files,
    relative_partition_label,
    require_artifact_root,
    timestamp_column_from_spec_payload,
)
from quant_research_platform.storage._dataset_physical import (
    DatasetFileIdentity,
    DatasetPartMetadata,
    dataset_file_identity,
    inspect_dataset_part,
    iter_dataset_part_batches,
)
from quant_research_platform.storage._dataset_schema import (
    logical_dtype_metadata_bytes,
    merge_logical_dtype_payloads,
    pandas_dtype_overrides_from_logical_payload,
    parse_logical_dtype_metadata,
    physical_schema_columns,
    physical_schema_from_schema,
    validate_dataset_manifest_consumable,
    validate_dataset_manifest_payload,
    validate_dataset_spec_payload,
)
from quant_research_platform.storage._dataset_types import (
    DatasetReadMode,
)


@dataclass(frozen=True, slots=True)
class DatasetSession:
    root: Path
    data_root: Path
    spec_bytes: bytes
    manifest_bytes: bytes
    all_paths: tuple[Path, ...]
    read_paths: tuple[Path, ...]
    logical_columns: tuple[str, ...]
    physical_columns: tuple[str, ...]
    timestamp_column: str | None
    time_window: TimeWindow | None
    schema_hash: str
    arrow_schema: pa.Schema
    logical_dtype_bytes: bytes
    part_metadata: tuple[tuple[Path, DatasetPartMetadata], ...]
    parquet_filters: tuple[tuple[str, str, Any], ...] | None

    @property
    def spec(self) -> dict[str, Any]:
        return parse_qrp_json_object(self.spec_bytes)

    @property
    def manifest(self) -> dict[str, Any]:
        return parse_qrp_json_object(self.manifest_bytes)

    @property
    def schema(self) -> list[dict[str, Any]]:
        return list(self.manifest["physical_schema"])

    @property
    def logical_dtypes(self) -> dict[str, Any]:
        return parse_logical_dtype_metadata(self.logical_dtype_bytes)

    def metadata_for(self, path: Path) -> DatasetPartMetadata:
        for candidate, metadata in self.part_metadata:
            if candidate == path:
                return metadata
        raise ValueError("dataset_part_not_in_session")

    def verify(self) -> None:
        if (self.root / "spec.json").read_bytes() != self.spec_bytes:
            raise ValueError("dataset_spec_identity_changed")
        if (
            self.root / "_manifest" / "manifest.json"
        ).read_bytes() != self.manifest_bytes:
            raise ValueError("dataset_manifest_identity_changed")
        current_paths = tuple(candidate_data_files(self.data_root, spec=self.spec))
        if current_paths != self.all_paths:
            raise ValueError("dataset_inventory_changed")
        for path, metadata in self.part_metadata:
            if dataset_file_identity(path) != metadata.identity:
                raise ValueError("dataset_file_identity_changed")


def read_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
    mode: DatasetReadMode = DatasetReadMode.MATERIALIZED,
    resource_policy: RuntimeResourcePolicy,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    context = open_dataset_session(
        dataset_root,
        columns=columns,
        time_window=time_window,
    )
    target_chunk_bytes = resource_policy.payload_target_bytes

    if mode is DatasetReadMode.CHUNKED:
        return iter_dataset_chunks(
            context,
            target_chunk_bytes=target_chunk_bytes,
        )
    if mode is not DatasetReadMode.MATERIALIZED:
        raise ValueError("unsupported_dataset_read_mode")
    return materialize_dataset_session(
        context,
        target_chunk_bytes=target_chunk_bytes,
        materialized_budget_bytes=resource_policy.task_memory_budget_bytes,
    )


def materialize_dataset_session(
    context: DatasetSession,
    *,
    target_chunk_bytes: int,
    materialized_budget_bytes: int | None = None,
) -> pd.DataFrame:
    """Materialize one validated context through the unique physical iterator."""

    frames: list[pd.DataFrame] = []
    observed_frame_bytes = 0
    materialized_budget = (
        None
        if materialized_budget_bytes is None
        else _validated_target_chunk_bytes(materialized_budget_bytes)
    )
    for frame in iter_dataset_chunks(
        context,
        target_chunk_bytes=target_chunk_bytes,
    ):
        observed_frame_bytes += pandas_frame_payload_bytes(frame)
        if (
            materialized_budget is not None
            and observed_frame_bytes > materialized_budget
        ):
            raise ValueError("dataset_materialized_result_exceeds_target_bytes")
        frames.append(frame)
    context.verify()
    output = pd.concat(frames, ignore_index=True)
    output_bytes = pandas_frame_payload_bytes(output)
    if (
        materialized_budget is not None
        and observed_frame_bytes + output_bytes > materialized_budget
    ):
        raise ValueError("dataset_materialized_result_exceeds_target_bytes")
    del frames
    return output


def open_dataset_session(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
    row_filters: tuple[tuple[str, str, Any], ...] = (),
    require_consumable: bool = True,
) -> DatasetSession:
    root = require_artifact_root(Path(dataset_root))
    spec_path = root / "spec.json"
    data_root = root / "data"
    manifest_path = root / "_manifest" / "manifest.json"
    if not spec_path.is_file():
        raise ValueError("dataset_spec_missing")
    if not data_root.is_dir():
        raise ValueError("dataset_data_dir_missing")
    spec_bytes = spec_path.read_bytes()
    spec_payload = parse_qrp_json_object(spec_bytes)
    if spec_payload.get("schema_id") != "qrp-dataset-spec":
        raise ValueError("dataset_spec_schema_id_invalid")
    validate_dataset_spec_payload(spec_payload)
    if not manifest_path.is_file():
        raise ValueError("dataset_manifest_missing")
    manifest_bytes = manifest_path.read_bytes()
    manifest = parse_qrp_json_object(manifest_bytes)
    if manifest.get("schema_id") != "qrp-dataset-manifest":
        raise ValueError("dataset_manifest_schema_id_invalid")
    validate_dataset_manifest_payload(manifest, spec=spec_payload)
    if require_consumable:
        validate_dataset_manifest_consumable(manifest)
    if not manifest_has_runtime_metadata(manifest):
        raise ValueError("manifest_runtime_metadata_missing")
    all_paths = tuple(candidate_data_files(data_root, spec=spec_payload))
    if not all_paths:
        raise ValueError("dataset_data_dir_missing")
    metadata_items = tuple(
        (path, inspect_dataset_part(path)) for path in all_paths
    )
    return _build_dataset_session(
        root=root,
        data_root=data_root,
        spec_bytes=spec_bytes,
        manifest_bytes=manifest_bytes,
        all_paths=all_paths,
        metadata_items=metadata_items,
        columns=columns,
        time_window=time_window,
        row_filters=row_filters,
        require_consumable=require_consumable,
    )


def _dataset_session_from_observed_parts(
    dataset_root: Path,
    *,
    metadata_items: tuple[tuple[Path, DatasetPartMetadata], ...],
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
) -> DatasetSession:
    """Freeze a writer receipt without reopening already observed Parquet parts."""

    root = require_artifact_root(Path(dataset_root))
    data_root = root / "data"
    spec_bytes = (root / "spec.json").read_bytes()
    manifest_bytes = (root / "_manifest" / "manifest.json").read_bytes()
    spec_payload = parse_qrp_json_object(spec_bytes)
    all_paths = tuple(candidate_data_files(data_root, spec=spec_payload))
    observed_by_path = dict(metadata_items)
    if len(observed_by_path) != len(metadata_items) or set(observed_by_path) != set(
        all_paths
    ):
        raise ValueError("dataset_inventory_changed")
    ordered_metadata = tuple((path, observed_by_path[path]) for path in all_paths)
    return _build_dataset_session(
        root=root,
        data_root=data_root,
        spec_bytes=spec_bytes,
        manifest_bytes=manifest_bytes,
        all_paths=all_paths,
        metadata_items=ordered_metadata,
        columns=columns,
        time_window=time_window,
        row_filters=(),
        require_consumable=True,
    )


def _build_dataset_session(
    *,
    root: Path,
    data_root: Path,
    spec_bytes: bytes,
    manifest_bytes: bytes,
    all_paths: tuple[Path, ...],
    metadata_items: tuple[tuple[Path, DatasetPartMetadata], ...],
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
    row_filters: tuple[tuple[str, str, Any], ...],
    require_consumable: bool,
) -> DatasetSession:
    spec_payload = parse_qrp_json_object(spec_bytes)
    if spec_payload.get("schema_id") != "qrp-dataset-spec":
        raise ValueError("dataset_spec_schema_id_invalid")
    validate_dataset_spec_payload(spec_payload)
    manifest = parse_qrp_json_object(manifest_bytes)
    if manifest.get("schema_id") != "qrp-dataset-manifest":
        raise ValueError("dataset_manifest_schema_id_invalid")
    validate_dataset_manifest_payload(manifest, spec=spec_payload)
    if require_consumable:
        validate_dataset_manifest_consumable(manifest)
    if not manifest_has_runtime_metadata(manifest):
        raise ValueError("manifest_runtime_metadata_missing")
    if not all_paths or not metadata_items:
        raise ValueError("dataset_data_dir_missing")
    first_metadata = metadata_items[0][1]
    schema = physical_schema_from_schema(first_metadata.schema)
    for _, metadata in metadata_items[1:]:
        if physical_schema_from_schema(metadata.schema) != schema:
            raise ValueError("dataset_schema_mismatch")
    logical_dtypes = merge_logical_dtype_payloads(
        [
            parse_logical_dtype_metadata(metadata.logical_dtype_payload)
            for _, metadata in metadata_items
        ]
    )
    schema_columns = physical_schema_columns(schema)
    schema_hash = canonical_json_hash(schema)
    validate_manifest_runtime_metadata(
        manifest,
        data_root=data_root,
        paths=list(all_paths),
    )
    observed_rows = sum(metadata.row_count for _, metadata in metadata_items)
    if int(manifest["row_count"]) != observed_rows:
        raise ValueError("manifest_row_count_mismatch")
    observed_partition_rows: dict[str, int] = {}
    for path, metadata in metadata_items:
        partition = relative_partition_label(data_root, path)
        observed_partition_rows[partition] = (
            observed_partition_rows.get(partition, 0) + metadata.row_count
        )
    if dict(sorted(observed_partition_rows.items())) != dict(
        sorted(manifest["partition_row_counts"].items())
    ):
        raise ValueError("manifest_partition_row_counts_mismatch")
    if int(manifest["file_size_bytes"]) != sum(
        metadata.file_size_bytes for _, metadata in metadata_items
    ):
        raise ValueError("manifest_file_size_bytes_mismatch")
    if manifest["logical_dtypes"] != logical_dtypes:
        raise ValueError("manifest_logical_dtypes_mismatch")
    if tuple(str(column) for column in spec_payload["partition_columns"]) == (
        "year",
    ) and any(
        path.parent == data_root and metadata.row_count != 0
        for path, metadata in metadata_items
    ):
        raise ValueError("invalid_dataset_part_files")
    if tuple(str(column) for column in manifest["columns"]) != schema_columns:
        raise ValueError("manifest_columns_mismatch")
    if manifest["physical_schema"] != schema:
        raise ValueError("manifest_physical_schema_mismatch")
    if str(manifest["schema_hash"]) != schema_hash:
        raise ValueError("manifest_schema_hash_mismatch")
    logical_columns = resolve_columns_from_schema(
        schema_columns,
        spec=spec_payload,
        columns=columns,
    )
    timestamp_column = timestamp_column_from_spec_payload(spec_payload)
    physical_columns = read_columns_for_time_window(
        logical_columns,
        spec=spec_payload,
        time_window=time_window,
    )
    physical_filters = parquet_filters_for_time_window(
        timestamp_column=timestamp_column,
        time_window=time_window,
    ) or []
    for column, operator, value in row_filters:
        if column not in schema_columns:
            raise ValueError("dataset_filter_column_not_found")
        if operator != "==":
            raise ValueError("unsupported_dataset_physical_filter")
        physical_filters.append((column, operator, value))
    physical_columns = tuple(
        dict.fromkeys(
            (*physical_columns, *(column for column, _, _ in physical_filters))
        )
    )
    session = DatasetSession(
        root=root,
        data_root=data_root,
        spec_bytes=spec_bytes,
        manifest_bytes=manifest_bytes,
        all_paths=all_paths,
        read_paths=tuple(
            prune_paths_for_time_window(
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
        schema_hash=schema_hash,
        arrow_schema=first_metadata.schema,
        logical_dtype_bytes=logical_dtype_metadata_bytes(logical_dtypes),
        part_metadata=metadata_items,
        parquet_filters=tuple(
            (column, operator, _freeze_filter_value(value))
            for column, operator, value in physical_filters
        )
        or None,
    )
    session.verify()
    return session


def iter_dataset_chunks(
    context: DatasetSession,
    *,
    target_chunk_bytes: int,
) -> Iterator[pd.DataFrame]:
    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    try:
        if not context.read_paths:
            yield _empty_frame(
                context,
                context.logical_columns,
                target_chunk_bytes=target_bytes,
            )
            return
        yielded = False
        for _, frame in iter_dataset_file_batches(
            context,
            columns=context.physical_columns,
            target_chunk_bytes=target_bytes,
        ):
            filtered = frame.loc[:, list(context.logical_columns)]
            for chunk in _bounded_frame_chunks(
                filtered,
                target_chunk_bytes=target_bytes,
            ):
                yielded = True
                yield chunk
        if not yielded:
            yield _empty_frame(
                context,
                context.logical_columns,
                target_chunk_bytes=target_bytes,
            )
    finally:
        context.verify()


def iter_declared_parquet_member_batches(
    path: Path,
    *,
    target_chunk_bytes: int,
    expected_identity: DatasetFileIdentity,
    columns: tuple[str, ...] | None = None,
    parquet_filters: list[tuple[str, str, Any]] | None = None,
    dtype_overrides: Mapping[str, object] | None = None,
    metadata_observer: Callable[[DatasetPartMetadata], None] | None = None,
    persisted_frame_observer: Callable[[pd.DataFrame], None] | None = None,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[pd.DataFrame]:
    """Traverse one frozen Parquet member through the sole Pandas converter."""

    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    for batch in iter_dataset_part_batches(
        Path(path),
        expected_identity=expected_identity,
        target_chunk_bytes=target_bytes,
        columns=columns,
        parquet_filters=parquet_filters,
        metadata_observer=metadata_observer,
    ):
        yield from _arrow_batch_to_bounded_pandas(
            batch,
            target_chunk_bytes=target_bytes,
            dtype_overrides=dtype_overrides,
            persisted_frame_observer=persisted_frame_observer,
            batch_observer=batch_observer,
        )


def iter_dataset_file_batches(
    context: DatasetSession,
    *,
    columns: tuple[str, ...],
    target_chunk_bytes: int,
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """Yield deterministic file batches after Arrow-level bounded conversion."""

    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    dtype_overrides = pandas_dtype_overrides_from_logical_payload(
        context.logical_dtypes,
        columns=columns,
    )
    context.verify()
    try:
        for path in context.read_paths:
            metadata = context.metadata_for(path)
            for frame in iter_declared_parquet_member_batches(
                path,
                expected_identity=metadata.identity,
                columns=columns,
                parquet_filters=context.parquet_filters,
                target_chunk_bytes=target_bytes,
                dtype_overrides=dtype_overrides,
            ):
                yield path, frame
    finally:
        context.verify()


def dataset_primary_key_schema(
    context: DatasetSession,
    primary_key: tuple[str, ...],
) -> pa.Schema:
    try:
        return pa.schema(
            [context.arrow_schema.field(column) for column in primary_key]
        )
    except KeyError as exc:
        raise ValueError("missing_dataset_primary_key_column") from exc


def _validated_target_chunk_bytes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("invalid_target_chunk_bytes")
    return value


def _freeze_filter_value(value: Any) -> Any:
    if isinstance(value, pa.Scalar):
        value = value.as_py()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(
        value,
        (
            str,
            bytes,
            bool,
            int,
            float,
            Decimal,
            date,
            datetime,
            time,
            timedelta,
            pd.Timestamp,
            pd.Timedelta,
        ),
    ):
        return value
    raise ValueError("invalid_dataset_physical_filter_value")


def _empty_frame(
    context: DatasetSession,
    columns: tuple[str, ...],
    *,
    target_chunk_bytes: int,
) -> pd.DataFrame:
    selected = pa.schema([context.arrow_schema.field(column) for column in columns])
    overrides = pandas_dtype_overrides_from_logical_payload(
        context.logical_dtypes,
        columns=columns,
    )
    frame = _arrow_slice_to_pandas(
        pa.RecordBatch.from_arrays(
            [pa.array([], type=field.type) for field in selected],
            schema=selected,
        )
    )
    output = frame.astype(overrides) if overrides else frame
    if pandas_frame_payload_bytes(frame) + (
        pandas_frame_payload_bytes(output) if output is not frame else 0
    ) > target_chunk_bytes:
        raise ValueError("dataset_row_exceeds_target_chunk_bytes")
    return output


def _arrow_batch_to_bounded_pandas(
    batch: pa.RecordBatch,
    *,
    target_chunk_bytes: int,
    dtype_overrides: Mapping[str, object] | None = None,
    persisted_frame_observer: Callable[[pd.DataFrame], None] | None = None,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[pd.DataFrame]:
    if batch.num_rows == 0:
        return
    pandas_budget = target_chunk_bytes
    overrides = dict(dtype_overrides or {})
    conversion_budget = pandas_budget
    # Arrow payload size does not predict the Pandas representation.  Probe the
    # indivisible row first, then grow only from the observed Pandas payload.
    rows_per_slice = 1
    calibrated = False
    start = 0
    while start < batch.num_rows:
        stop = min(batch.num_rows, start + rows_per_slice)
        arrow_slice = batch.slice(start, stop - start)
        frame = _arrow_slice_to_pandas(arrow_slice)
        frame_bytes = pandas_frame_payload_bytes(frame)
        if frame_bytes > conversion_budget:
            if len(frame) == 1:
                raise ValueError("dataset_row_exceeds_target_chunk_bytes")
            rows_per_slice = max(1, rows_per_slice // 2)
            continue
        if persisted_frame_observer is not None:
            persisted_frame_observer(frame)
        output = frame.astype(overrides) if overrides else frame
        output_bytes = pandas_frame_payload_bytes(output)
        if frame_bytes + output_bytes > pandas_budget and output is not frame:
            if len(frame) == 1:
                raise ValueError("dataset_row_exceeds_target_chunk_bytes")
            rows_per_slice = max(1, rows_per_slice // 2)
            continue
        measured_bytes = max(frame_bytes, output_bytes)
        measured_row_bytes = max(
            1,
            (measured_bytes + len(frame) - 1) // len(frame),
        )
        if not calibrated:
            calibrated = True
            calibrated_rows = max(
                1,
                min(
                    batch.num_rows - start,
                    conversion_budget // measured_row_bytes,
                ),
            )
            if calibrated_rows > rows_per_slice:
                del output, frame
                rows_per_slice = calibrated_rows
                continue
        if batch_observer is not None:
            batch_observer(arrow_slice, output)
        yield output
        start = stop
        if start >= batch.num_rows:
            continue
        rows_per_slice = max(
            1,
            min(
                batch.num_rows - start,
                conversion_budget // measured_row_bytes,
            ),
        )


def _arrow_slice_to_pandas(arrow_slice: pa.RecordBatch) -> pd.DataFrame:
    return arrow_slice.to_pandas(use_threads=False)


def _bounded_frame_chunks(
    frame: pd.DataFrame,
    *,
    target_chunk_bytes: int,
) -> Iterator[pd.DataFrame]:
    if frame.empty:
        return
    frame_bytes = pandas_frame_payload_bytes(frame)
    rows_per_chunk = max(
        1,
        min(
            len(frame),
            len(frame) * target_chunk_bytes // max(1, frame_bytes),
        ),
    )
    start = 0
    while start < len(frame):
        stop = min(len(frame), start + rows_per_chunk)
        candidate = frame.iloc[start:stop]
        candidate_bytes = pandas_frame_payload_bytes(candidate)
        if candidate_bytes > target_chunk_bytes:
            if len(candidate) == 1:
                raise ValueError("dataset_row_exceeds_target_chunk_bytes")
            rows_per_chunk = max(1, rows_per_chunk // 2)
            continue
        yield candidate.copy()
        start = stop
        if start < len(frame) and candidate_bytes * 2 < target_chunk_bytes:
            rows_per_chunk = min(len(frame) - start, rows_per_chunk * 2)


def filter_frame_by_time_window(
    frame: pd.DataFrame,
    *,
    spec: Mapping[str, Any],
    time_window: TimeWindow | None,
) -> pd.DataFrame:
    if time_window is None:
        return frame
    timestamp_column = timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    if timestamp_column not in frame:
        raise ValueError("time_window_timestamp_column_not_selected")
    values = strict_timestamp_series(
        frame[timestamp_column],
        error="invalid_time_window_timestamp_value",
    )
    start, end, end_inclusive = time_window_timestamp_interval(time_window)
    end_timestamp = pd.Timestamp(end)
    end_mask = values.le(end_timestamp) if end_inclusive else values.lt(end_timestamp)
    mask = values.ge(pd.Timestamp(start)) & end_mask
    return frame.loc[mask].copy()


def prune_paths_for_time_window(
    data_root: Path,
    paths: tuple[Path, ...],
    *,
    spec: Mapping[str, Any],
    time_window: TimeWindow | None,
) -> tuple[Path, ...]:
    if time_window is None:
        return paths
    timestamp_column = timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    partition_columns = tuple(
        str(column) for column in spec.get("partition_columns", ())
    )
    if partition_columns != ("year",):
        return paths
    start, end = time_window_date_bounds(time_window)
    years = {
        f"year={year}"
        for year in range(int(start.year), int(end.year) + 1)
    }
    return tuple(
        path for path in paths if relative_partition_label(data_root, path) in years
    )


def parquet_filters_for_time_window(
    *,
    timestamp_column: str | None,
    time_window: TimeWindow | None,
) -> list[tuple[str, str, Any]] | None:
    if time_window is None:
        return None
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    start, end, end_inclusive = time_window_timestamp_interval(time_window)
    upper_operator = "<=" if end_inclusive else "<"
    return [
        (timestamp_column, ">=", pd.Timestamp(start)),
        (timestamp_column, upper_operator, pd.Timestamp(end)),
    ]


def resolve_columns_from_schema(
    schema_columns: tuple[str, ...],
    *,
    spec: Mapping[str, Any],
    columns: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if columns is None:
        requested = tuple(str(column) for column in spec["required_columns"])
    else:
        requested = tuple(str(column) for column in columns)
    if not requested:
        raise ValueError("dataset_columns_empty")
    missing = [column for column in requested if column not in schema_columns]
    if missing:
        raise ValueError("dataset_column_not_found")
    return requested


def read_columns_for_time_window(
    columns: tuple[str, ...],
    *,
    spec: Mapping[str, Any],
    time_window: TimeWindow | None,
) -> tuple[str, ...]:
    if time_window is None:
        return columns
    timestamp_column = timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    return tuple(dict.fromkeys((*columns, timestamp_column)))
