"""Dataset reading implementation."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import date, time
from decimal import Decimal
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pandas.api.types import union_categoricals

from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_date_bounds,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
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
from quant_research_platform.storage._dataset_schema import (
    sha256_json,
    validate_dataset_manifest_instance_metadata,
    validated_schema,
)
from quant_research_platform.storage._dataset_types import (
    DatasetReadMode,
)
from quant_research_platform.storage.manifests import read_qrp_json

DEFAULT_DATASET_PHYSICAL_TARGET_BYTES = 64 * 1024 * 1024
_PARQUET_TO_PANDAS_EXPANSION_FACTOR = 4
_VARIABLE_WIDTH_TO_PANDAS_EXPANSION_FACTOR = 32
_PHYSICAL_WORKING_SET_PARTS = 4
_PANDAS_PROBE_ROWS = 64
_PANDAS_FRAME_OVERHEAD_BYTES = 192
_PYTHON_POINTER_BYTES = max(
    1,
    sys.getsizeof((None,)) - sys.getsizeof(()),
)


@dataclass(frozen=True)
class DatasetReadContext:
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


@dataclass(frozen=True, slots=True)
class DatasetEqualityGroupPlan:
    columns: tuple[str, ...]
    keys: tuple[tuple[Any, ...], ...]
    paths_by_key: dict[tuple[Any, ...], tuple[Path, ...]]
    retained_bytes: int


def read_dataset(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
    mode: DatasetReadMode = DatasetReadMode.MATERIALIZED,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    context = build_dataset_read_context(
        dataset_root,
        columns=columns,
        time_window=time_window,
    )

    if mode is DatasetReadMode.CHUNKED:
        return iter_dataset_chunks(context)
    if mode is not DatasetReadMode.MATERIALIZED:
        raise ValueError("unsupported_dataset_read_mode")
    return materialize_dataset_context(context)


def materialize_dataset_context(
    context: DatasetReadContext,
    *,
    target_chunk_bytes: int | None = None,
    materialized_budget_bytes: int | None = None,
) -> pd.DataFrame:
    """Materialize one validated context through the unique physical iterator."""

    frames: list[pd.DataFrame] = []
    observed_frame_bytes = 0
    input_frame_budget = (
        None
        if materialized_budget_bytes is None
        else _validated_target_chunk_bytes(materialized_budget_bytes) // 3
    )
    for frame in iter_dataset_chunks(
        context,
        target_chunk_bytes=target_chunk_bytes,
    ):
        observed_frame_bytes += _pandas_frame_bytes(frame)
        if input_frame_budget is not None and observed_frame_bytes > input_frame_budget:
            raise ValueError("dataset_materialized_result_exceeds_target_bytes")
        frames.append(frame)
    categorical_columns = tuple(
        column
        for column in frames[0].columns
        if all(
            isinstance(frame[column].dtype, pd.CategoricalDtype)
            for frame in frames
        )
    )
    categorical_values: dict[str, pd.Categorical] = {}
    for column in categorical_columns:
        try:
            categorical_values[column] = union_categoricals(
                [frame[column].array for frame in frames]
            )
        except TypeError as exc:
            raise ValueError("dataset_categorical_dtype_mismatch") from exc
    output = pd.concat(frames, ignore_index=True)
    for column, values in categorical_values.items():
        output[column] = pd.Series(values)
    return output


def build_dataset_read_context(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
    row_filters: tuple[tuple[str, str, Any], ...] = (),
) -> DatasetReadContext:
    root = require_artifact_root(Path(dataset_root))
    spec_path = root / "spec.json"
    data_root = root / "data"
    manifest_path = root / "_manifest" / "manifest.json"
    if not spec_path.is_file():
        raise ValueError("dataset_spec_missing")
    if not data_root.is_dir():
        raise ValueError("dataset_data_dir_missing")
    spec_payload = read_qrp_json(
        spec_path,
        expected_schema_id="qrp-dataset-spec",
        error="dataset_spec_schema_id_invalid",
    )
    if not manifest_path.is_file():
        raise ValueError("dataset_manifest_missing")
    manifest = read_qrp_json(
        manifest_path,
        expected_schema_id="qrp-dataset-manifest",
        error="dataset_manifest_schema_id_invalid",
    )
    validate_dataset_manifest_instance_metadata(manifest)
    if not manifest_has_runtime_metadata(manifest):
        raise ValueError("manifest_runtime_metadata_missing")
    all_paths = tuple(candidate_data_files(data_root, spec=spec_payload))
    if not all_paths:
        raise ValueError("dataset_data_dir_missing")
    schema = validated_schema(list(all_paths))
    schema_columns = tuple(name for name, _ in schema)
    schema_hash = sha256_json(schema)
    validate_manifest_runtime_metadata(
        manifest,
        data_root=data_root,
        paths=list(all_paths),
        validate_row_counts=True,
    )
    if tuple(str(column) for column in manifest["columns"]) != schema_columns:
        raise ValueError("manifest_columns_mismatch")
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
    return DatasetReadContext(
        root=root,
        data_root=data_root,
        spec=spec_payload,
        manifest=manifest,
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
        schema=schema,
        schema_hash=schema_hash,
        parquet_filters=physical_filters or None,
    )


def iter_dataset_chunks(
    context: DatasetReadContext,
    *,
    target_chunk_bytes: int | None = None,
) -> Iterator[pd.DataFrame]:
    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
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


def iter_dataset_file_batches(
    context: DatasetReadContext,
    *,
    columns: tuple[str, ...],
    target_chunk_bytes: int | None = None,
    categorical_dtypes: Mapping[str, pd.CategoricalDtype] | None = None,
    categorical_state_bytes: int | None = None,
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """Yield deterministic file batches after Arrow-level bounded conversion."""

    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    null_columns = _dataset_null_columns(context.all_paths, columns)
    if categorical_dtypes is None or categorical_state_bytes is None:
        categorical_dtypes, categorical_state_bytes = dataset_categorical_dtypes(
            context.all_paths,
            columns,
            target_state_bytes=max(1, target_bytes // 8),
        )
    physical_target_bytes = target_bytes - categorical_state_bytes
    if physical_target_bytes <= 0:
        raise ValueError("dataset_categorical_state_exceeds_target_bytes")
    for path in context.read_paths:
        for frame in iter_parquet_file_batches(
            path,
            columns=columns,
            parquet_filters=context.parquet_filters,
            target_chunk_bytes=physical_target_bytes,
            null_columns=null_columns,
            categorical_dtypes=categorical_dtypes,
        ):
            yield path, frame


def iter_parquet_file_batches(
    path: Path,
    *,
    target_chunk_bytes: int,
    columns: tuple[str, ...] | None = None,
    parquet_filters: list[tuple[str, str, Any]] | None = None,
    null_columns: frozenset[str] | None = None,
    categorical_dtypes: Mapping[str, pd.CategoricalDtype] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield bounded Pandas batches from one Parquet file."""

    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    parquet_file = pq.ParquetFile(path)
    try:
        selected_columns = columns or tuple(parquet_file.schema_arrow.names)
        selected_fields = tuple(
            parquet_file.schema_arrow.field(name) for name in selected_columns
        )
        observed_null_columns = (
            _parquet_null_columns(parquet_file, selected_columns)
            if null_columns is None
            else null_columns
        )
        dtype_overrides = _pandas_dtype_overrides(
            pa.schema(selected_fields),
            observed_null_columns,
            categorical_dtypes=categorical_dtypes,
        )
    finally:
        parquet_file.close()
    for batch in iter_parquet_arrow_batches(
        path,
        columns=selected_columns,
        parquet_filters=parquet_filters,
        target_chunk_bytes=target_bytes,
    ):
        yield from _arrow_batch_to_bounded_pandas(
            batch,
            target_chunk_bytes=target_bytes,
            dtype_overrides=dtype_overrides,
        )


def iter_parquet_arrow_batches(
    path: Path,
    *,
    target_chunk_bytes: int,
    columns: tuple[str, ...] | None = None,
    parquet_filters: list[tuple[str, str, Any]] | None = None,
    row_groups: tuple[int, ...] | None = None,
    source_batch_rows: int | None = None,
) -> Iterator[pa.RecordBatch]:
    """Yield bounded Arrow batches through the single physical decoder."""

    target_bytes = _validated_target_chunk_bytes(target_chunk_bytes)
    parquet_file = pq.ParquetFile(path)
    try:
        selected_columns = columns or tuple(parquet_file.schema_arrow.names)
        selected_fields = tuple(
            parquet_file.schema_arrow.field(name) for name in selected_columns
        )
        variable_width = any(
            _is_variable_width(field.type) for field in selected_fields
        )
        batch_size = source_batch_rows or _target_batch_rows(
            parquet_file,
            columns=selected_columns,
            target_chunk_bytes=target_bytes,
        )
    finally:
        parquet_file.close()
    if row_groups is not None:
        arrow_budget = max(1, target_bytes // _PHYSICAL_WORKING_SET_PARTS)
        for batch in _raw_arrow_batches(
            path,
            columns=selected_columns,
            parquet_filters=None,
            batch_size=batch_size,
            row_groups=row_groups,
            max_page_uncompressed_bytes=arrow_budget,
        ):
            _require_arrow_batch_budget(batch, arrow_budget=arrow_budget)
            yield batch
        return
    yield from _iter_filtered_arrow_batches(
        path,
        columns=selected_columns,
        parquet_filters=parquet_filters,
        batch_size=batch_size,
        target_chunk_bytes=target_bytes,
        probe_variable_width=variable_width,
    )


def dataset_primary_key_schema(
    context: DatasetReadContext,
    primary_key: tuple[str, ...],
) -> pa.Schema:
    parquet_file = pq.ParquetFile(context.all_paths[0])
    try:
        parquet_schema = parquet_file.schema_arrow
        return pa.schema(
            [parquet_schema.field(column) for column in primary_key]
        )
    except KeyError as exc:
        raise ValueError("missing_dataset_primary_key_column") from exc
    finally:
        parquet_file.close()


def build_dataset_equality_group_plan(
    context: DatasetReadContext,
    *,
    columns: tuple[str, ...],
    target_chunk_bytes: int,
    target_state_bytes: int,
) -> DatasetEqualityGroupPlan:
    schema_columns = {name for name, _ in context.schema}
    if not columns or any(column not in schema_columns for column in columns):
        raise ValueError("invalid_dataset_equality_group_columns")
    if (
        isinstance(target_state_bytes, bool)
        or not isinstance(target_state_bytes, int)
        or target_state_bytes <= 0
    ):
        raise ValueError("invalid_dataset_equality_group_state_budget")
    filter_columns = tuple(
        column for column, _, _ in context.parquet_filters or ()
    )
    discovery_columns = tuple(dict.fromkeys((*columns, *filter_columns)))
    paths_by_key: dict[tuple[Any, ...], dict[Path, None]] = {}
    state_bytes = sys.getsizeof(paths_by_key)
    for path, frame in iter_dataset_file_batches(
        context,
        columns=discovery_columns,
        target_chunk_bytes=target_chunk_bytes,
    ):
        for row in frame.loc[:, list(columns)].itertuples(index=False, name=None):
            key = tuple(row)
            if any(
                not pd.api.types.is_scalar(value) or bool(pd.isna(value))
                for value in key
            ):
                raise ValueError("invalid_dataset_equality_group_key")
            try:
                hash(key)
            except TypeError as exc:
                raise ValueError("invalid_dataset_equality_group_key") from exc
            candidate_paths = paths_by_key.get(key)
            is_new_key = candidate_paths is None
            outer_growth = 0
            if candidate_paths is None:
                previous_outer_bytes = sys.getsizeof(paths_by_key)
                candidate_paths = {}
                paths_by_key[key] = candidate_paths
                outer_growth = sys.getsizeof(paths_by_key) - previous_outer_bytes
            is_new_path = path not in candidate_paths
            path_growth = 0
            if is_new_path:
                previous_path_bytes = sys.getsizeof(candidate_paths)
                candidate_paths[path] = None
                path_growth = sys.getsizeof(candidate_paths) - previous_path_bytes
            state_bytes = _equality_group_plan_bytes(
                state_bytes,
                key=key,
                path=path,
                is_new_key=is_new_key,
                is_new_path=is_new_path,
                outer_growth=outer_growth,
                path_growth=path_growth,
            )
            if state_bytes > target_state_bytes:
                raise ValueError("dataset_equality_group_index_exceeds_target_bytes")
    finalization_bytes = _equality_group_plan_finalization_bytes(paths_by_key)
    if state_bytes + finalization_bytes > target_state_bytes:
        raise ValueError("dataset_equality_group_index_exceeds_target_bytes")
    ordered_keys = tuple(sorted(paths_by_key))
    ordered_paths = {
        key: tuple(paths_by_key[key])
        for key in ordered_keys
    }
    retained_bytes = _equality_group_plan_retained_bytes(
        ordered_keys,
        ordered_paths,
    )
    if retained_bytes > target_state_bytes:
        raise ValueError("dataset_equality_group_index_exceeds_target_bytes")
    return DatasetEqualityGroupPlan(
        columns=columns,
        keys=ordered_keys,
        paths_by_key=ordered_paths,
        retained_bytes=retained_bytes,
    )


def dataset_context_for_equality_group(
    context: DatasetReadContext,
    plan: DatasetEqualityGroupPlan,
    key: tuple[Any, ...],
) -> DatasetReadContext:
    if key not in plan.paths_by_key or len(key) != len(plan.columns):
        raise ValueError("unknown_dataset_equality_group_key")
    filters = [*(context.parquet_filters or ())]
    filters.extend(
        (column, "==", value)
        for column, value in zip(plan.columns, key, strict=True)
    )
    physical_columns = tuple(dict.fromkeys((*context.physical_columns, *plan.columns)))
    return replace(
        context,
        read_paths=plan.paths_by_key[key],
        physical_columns=physical_columns,
        parquet_filters=filters,
    )


def _equality_group_plan_bytes(
    current_bytes: int,
    *,
    key: tuple[Any, ...],
    path: Path,
    is_new_key: bool,
    is_new_path: bool,
    outer_growth: int = 0,
    path_growth: int = 0,
) -> int:
    return (
        current_bytes
        + (
            outer_growth
            + sys.getsizeof(key)
            + sum(sys.getsizeof(value) for value in key)
            + sys.getsizeof({})
            if is_new_key
            else 0
        )
        + (
            path_growth
            + sys.getsizeof(path)
            + sys.getsizeof(path.parts)
            + sum(sys.getsizeof(part) for part in path.parts)
            if is_new_path
            else 0
        )
    )


def _equality_group_plan_finalization_bytes(
    paths_by_key: Mapping[tuple[Any, ...], Mapping[Path, None]],
) -> int:
    key_count = len(paths_by_key)
    pointer_bytes = _PYTHON_POINTER_BYTES
    mapping_bound = sys.getsizeof({}) + key_count * 8 * pointer_bytes
    sorted_list_bound = sys.getsizeof([]) + key_count * pointer_bytes
    keys_tuple_bound = sys.getsizeof(()) + key_count * pointer_bytes
    path_tuple_bound = sum(
        sys.getsizeof(()) + len(paths) * pointer_bytes
        for paths in paths_by_key.values()
    )
    return (
        mapping_bound
        + sorted_list_bound
        + keys_tuple_bound
        + path_tuple_bound
        + sys.getsizeof(object.__new__(DatasetEqualityGroupPlan))
    )


def _equality_group_plan_retained_bytes(
    keys: tuple[tuple[Any, ...], ...],
    paths_by_key: Mapping[tuple[Any, ...], tuple[Path, ...]],
) -> int:
    retained_bytes = (
        sys.getsizeof(object.__new__(DatasetEqualityGroupPlan))
        + sys.getsizeof(keys)
        + sys.getsizeof(paths_by_key)
    )
    for key in keys:
        retained_bytes += sys.getsizeof(key)
        retained_bytes += sum(sys.getsizeof(value) for value in key)
        paths = paths_by_key[key]
        retained_bytes += sys.getsizeof(paths)
        for path in paths:
            retained_bytes += sys.getsizeof(path)
            retained_bytes += sys.getsizeof(path.parts)
            retained_bytes += sum(sys.getsizeof(part) for part in path.parts)
    return retained_bytes


def _validated_target_chunk_bytes(value: int | None) -> int:
    if value is None:
        return DEFAULT_DATASET_PHYSICAL_TARGET_BYTES
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("invalid_target_chunk_bytes")
    return value


def _target_batch_rows(
    parquet_file: pq.ParquetFile,
    *,
    columns: tuple[str, ...],
    target_chunk_bytes: int,
) -> int:
    selected_fields = tuple(parquet_file.schema_arrow.field(name) for name in columns)
    fixed_row_bytes = tuple(
        _fixed_width_arrow_row_bytes(field.type) for field in selected_fields
    )
    if any(value is None for value in fixed_row_bytes):
        return 1
    selected_positions = _projected_parquet_leaf_positions(
        parquet_file,
        columns,
    )
    maximum_row_bytes = max(1, sum(value or 0 for value in fixed_row_bytes))
    metadata = parquet_file.metadata
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        if row_group.num_rows <= 0:
            continue
        projected_bytes = sum(
            max(0, int(row_group.column(index).total_uncompressed_size))
            for index in selected_positions
        )
        row_bytes = max(
            1,
            (projected_bytes + row_group.num_rows - 1) // row_group.num_rows,
        )
        maximum_row_bytes = max(maximum_row_bytes, row_bytes)
    conservative_row_bytes = maximum_row_bytes * _PARQUET_TO_PANDAS_EXPANSION_FACTOR
    arrow_budget = max(1, target_chunk_bytes // _PHYSICAL_WORKING_SET_PARTS)
    return max(1, arrow_budget // conservative_row_bytes)


def _projected_parquet_leaf_positions(
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
) -> tuple[int, ...]:
    positions: list[int] = []
    for column in columns:
        matches = [
            index
            for index in range(parquet_file.metadata.num_columns)
            if (
                parquet_file.schema.column(index).path == column
                or parquet_file.schema.column(index).path.startswith(f"{column}.")
            )
        ]
        if not matches:
            raise ValueError("dataset_column_not_found")
        positions.extend(matches)
    return tuple(positions)


def _parquet_null_columns(
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
) -> frozenset[str]:
    null_columns: set[str] = set()
    for column in columns:
        positions = _projected_parquet_leaf_positions(parquet_file, (column,))
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            row_group = parquet_file.metadata.row_group(row_group_index)
            statistics = tuple(
                row_group.column(position).statistics for position in positions
            )
            if any(
                item is None or item.null_count is None or int(item.null_count) > 0
                for item in statistics
            ):
                null_columns.add(column)
                break
    return frozenset(null_columns)


def _dataset_null_columns(
    paths: tuple[Path, ...],
    columns: tuple[str, ...],
) -> frozenset[str]:
    null_columns: set[str] = set()
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        try:
            null_columns.update(
                _parquet_null_columns(parquet_file, columns)
            )
        finally:
            parquet_file.close()
    return frozenset(null_columns)


def _pandas_dtype_overrides(
    schema: pa.Schema,
    null_columns: frozenset[str],
    *,
    categorical_dtypes: Mapping[str, pd.CategoricalDtype] | None = None,
) -> dict[str, object]:
    overrides: dict[str, object] = dict(categorical_dtypes or {})
    for field in schema:
        if field.name not in null_columns:
            continue
        if pa.types.is_integer(field.type):
            prefix = "Int" if pa.types.is_signed_integer(field.type) else "UInt"
            overrides[field.name] = f"{prefix}{field.type.bit_width}"
        elif pa.types.is_boolean(field.type):
            overrides[field.name] = "boolean"
    return overrides


def dataset_categorical_dtypes(
    paths: tuple[Path, ...],
    columns: tuple[str, ...],
    *,
    target_state_bytes: int = DEFAULT_DATASET_PHYSICAL_TARGET_BYTES // 8,
) -> tuple[dict[str, pd.CategoricalDtype], int]:
    if target_state_bytes <= 0:
        raise ValueError("dataset_categorical_state_exceeds_target_bytes")
    categories: dict[str, dict[object, None]] = {}
    ordered_domains: dict[str, tuple[object, ...]] = {}
    ordered_columns: dict[str, bool] = {}
    observed_state_bytes = 0
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        try:
            dictionary_columns = tuple(
                column
                for column in columns
                if pa.types.is_dictionary(parquet_file.schema_arrow.field(column).type)
            )
            for column in dictionary_columns:
                ordered_columns[column] = bool(
                    parquet_file.schema_arrow.field(column).type.ordered
                )
            if not dictionary_columns:
                continue
            for row_group_index in range(parquet_file.metadata.num_row_groups):
                _require_categorical_row_group_budget(
                    parquet_file,
                    row_group_index=row_group_index,
                    columns=dictionary_columns,
                    target_state_bytes=target_state_bytes,
                )
                batches = iter_parquet_arrow_batches(
                    path,
                    columns=dictionary_columns,
                    target_chunk_bytes=target_state_bytes
                    * _PHYSICAL_WORKING_SET_PARTS,
                    row_groups=(row_group_index,),
                    source_batch_rows=1,
                )
                try:
                    batch = next(batches, None)
                finally:
                    batches.close()
                if batch is None:
                    continue
                for position, column in enumerate(dictionary_columns):
                    array = batch.column(position)
                    if not isinstance(array, pa.DictionaryArray):
                        raise ValueError("dataset_categorical_dtype_mismatch")
                    if (
                        observed_state_bytes
                        + max(1, int(array.dictionary.nbytes))
                        * _VARIABLE_WIDTH_TO_PANDAS_EXPANSION_FACTOR
                        > target_state_bytes
                    ):
                        raise ValueError(
                            "dataset_categorical_state_exceeds_target_bytes"
                        )
                    values = tuple(array.dictionary.to_pylist())
                    if ordered_columns[column]:
                        expected = ordered_domains.get(column)
                        if expected is None:
                            ordered_domains[column] = values
                            observed_state_bytes += max(
                                max(1, int(array.dictionary.nbytes))
                                * _VARIABLE_WIDTH_TO_PANDAS_EXPANSION_FACTOR,
                                sum(sys.getsizeof(value) for value in values),
                            )
                            if observed_state_bytes > target_state_bytes:
                                raise ValueError(
                                    "dataset_categorical_state_exceeds_target_bytes"
                                )
                            continue
                        if values != expected:
                            raise ValueError("dataset_categorical_dtype_mismatch")
                        continue
                    domain = categories.setdefault(column, {})
                    previous_bytes = sys.getsizeof(domain)
                    new_values = tuple(
                        value for value in values if value not in domain
                    )
                    for value in values:
                        domain[value] = None
                    observed_state_bytes += sys.getsizeof(domain) - previous_bytes
                    observed_state_bytes += sum(
                        sys.getsizeof(value) for value in new_values
                    )
                    if observed_state_bytes > target_state_bytes:
                        raise ValueError(
                            "dataset_categorical_state_exceeds_target_bytes"
                        )
        finally:
            parquet_file.close()
    dtypes: dict[str, pd.CategoricalDtype] = {}
    for column, ordered in ordered_columns.items():
        values = (
            ordered_domains[column]
            if ordered
            else tuple(
                sorted(
                    categories.get(column, {}),
                    key=lambda value: (
                        type(value).__module__,
                        type(value).__qualname__,
                        repr(value),
                    ),
                )
            )
        )
        dtypes[column] = pd.CategoricalDtype(categories=values, ordered=ordered)
    dtype_state_bytes = sum(
        sys.getsizeof(dtype)
        + int(pd.Series(dtype.categories).memory_usage(index=True, deep=True))
        for dtype in dtypes.values()
    )
    observed_state_bytes = max(observed_state_bytes, dtype_state_bytes)
    if observed_state_bytes > target_state_bytes:
        raise ValueError("dataset_categorical_state_exceeds_target_bytes")
    return dtypes, observed_state_bytes


def _require_categorical_row_group_budget(
    parquet_file: pq.ParquetFile,
    *,
    row_group_index: int,
    columns: tuple[str, ...],
    target_state_bytes: int,
) -> None:
    positions = _projected_parquet_leaf_positions(parquet_file, columns)
    row_group = parquet_file.metadata.row_group(row_group_index)
    encoded_bytes = sum(
        max(0, int(row_group.column(position).total_uncompressed_size))
        for position in positions
    )
    if encoded_bytes > target_state_bytes:
        raise ValueError("dataset_categorical_state_exceeds_target_bytes")


def _iter_filtered_arrow_batches(
    path: Path,
    *,
    columns: tuple[str, ...],
    parquet_filters: list[tuple[str, str, Any]] | None,
    batch_size: int,
    target_chunk_bytes: int,
    probe_variable_width: bool,
) -> Iterator[pa.RecordBatch]:
    arrow_budget = max(1, target_chunk_bytes // _PHYSICAL_WORKING_SET_PARTS)
    if probe_variable_width:
        yield from _iter_variable_width_arrow_batches(
            path,
            columns=columns,
            parquet_filters=parquet_filters,
            arrow_budget=arrow_budget,
        )
        return
    source_budget = (
        arrow_budget
        if not parquet_filters
        else max(1, arrow_budget // 2)
    )
    for source_batch in _raw_arrow_batches(
        path,
        columns=columns,
        parquet_filters=parquet_filters,
        batch_size=batch_size,
        max_page_uncompressed_bytes=source_budget,
    ):
        _require_arrow_batch_budget(source_batch, arrow_budget=source_budget)
        filtered = _filter_arrow_batch(source_batch, parquet_filters)
        if filtered is not source_batch:
            _require_arrow_batch_budget(filtered, arrow_budget=source_budget)
        if filtered.num_rows:
            yield filtered


def _iter_variable_width_arrow_batches(
    path: Path,
    *,
    columns: tuple[str, ...],
    parquet_filters: list[tuple[str, str, Any]] | None,
    arrow_budget: int,
) -> Iterator[pa.RecordBatch]:
    component_budget = max(1, arrow_budget // 2)
    buffered: list[pa.RecordBatch] = []
    buffered_arrow_bytes = 0
    buffered_python_bytes = sys.getsizeof(buffered)
    for source_batch in _adaptive_variable_width_source_batches(
        path,
        columns=columns,
        parquet_filters=parquet_filters,
        component_budget=component_budget,
        page_budget=component_budget,
    ):
        _require_arrow_batch_budget(source_batch, arrow_budget=component_budget)
        filtered = _filter_arrow_batch(source_batch, parquet_filters)
        del source_batch
        if filtered.num_rows == 0:
            continue
        batch_arrow_bytes = filtered.nbytes
        batch_python_bytes = sys.getsizeof(filtered)
        if batch_arrow_bytes > component_budget or batch_python_bytes > arrow_budget:
            raise ValueError("dataset_row_exceeds_target_chunk_bytes")
        prior_list_bytes = sys.getsizeof(buffered)
        buffered.append(filtered)
        candidate_arrow_bytes = buffered_arrow_bytes + batch_arrow_bytes
        candidate_python_bytes = (
            buffered_python_bytes
            - prior_list_bytes
            + sys.getsizeof(buffered)
            + batch_python_bytes
        )
        if len(buffered) > 1 and (
            candidate_arrow_bytes > component_budget
            or candidate_python_bytes > arrow_budget
        ):
            buffered.pop()
            yield _coalesced_record_batch(buffered, arrow_budget=arrow_budget)
            buffered = [filtered]
            buffered_arrow_bytes = batch_arrow_bytes
            buffered_python_bytes = sys.getsizeof(buffered) + batch_python_bytes
            continue
        buffered_arrow_bytes = candidate_arrow_bytes
        buffered_python_bytes = candidate_python_bytes
    if buffered:
        yield _coalesced_record_batch(buffered, arrow_budget=arrow_budget)


def _adaptive_variable_width_source_batches(
    path: Path,
    *,
    columns: tuple[str, ...],
    parquet_filters: list[tuple[str, str, Any]] | None,
    component_budget: int,
    page_budget: int | None = None,
) -> Iterator[pa.RecordBatch]:
    physical_page_budget = page_budget or component_budget
    probe_batches = _raw_arrow_batches(
        path,
        columns=columns,
        parquet_filters=parquet_filters,
        batch_size=1,
        max_page_uncompressed_bytes=physical_page_budget,
    )
    try:
        probe = next(probe_batches, None)
        if probe is None:
            return
        yield probe
        batch_size = _variable_width_source_batch_rows(
            path,
            columns=columns,
            component_budget=component_budget,
            probe=probe,
        )
        if batch_size == 1:
            yield from probe_batches
            return
    finally:
        probe_batches.close()

    skipped_rows = probe.num_rows
    for batch in _raw_arrow_batches(
        path,
        columns=columns,
        parquet_filters=parquet_filters,
        batch_size=batch_size,
        max_page_uncompressed_bytes=physical_page_budget,
    ):
        if skipped_rows:
            if batch.num_rows <= skipped_rows:
                skipped_rows -= batch.num_rows
                continue
            batch = batch.slice(skipped_rows)
            skipped_rows = 0
        if batch.num_rows:
            yield batch


def _variable_width_source_batch_rows(
    path: Path,
    *,
    columns: tuple[str, ...],
    component_budget: int,
    probe: pa.RecordBatch,
) -> int:
    parquet_file = pq.ParquetFile(path)
    try:
        variable_columns = tuple(
            field.name
            for field in probe.schema
            if _is_variable_width(field.type)
        )
        variable_positions = _projected_parquet_leaf_positions(
            parquet_file,
            variable_columns,
        )
        fixed_row_bytes = sum(
            _fixed_width_arrow_row_bytes(field.type) or 0
            for field in probe.schema
        )
        maximum_variable_row_bytes = 0
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            row_group = parquet_file.metadata.row_group(row_group_index)
            row_count = max(1, int(row_group.num_rows))
            variable_chunk_bytes = 0
            dictionary_encoded = False
            for index in variable_positions:
                column_chunk = row_group.column(index)
                variable_chunk_bytes += max(
                    0,
                    int(column_chunk.total_uncompressed_size),
                )
                dictionary_encoded = dictionary_encoded or any(
                    "DICTIONARY" in str(encoding)
                    for encoding in column_chunk.encodings
                )
            variable_row_bytes = (
                variable_chunk_bytes
                if dictionary_encoded
                else (variable_chunk_bytes + row_count - 1) // row_count
            )
            maximum_variable_row_bytes = max(
                maximum_variable_row_bytes,
                variable_row_bytes,
            )
    finally:
        parquet_file.close()
    probe_row_bytes = max(
        1,
        (probe.nbytes + probe.num_rows - 1) // probe.num_rows,
    )
    conservative_row_bytes = max(
        probe_row_bytes,
        fixed_row_bytes + maximum_variable_row_bytes,
    )
    return max(1, component_budget // conservative_row_bytes)


def _coalesced_record_batch(
    batches: list[pa.RecordBatch],
    *,
    arrow_budget: int,
) -> pa.RecordBatch:
    if len(batches) == 1:
        output = batches[0]
    else:
        table = pa.Table.from_batches(batches).combine_chunks()
        output_batches = table.to_batches(max_chunksize=table.num_rows)
        if len(output_batches) != 1:
            raise ValueError("invalid_dataset_arrow_batch")
        output = output_batches[0]
        del table, output_batches
    _require_arrow_batch_budget(output, arrow_budget=arrow_budget)
    return output


def _require_arrow_batch_budget(
    batch: pa.RecordBatch,
    *,
    arrow_budget: int,
) -> None:
    if batch.nbytes <= arrow_budget:
        return
    if batch.num_rows == 1:
        raise ValueError("dataset_row_exceeds_target_chunk_bytes")
    raise ValueError("dataset_arrow_batch_exceeds_target_chunk_bytes")


def _raw_arrow_batches(
    path: Path,
    *,
    columns: tuple[str, ...],
    parquet_filters: list[tuple[str, str, Any]] | None,
    batch_size: int,
    row_groups: tuple[int, ...] | None = None,
    max_page_uncompressed_bytes: int,
) -> Iterator[pa.RecordBatch]:
    parquet_file = pq.ParquetFile(path)
    try:
        selected_row_groups = (
            _matching_row_groups(parquet_file, parquet_filters)
            if row_groups is None
            else row_groups
        )
        _require_selected_parquet_page_budget(
            path,
            parquet_file=parquet_file,
            columns=columns,
            row_groups=tuple(selected_row_groups),
            max_page_uncompressed_bytes=max_page_uncompressed_bytes,
        )
        yield from parquet_file.iter_batches(
            batch_size=batch_size,
            row_groups=selected_row_groups,
            columns=list(columns),
            use_threads=False,
        )
    finally:
        parquet_file.close()


def _require_selected_parquet_page_budget(
    path: Path,
    *,
    parquet_file: pq.ParquetFile,
    columns: tuple[str, ...],
    row_groups: tuple[int, ...],
    max_page_uncompressed_bytes: int,
) -> None:
    positions = _projected_parquet_leaf_positions(parquet_file, columns)
    with Path(path).open("rb") as stream:
        for row_group_index in row_groups:
            row_group = parquet_file.metadata.row_group(row_group_index)
            if int(row_group.num_rows) == 0:
                continue
            for position in positions:
                column = row_group.column(position)
                offsets = tuple(
                    offset
                    for offset in (
                        int(column.dictionary_page_offset or 0),
                        int(column.data_page_offset or 0),
                    )
                    if offset > 0
                )
                if not offsets:
                    raise ValueError("invalid_dataset_parquet_page_metadata")
                start = min(offsets)
                end = start + int(column.total_compressed_size)
                stream.seek(start)
                while stream.tell() < end:
                    uncompressed_size, compressed_size = _read_parquet_page_header(
                        stream
                    )
                    if uncompressed_size > max_page_uncompressed_bytes:
                        if int(row_group.num_rows) == 1:
                            raise ValueError("dataset_row_exceeds_target_chunk_bytes")
                        raise ValueError(
                            "dataset_parquet_page_exceeds_target_chunk_bytes"
                        )
                    payload_end = stream.tell() + compressed_size
                    if compressed_size < 0 or payload_end > end:
                        raise ValueError("invalid_dataset_parquet_page_metadata")
                    stream.seek(payload_end)
                if stream.tell() != end:
                    raise ValueError("invalid_dataset_parquet_page_metadata")


def _read_parquet_page_header(stream: BinaryIO) -> tuple[int, int]:
    previous_field_id = 0
    sizes: dict[int, int] = {}
    while True:
        field_header = _read_compact_byte(stream)
        field_type = field_header & 0x0F
        if field_type == 0:
            break
        field_delta = field_header >> 4
        field_id = (
            previous_field_id + field_delta
            if field_delta
            else _read_compact_integer(stream)
        )
        previous_field_id = field_id
        if field_id in (2, 3) and field_type in (4, 5, 6):
            sizes[field_id] = _read_compact_integer(stream)
        else:
            _skip_compact_value(stream, field_type)
    uncompressed_size = sizes.get(2)
    compressed_size = sizes.get(3)
    if (
        uncompressed_size is None
        or compressed_size is None
        or uncompressed_size < 0
        or compressed_size < 0
    ):
        raise ValueError("invalid_dataset_parquet_page_metadata")
    return uncompressed_size, compressed_size


def _read_compact_byte(stream: BinaryIO) -> int:
    value = stream.read(1)
    if len(value) != 1:
        raise ValueError("invalid_dataset_parquet_page_metadata")
    return value[0]


def _read_compact_unsigned(stream: BinaryIO) -> int:
    value = 0
    for shift in range(0, 70, 7):
        byte = _read_compact_byte(stream)
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value
    raise ValueError("invalid_dataset_parquet_page_metadata")


def _read_compact_integer(stream: BinaryIO) -> int:
    value = _read_compact_unsigned(stream)
    return (value >> 1) ^ -(value & 1)


def _skip_compact_value(
    stream: BinaryIO,
    field_type: int,
    *,
    collection_value: bool = False,
) -> None:
    if field_type == 0:
        raise ValueError("invalid_dataset_parquet_page_metadata")
    if field_type in (1, 2):
        if collection_value:
            _read_compact_byte(stream)
        return
    if field_type == 3:
        _read_compact_byte(stream)
        return
    if field_type in (4, 5, 6):
        _read_compact_unsigned(stream)
        return
    if field_type == 7:
        stream.seek(8, 1)
        return
    if field_type == 8:
        size = _read_compact_unsigned(stream)
        stream.seek(size, 1)
        return
    if field_type in (9, 10):
        header = _read_compact_byte(stream)
        size = header >> 4
        element_type = header & 0x0F
        if size == 15:
            size = _read_compact_unsigned(stream)
        for _ in range(size):
            _skip_compact_value(
                stream,
                element_type,
                collection_value=True,
            )
        return
    if field_type == 11:
        size = _read_compact_unsigned(stream)
        if size:
            types = _read_compact_byte(stream)
            key_type = types >> 4
            value_type = types & 0x0F
            for _ in range(size):
                _skip_compact_value(stream, key_type, collection_value=True)
                _skip_compact_value(stream, value_type, collection_value=True)
        return
    if field_type == 12:
        previous_field_id = 0
        while True:
            header = _read_compact_byte(stream)
            nested_type = header & 0x0F
            if nested_type == 0:
                return
            field_delta = header >> 4
            previous_field_id = (
                previous_field_id + field_delta
                if field_delta
                else _read_compact_integer(stream)
            )
            _skip_compact_value(stream, nested_type)
    raise ValueError("invalid_dataset_parquet_page_metadata")


def _matching_row_groups(
    parquet_file: pq.ParquetFile,
    filters: list[tuple[str, str, Any]] | None,
) -> list[int]:
    indexes = list(range(parquet_file.metadata.num_row_groups))
    if not filters:
        return indexes
    positions = _top_level_parquet_leaf_positions(parquet_file, filters)
    matching: list[int] = []
    for row_group_index in indexes:
        row_group = parquet_file.metadata.row_group(row_group_index)
        if all(
            _row_group_may_match(
                (
                    None
                    if positions[column] is None
                    else row_group.column(positions[column]).statistics
                ),
                operator=operator,
                value=value,
            )
            for column, operator, value in filters
        ):
            matching.append(row_group_index)
    return matching


def _top_level_parquet_leaf_positions(
    parquet_file: pq.ParquetFile,
    filters: list[tuple[str, str, Any]],
) -> dict[str, int | None]:
    positions: dict[str, int | None] = {}
    for column, _, _ in filters:
        matches = [
            index
            for index in range(parquet_file.metadata.num_columns)
            if parquet_file.schema.column(index).path == column
        ]
        positions[column] = matches[0] if len(matches) == 1 else None
    return positions


def _row_group_may_match(
    statistics: Any,
    *,
    operator: str,
    value: Any,
) -> bool:
    if statistics is None or not statistics.has_min_max:
        return True
    minimum = statistics.min
    maximum = statistics.max
    try:
        if operator == ">=":
            return maximum >= value
        if operator == ">":
            return maximum > value
        if operator == "<=":
            return minimum <= value
        if operator == "<":
            return minimum < value
        if operator == "==":
            return minimum <= value <= maximum
    except (TypeError, ValueError):
        return True
    raise ValueError("unsupported_dataset_physical_filter")


def _filter_arrow_batch(
    batch: pa.RecordBatch,
    filters: list[tuple[str, str, Any]] | None,
) -> pa.RecordBatch:
    mask: pa.Array | None = None
    operations = {
        ">=": pc.greater_equal,
        ">": pc.greater,
        "<=": pc.less_equal,
        "<": pc.less,
        "==": pc.equal,
    }
    for column, operator, value in filters or ():
        operation = operations.get(operator)
        if operation is None:
            raise ValueError("unsupported_dataset_physical_filter")
        condition = operation(batch.column(batch.schema.get_field_index(column)), value)
        mask = condition if mask is None else pc.and_kleene(mask, condition)
    return batch if mask is None else batch.filter(mask)


def _empty_frame(
    context: DatasetReadContext,
    columns: tuple[str, ...],
    *,
    target_chunk_bytes: int = DEFAULT_DATASET_PHYSICAL_TARGET_BYTES,
) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(context.all_paths[0])
    try:
        schema = parquet_file.schema_arrow
        selected = pa.schema([schema.field(column) for column in columns])
        null_columns = _dataset_null_columns(context.all_paths, columns)
        categorical_dtypes, _ = dataset_categorical_dtypes(
            context.all_paths,
            columns,
            target_state_bytes=max(1, target_chunk_bytes // 8),
        )
        overrides = _pandas_dtype_overrides(
            selected,
            null_columns,
            categorical_dtypes=categorical_dtypes,
        )
        pandas_budget = max(1, target_chunk_bytes // _PHYSICAL_WORKING_SET_PARTS)
        conversion_budget = max(1, pandas_budget // 2) if overrides else pandas_budget
        if _empty_pandas_conversion_bound_bytes(selected) > conversion_budget:
            raise ValueError("dataset_row_exceeds_target_chunk_bytes")
        frame = pa.Table.from_batches([], schema=selected).to_pandas(use_threads=False)
        output = frame.astype(overrides) if overrides else frame
        frame_bytes = _pandas_frame_bytes(frame)
        output_bytes = _pandas_frame_bytes(output)
        if frame_bytes > conversion_budget or (
            output is not frame and frame_bytes + output_bytes > pandas_budget
        ):
            raise ValueError("dataset_row_exceeds_target_chunk_bytes")
        return output
    finally:
        parquet_file.close()


def _is_variable_width(data_type: pa.DataType) -> bool:
    return _fixed_width_arrow_row_bytes(data_type) is None


def _requires_conservative_pandas_probe(data_type: pa.DataType) -> bool:
    return _is_variable_width(data_type) or (
        pa.types.is_fixed_size_binary(data_type)
        or pa.types.is_decimal(data_type)
        or pa.types.is_date(data_type)
        or pa.types.is_time(data_type)
    )


def _pandas_conversion_bound_bytes(batch: pa.RecordBatch) -> int:
    return (
        _PANDAS_FRAME_OVERHEAD_BYTES
        + int(pd.RangeIndex(batch.num_rows).memory_usage(deep=True))
        + _pandas_column_axis_bytes(batch.schema.names)
        + sum(
            max(1, int(batch.column(position).nbytes))
            + _pandas_array_object_overhead(
                batch.column(position),
                field.type,
                nested=False,
            )
            for position, field in enumerate(batch.schema)
        )
    )


def _empty_pandas_conversion_bound_bytes(schema: pa.Schema) -> int:
    return (
        _PANDAS_FRAME_OVERHEAD_BYTES
        + int(pd.RangeIndex(0).memory_usage(deep=True))
        + _pandas_column_axis_bytes(schema.names)
        + len(schema)
    )


def _pandas_column_axis_bytes(names: Iterable[str]) -> int:
    return int(pd.Index(tuple(names)).memory_usage(deep=True))


def _pandas_frame_bytes(frame: pd.DataFrame) -> int:
    return (
        _PANDAS_FRAME_OVERHEAD_BYTES
        + int(frame.memory_usage(index=True, deep=True).sum())
        + int(frame.columns.memory_usage(deep=True))
    )


def _pandas_array_object_overhead(
    array: pa.Array,
    data_type: pa.DataType,
    *,
    nested: bool,
) -> int:
    row_count = len(array)
    if pa.types.is_struct(data_type):
        container_overhead = row_count * (
            sys.getsizeof({})
            + _PYTHON_POINTER_BYTES
            + sum(
                sys.getsizeof(field.name) + (2 * _PYTHON_POINTER_BYTES)
                for field in data_type
            )
        )
        return container_overhead + sum(
            _pandas_array_object_overhead(
                array.field(position),
                field.type,
                nested=True,
            )
            for position, field in enumerate(data_type)
        )
    if pa.types.is_fixed_size_list(data_type):
        value_count = row_count * data_type.list_size
        values = array.values.slice(array.offset * data_type.list_size, value_count)
        return (
            row_count * (sys.getsizeof([]) + _PYTHON_POINTER_BYTES)
            + value_count * _PYTHON_POINTER_BYTES
            + _pandas_array_object_overhead(
                values,
                data_type.value_type,
                nested=True,
            )
        )
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        start, stop = _list_value_bounds(array)
        values = array.values.slice(start, stop - start)
        return (
            row_count * (sys.getsizeof([]) + _PYTHON_POINTER_BYTES)
            + (stop - start) * _PYTHON_POINTER_BYTES
            + _pandas_array_object_overhead(
                values,
                data_type.value_type,
                nested=True,
            )
        )
    if pa.types.is_map(data_type):
        start, stop = _list_value_bounds(array)
        value_count = stop - start
        return (
            row_count * (sys.getsizeof([]) + _PYTHON_POINTER_BYTES)
            + value_count * sys.getsizeof((None, None))
            + _pandas_array_object_overhead(
                array.keys.slice(start, value_count),
                data_type.key_type,
                nested=True,
            )
            + _pandas_array_object_overhead(
                array.items.slice(start, value_count),
                data_type.item_type,
                nested=True,
            )
        )
    if nested:
        return row_count * _nested_pandas_scalar_overhead(data_type)
    return row_count * _pandas_object_cell_overhead(data_type)


def _list_value_bounds(array: pa.Array) -> tuple[int, int]:
    offsets = array.offsets
    return int(offsets[0].as_py()), int(offsets[-1].as_py())


def _nested_pandas_scalar_overhead(data_type: pa.DataType) -> int:
    if pa.types.is_dictionary(data_type):
        return _nested_pandas_scalar_overhead(data_type.value_type)
    direct = _pandas_object_cell_overhead(data_type)
    if direct:
        if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
            return direct + sys.getsizeof("")
        return direct
    if pa.types.is_boolean(data_type):
        return sys.getsizeof(False) + _PYTHON_POINTER_BYTES
    if pa.types.is_signed_integer(data_type) or pa.types.is_unsigned_integer(
        data_type
    ):
        return sys.getsizeof(0) + _PYTHON_POINTER_BYTES
    if pa.types.is_floating(data_type):
        return sys.getsizeof(0.0) + _PYTHON_POINTER_BYTES
    if pa.types.is_timestamp(data_type):
        return sys.getsizeof(pd.Timestamp(0)) + _PYTHON_POINTER_BYTES
    if pa.types.is_duration(data_type):
        return sys.getsizeof(pd.Timedelta(0)) + _PYTHON_POINTER_BYTES
    return sys.getsizeof(None) + (2 * _PYTHON_POINTER_BYTES)


def _pandas_object_cell_overhead(data_type: pa.DataType) -> int:
    if pa.types.is_null(data_type):
        return sys.getsizeof(None) + _PYTHON_POINTER_BYTES
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return _PYTHON_POINTER_BYTES
    if pa.types.is_dictionary(data_type):
        return _PYTHON_POINTER_BYTES
    if (
        pa.types.is_binary(data_type)
        or pa.types.is_large_binary(data_type)
        or pa.types.is_fixed_size_binary(data_type)
    ):
        return sys.getsizeof(b"") + _PYTHON_POINTER_BYTES
    if pa.types.is_decimal(data_type):
        return sys.getsizeof(Decimal(0)) + _PYTHON_POINTER_BYTES
    if pa.types.is_date(data_type):
        return sys.getsizeof(date.min) + _PYTHON_POINTER_BYTES
    if pa.types.is_time(data_type):
        return sys.getsizeof(time.min) + _PYTHON_POINTER_BYTES
    if (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
        or pa.types.is_map(data_type)
        or pa.types.is_struct(data_type)
    ):
        return sys.getsizeof([]) + _PYTHON_POINTER_BYTES
    return 0


def _fixed_width_arrow_row_bytes(data_type: pa.DataType) -> int | None:
    if pa.types.is_null(data_type):
        return 1
    if pa.types.is_boolean(data_type):
        return 1
    if pa.types.is_fixed_size_binary(data_type):
        return int(data_type.byte_width)
    if pa.types.is_decimal(data_type):
        return max(1, int(data_type.bit_width) // 8)
    if (
        pa.types.is_signed_integer(data_type)
        or pa.types.is_unsigned_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_date(data_type)
        or pa.types.is_time(data_type)
        or pa.types.is_timestamp(data_type)
        or pa.types.is_duration(data_type)
    ):
        return max(1, int(data_type.bit_width) // 8)
    return None


def _arrow_batch_to_bounded_pandas(
    batch: pa.RecordBatch,
    *,
    target_chunk_bytes: int,
    dtype_overrides: Mapping[str, object] | None = None,
) -> Iterator[pd.DataFrame]:
    if batch.num_rows == 0:
        return
    pandas_budget = max(1, target_chunk_bytes // _PHYSICAL_WORKING_SET_PARTS)
    overrides = dict(dtype_overrides or {})
    conversion_budget = (
        max(1, pandas_budget // 2) if overrides else pandas_budget
    )
    arrow_row_bytes = max(1, (batch.nbytes + batch.num_rows - 1) // batch.num_rows)
    expansion = (
        _VARIABLE_WIDTH_TO_PANDAS_EXPANSION_FACTOR
        if any(
            _requires_conservative_pandas_probe(field.type)
            for field in batch.schema
        )
        else _PARQUET_TO_PANDAS_EXPANSION_FACTOR
    )
    conversion_payload_bytes = max(
        1,
        conversion_budget
        - min(_PANDAS_FRAME_OVERHEAD_BYTES, max(0, conversion_budget - 1)),
    )
    rows_per_slice = max(
        1,
        min(
            batch.num_rows,
            _PANDAS_PROBE_ROWS,
            conversion_payload_bytes // max(1, arrow_row_bytes * expansion),
        ),
    )
    start = 0
    while start < batch.num_rows:
        stop = min(batch.num_rows, start + rows_per_slice)
        arrow_slice = batch.slice(start, stop - start)
        type_aware_bound = _pandas_conversion_bound_bytes(arrow_slice)
        conversion_bound_bytes = (
            type_aware_bound
            if len(arrow_slice) == 1
            else max(
                type_aware_bound,
                _PANDAS_FRAME_OVERHEAD_BYTES + arrow_slice.nbytes * expansion,
            )
        )
        if conversion_bound_bytes > conversion_budget:
            if len(arrow_slice) == 1:
                raise ValueError("dataset_row_exceeds_target_chunk_bytes")
            rows_per_slice = max(1, rows_per_slice // 2)
            continue
        frame = _arrow_slice_to_pandas(arrow_slice)
        frame_bytes = _pandas_frame_bytes(frame)
        if frame_bytes > conversion_budget:
            if len(frame) == 1:
                raise ValueError("dataset_row_exceeds_target_chunk_bytes")
            rows_per_slice = max(1, rows_per_slice // 2)
            continue
        output = frame.astype(overrides) if overrides else frame
        output_bytes = _pandas_frame_bytes(output)
        if frame_bytes + output_bytes > pandas_budget and output is not frame:
            if len(frame) == 1:
                raise ValueError("dataset_row_exceeds_target_chunk_bytes")
            rows_per_slice = max(1, rows_per_slice // 2)
            continue
        yield output
        start = stop
        if start >= batch.num_rows:
            continue
        measured_bytes = max(frame_bytes, output_bytes)
        measured_row_bytes = max(
            1,
            (measured_bytes + len(frame) - 1) // len(frame),
        )
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
    frame_bytes = _pandas_frame_bytes(frame)
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
        candidate_bytes = _pandas_frame_bytes(candidate)
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
