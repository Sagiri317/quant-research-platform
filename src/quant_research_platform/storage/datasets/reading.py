"""Dataset reading implementation."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from quant_research_platform.contracts.json_codec import (
    canonical_json_hash,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_date_bounds,
    time_window_timestamp_interval,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import DEFAULT_BATCH_ROWS
from quant_research_platform.storage.datasets.inventory import (
    DatasetSessionInventory,
)
from quant_research_platform.storage.datasets.paths import (
    format_year_partition_label,
    parse_year_partition_label,
    relative_partition_label,
    timestamp_column_from_spec_payload,
)
from quant_research_platform.storage.datasets.schema import (
    logical_dtype_metadata_bytes,
    merge_logical_dtype_payloads,
    pandas_dtype_overrides_from_logical_payload,
    parse_logical_dtype_metadata,
    physical_schema_columns,
    physical_schema_from_schema,
    validate_dataset_manifest_consumable,
    validate_dataset_manifest_payload,
    validate_dataset_spec_payload,
    validate_persisted_logical_schema,
)
from quant_research_platform.storage.datasets.types import DatasetReadMode
from quant_research_platform.storage.identity import (
    DatasetFileIdentity,
    _file_identity_bytes,
    dataset_file_identity,
    filesystem_is_dir,
    filesystem_is_file,
    require_artifact_root,
    require_unaliased_path,
)
from quant_research_platform.storage.manifests import read_bounded_qrp_json
from quant_research_platform.storage.parquet import (
    DatasetPartMetadata,
    PreparedMembershipValues,
    _physical_filter_scalar,
    inspect_dataset_part,
    iter_dataset_part_batches,
)


@dataclass(frozen=True, slots=True)
class DatasetSession:
    root: Path
    data_root: Path
    spec_bytes: bytes
    spec_identity_bytes: bytes
    manifest_bytes: bytes
    manifest_identity_bytes: bytes
    inventory: DatasetSessionInventory
    logical_columns: tuple[str, ...]
    physical_columns: tuple[str, ...]
    timestamp_column: str | None
    time_window: TimeWindow | None
    schema_hash: str
    arrow_schema: pa.Schema
    logical_dtype_bytes: bytes
    parquet_filters: tuple[tuple[str, str, Any], ...] | None
    all_paths: tuple[Path, ...]
    read_paths: tuple[Path, ...]

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
        return self.inventory.metadata_for(path)

    def verify(self) -> None:
        spec_identity = dataset_file_identity(self.root / "spec.json")
        if _file_identity_bytes(spec_identity) != self.spec_identity_bytes:
            raise ValueError("dataset_spec_identity_changed")
        spec_bytes, _ = read_bounded_qrp_json(
            self.root / "spec.json", expected_identity=spec_identity
        )
        if spec_bytes != self.spec_bytes:
            raise ValueError("dataset_spec_identity_changed")
        manifest_identity = dataset_file_identity(
            self.root / "_manifest" / "manifest.json"
        )
        if _file_identity_bytes(manifest_identity) != self.manifest_identity_bytes:
            raise ValueError("dataset_manifest_identity_changed")
        manifest_bytes, _ = read_bounded_qrp_json(
            self.root / "_manifest" / "manifest.json",
            expected_identity=manifest_identity,
        )
        if manifest_bytes != self.manifest_bytes:
            raise ValueError("dataset_manifest_identity_changed")
        self.inventory.verify_filesystem(self.spec)

    def filtered_view(self, time_window: TimeWindow) -> DatasetSession:
        spec = self.spec
        return replace(
            self,
            read_paths=prune_paths_for_time_window(
                self.data_root, self.read_paths, spec=spec, time_window=time_window
            ),
            time_window=time_window,
            parquet_filters=(
                *(self.parquet_filters or ()),
                *(
                    parquet_filters_for_time_window(
                        timestamp_column=self.timestamp_column, time_window=time_window
                    )
                    or ()
                ),
            ),
            physical_columns=tuple(
                dict.fromkeys(
                    (
                        *self.physical_columns,
                        *read_columns_for_time_window(
                            self.logical_columns, spec=spec, time_window=time_window
                        ),
                    )
                )
            ),
        )


def read_dataset(
    dataset_root: Path,
    *,
    columns=None,
    time_window=None,
    mode=DatasetReadMode.MATERIALIZED,
):
    if mode is DatasetReadMode.CHUNKED:

        def activate():
            context = open_dataset_session(
                dataset_root, columns=columns, time_window=time_window
            )
            yield from iter_dataset_chunks(context, batch_rows=DEFAULT_BATCH_ROWS)

        return activate()
    if mode is not DatasetReadMode.MATERIALIZED:
        raise ValueError("unsupported_dataset_read_mode")
    context = open_dataset_session(
        dataset_root, columns=columns, time_window=time_window
    )
    return materialize_dataset_session(context, batch_rows=DEFAULT_BATCH_ROWS)


def materialize_dataset_session(
    context: DatasetSession, *, batch_rows: int = DEFAULT_BATCH_ROWS
) -> pd.DataFrame:
    """Materialize the validated stream once, preserving declared logical dtypes."""
    with closing_iterator(
        iter_dataset_chunks(context, batch_rows=batch_rows)
    ) as chunks:
        frames = list(chunks)
    return (
        pd.concat(frames, ignore_index=True)
        if len(frames) > 1
        else frames[0].reset_index(drop=True)
    )


def open_dataset_session(
    dataset_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
    row_filters: tuple[tuple[str, str, Any], ...] = (),
    require_consumable: bool = True,
) -> DatasetSession:
    root = require_artifact_root(Path(dataset_root))
    spec_path, data_root, manifest_path = (
        root / "spec.json",
        root / "data",
        root / "_manifest" / "manifest.json",
    )
    require_unaliased_path(spec_path, error="dataset_metadata_file_invalid")
    require_unaliased_path(data_root, error="invalid_dataset_part_files")
    require_unaliased_path(manifest_path, error="dataset_metadata_file_invalid")
    if not filesystem_is_file(spec_path):
        raise ValueError("dataset_spec_missing")
    if not filesystem_is_dir(data_root):
        raise ValueError("dataset_data_dir_missing")
    if not filesystem_is_file(manifest_path):
        raise ValueError("dataset_manifest_missing")
    try:
        spec_identity = dataset_file_identity(spec_path)
        manifest_identity = dataset_file_identity(manifest_path)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise ValueError("dataset_metadata_file_invalid") from exc
    spec_bytes, spec_payload = read_bounded_qrp_json(
        spec_path, expected_identity=spec_identity
    )
    if spec_payload.get("schema_id") != "qrp-dataset-spec":
        raise ValueError("dataset_spec_schema_id_invalid")
    validate_dataset_spec_payload(spec_payload)
    manifest_bytes, manifest = read_bounded_qrp_json(
        manifest_path, expected_identity=manifest_identity
    )
    if manifest.get("schema_id") != "qrp-dataset-manifest":
        raise ValueError("dataset_manifest_schema_id_invalid")
    validate_dataset_manifest_payload(manifest, spec=spec_payload)
    if require_consumable:
        validate_dataset_manifest_consumable(manifest)
    inventory = DatasetSessionInventory(data_root=data_root)
    inventory.discover(spec_payload)
    paths = inventory.paths
    if not paths:
        raise ValueError("dataset_data_dir_missing")
    for path in paths:
        inventory.observe(path, inspect_dataset_part(path))
    if inventory.content_hash() != manifest["content_hash"]:
        raise ValueError("dataset_content_identity_changed")
    session = _build_dataset_session(
        root=root,
        data_root=data_root,
        spec_bytes=spec_bytes,
        spec_identity_bytes=_file_identity_bytes(spec_identity),
        spec_payload=spec_payload,
        manifest_bytes=manifest_bytes,
        manifest_identity_bytes=_file_identity_bytes(manifest_identity),
        manifest=manifest,
        inventory=inventory,
        columns=columns,
        time_window=time_window,
        row_filters=row_filters,
    )
    return session


def dataset_session_from_inventory(
    dataset_root: Path,
    *,
    inventory: DatasetSessionInventory,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
) -> DatasetSession:
    """Freeze a writer receipt without reopening already observed Parquet parts."""
    root = require_artifact_root(Path(dataset_root))
    data_root, spec_path, manifest_path = (
        root / "data",
        root / "spec.json",
        root / "_manifest" / "manifest.json",
    )
    require_unaliased_path(data_root, error="invalid_dataset_part_files")
    require_unaliased_path(spec_path, error="dataset_metadata_file_invalid")
    require_unaliased_path(manifest_path, error="dataset_metadata_file_invalid")
    try:
        spec_identity = dataset_file_identity(spec_path)
        manifest_identity = dataset_file_identity(manifest_path)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise ValueError("dataset_metadata_file_invalid") from exc
    spec_bytes, spec_payload = read_bounded_qrp_json(
        spec_path, expected_identity=spec_identity
    )
    manifest_bytes, manifest = read_bounded_qrp_json(
        manifest_path, expected_identity=manifest_identity
    )
    validate_dataset_spec_payload(spec_payload)
    validate_dataset_manifest_payload(manifest, spec=spec_payload)
    validate_dataset_manifest_consumable(manifest)
    inventory.verify_filesystem(spec_payload)
    return _build_dataset_session(
        root=root,
        data_root=data_root,
        spec_bytes=spec_bytes,
        spec_identity_bytes=_file_identity_bytes(spec_identity),
        spec_payload=spec_payload,
        manifest_bytes=manifest_bytes,
        manifest_identity_bytes=_file_identity_bytes(manifest_identity),
        manifest=manifest,
        inventory=inventory,
        columns=columns,
        time_window=time_window,
        row_filters=(),
    )


def _build_dataset_session(
    *,
    root: Path,
    data_root: Path,
    spec_bytes: bytes,
    spec_identity_bytes: bytes,
    spec_payload: dict[str, Any],
    manifest_bytes: bytes,
    manifest_identity_bytes: bytes,
    manifest: dict[str, Any],
    inventory: DatasetSessionInventory,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
    row_filters: tuple[tuple[str, str, Any], ...],
) -> DatasetSession:
    all_paths = inventory.paths
    if not all_paths:
        raise ValueError("dataset_data_dir_missing")
    first_path = all_paths[0]
    first_metadata = inventory.metadata_for(first_path)
    schema = physical_schema_from_schema(first_metadata.schema)
    logical_dtypes: dict[str, Any] | None = None
    observed_rows = 0
    observed_file_bytes = 0
    observed_partition_rows: dict[str, int] = {}
    for path, metadata in inventory.iter_entries():
        if physical_schema_from_schema(metadata.schema) != schema:
            raise ValueError("dataset_schema_mismatch")
        current_dtypes = parse_logical_dtype_metadata(metadata.logical_dtype_payload)
        validate_persisted_logical_schema(metadata.schema, current_dtypes)
        logical_dtypes = (
            merge_logical_dtype_payloads([current_dtypes])
            if logical_dtypes is None
            else merge_logical_dtype_payloads([logical_dtypes, current_dtypes])
        )
        observed_rows += metadata.row_count
        observed_file_bytes += metadata.file_size_bytes
        partition = relative_partition_label(data_root, path)
        observed_partition_rows[partition] = (
            observed_partition_rows.get(partition, 0) + metadata.row_count
        )
        if (
            tuple(spec_payload["partition_columns"]) == ("year",)
            and path.parent == data_root
            and (metadata.row_count != 0)
        ):
            raise ValueError("invalid_dataset_part_files")
    assert logical_dtypes is not None
    schema_columns = physical_schema_columns(schema)
    schema_hash = canonical_json_hash(schema)
    if int(manifest["file_count"]) != len(all_paths):
        raise ValueError("manifest_file_count_mismatch")
    if set(manifest["partition_row_counts"]) != set(observed_partition_rows):
        raise ValueError("manifest_partition_row_counts_mismatch")
    if int(manifest["row_count"]) != observed_rows:
        raise ValueError("manifest_row_count_mismatch")
    if dict(sorted(observed_partition_rows.items())) != dict(
        sorted(manifest["partition_row_counts"].items())
    ):
        raise ValueError("manifest_partition_row_counts_mismatch")
    if int(manifest["file_size_bytes"]) != observed_file_bytes:
        raise ValueError("manifest_file_size_bytes_mismatch")
    if manifest["logical_dtypes"] != logical_dtypes:
        raise ValueError("manifest_logical_dtypes_mismatch")
    logical_dtype_bytes = logical_dtype_metadata_bytes(logical_dtypes)
    if tuple(str(column) for column in manifest["columns"]) != schema_columns:
        raise ValueError("manifest_columns_mismatch")
    if manifest["physical_schema"] != schema:
        raise ValueError("manifest_physical_schema_mismatch")
    if str(manifest["schema_hash"]) != schema_hash:
        raise ValueError("manifest_schema_hash_mismatch")
    logical_columns = resolve_columns_from_schema(
        schema_columns, spec=spec_payload, columns=columns
    )
    timestamp_column = timestamp_column_from_spec_payload(spec_payload)
    physical_columns = read_columns_for_time_window(
        logical_columns, spec=spec_payload, time_window=time_window
    )
    physical_columns = tuple(
        dict.fromkeys((*physical_columns, *(column for column, _, _ in row_filters)))
    )
    physical_filters = (
        parquet_filters_for_time_window(
            timestamp_column=timestamp_column, time_window=time_window
        )
        or []
    )
    for column, operator, value in row_filters:
        if column not in schema_columns:
            raise ValueError("dataset_filter_column_not_found")
        if operator not in {"==", "in"}:
            raise ValueError("unsupported_dataset_physical_filter")
        field = first_metadata.schema.field(column)
        if operator == "in":
            if type(value) is not tuple or not value:
                raise ValueError("invalid_dataset_physical_filter_value")
            validated_value = _prepare_in_filter(field, value)
        else:
            validated_value = _validated_physical_filter_value(field, value)
        physical_filters.append((column, operator, validated_value))
    session = DatasetSession(
        all_paths=all_paths,
        read_paths=prune_paths_for_time_window(
            data_root, all_paths, spec=spec_payload, time_window=time_window
        ),
        root=root,
        data_root=data_root,
        spec_bytes=spec_bytes,
        spec_identity_bytes=spec_identity_bytes,
        manifest_bytes=manifest_bytes,
        manifest_identity_bytes=manifest_identity_bytes,
        inventory=inventory,
        logical_columns=logical_columns,
        physical_columns=physical_columns,
        timestamp_column=timestamp_column,
        time_window=time_window,
        schema_hash=schema_hash,
        arrow_schema=first_metadata.schema,
        logical_dtype_bytes=logical_dtype_bytes,
        parquet_filters=tuple(
            (
                (column, operator, _freeze_filter_value(value))
                for column, operator, value in physical_filters
            )
        )
        or None,
    )
    session.verify()
    return session


def iter_dataset_chunks(
    context: DatasetSession,
    *,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[pd.DataFrame]:
    rows_per_batch = _validated_batch_rows(batch_rows)
    if not context.read_paths:
        try:
            empty = _empty_frame(context, context.logical_columns)
            try:
                yield empty
            except GeneratorExit:
                return
        finally:
            _verify_preserving_primary(context)
        return
    yielded = False
    with closing_iterator(
        iter_dataset_file_batches(
            context,
            columns=context.physical_columns,
            batch_rows=rows_per_batch,
            batch_observer=batch_observer,
        )
    ) as file_batches:
        for _, frame in file_batches:
            filtered = frame.loc[:, list(context.logical_columns)]
            if filtered.empty:
                continue
            yielded = True
            try:
                yield filtered
            except GeneratorExit:
                return
    if not yielded:
        try:
            empty = _empty_frame(context, context.logical_columns)
            try:
                yield empty
            except GeneratorExit:
                return
        finally:
            _verify_preserving_primary(context)


def iter_declared_parquet_member_batches(
    path: Path,
    *,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    expected_identity: DatasetFileIdentity,
    columns: tuple[str, ...] | None = None,
    parquet_filters: list[tuple[str, str, Any]] | None = None,
    dtype_overrides: Mapping[str, object] | None = None,
    metadata_observer: Callable[[DatasetPartMetadata], None] | None = None,
    persisted_frame_observer: Callable[[pd.DataFrame], None] | None = None,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[pd.DataFrame]:
    """Traverse one frozen member through the sole logical Pandas converter."""
    rows_per_batch = _validated_batch_rows(batch_rows)
    with closing_iterator(
        iter_dataset_part_batches(
            Path(path),
            expected_identity=expected_identity,
            batch_rows=rows_per_batch,
            columns=columns,
            parquet_filters=parquet_filters,
            metadata_observer=metadata_observer,
        )
    ) as batches:
        for batch in batches:
            # Observer failures remain outside the consumer-close boundary.
            for frame in _arrow_batch_to_bounded_pandas(
                batch,
                batch_rows=rows_per_batch,
                dtype_overrides=dtype_overrides,
                persisted_frame_observer=persisted_frame_observer,
                batch_observer=batch_observer,
            ):
                try:
                    yield frame
                except GeneratorExit:
                    return


def iter_dataset_file_batches(
    context: DatasetSession,
    *,
    columns: tuple[str, ...],
    batch_rows: int = DEFAULT_BATCH_ROWS,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """Yield deterministic file batches after Arrow-level bounded conversion."""
    rows_per_batch = _validated_batch_rows(batch_rows)
    dtype_overrides = pandas_dtype_overrides_from_logical_payload(
        context.logical_dtypes, columns=columns
    )
    context.verify()
    try:
        for path in context.read_paths:
            metadata = context.metadata_for(path)
            with closing_iterator(
                iter_declared_parquet_member_batches(
                    path,
                    expected_identity=metadata.identity,
                    columns=columns,
                    parquet_filters=context.parquet_filters,
                    batch_rows=rows_per_batch,
                    dtype_overrides=dtype_overrides,
                    batch_observer=batch_observer,
                )
            ) as member_batches:
                while True:
                    try:
                        frame = next(member_batches)
                    except StopIteration:
                        break
                    value = (path, frame)
                    try:
                        yield value
                    except GeneratorExit:
                        return
    finally:
        _verify_preserving_primary(context)


def _verify_preserving_primary(context: DatasetSession) -> None:
    primary = sys.exception()
    if primary is None:
        context.verify()
        return
    try:
        context.verify()
    except BaseException as verification_error:
        primary.add_note(
            f"dataset verification also failed: {type(verification_error).__name__}"
        )


def dataset_primary_key_schema(
    context: DatasetSession, primary_key: tuple[str, ...]
) -> pa.Schema:
    try:
        return pa.schema([context.arrow_schema.field(column) for column in primary_key])
    except KeyError as exc:
        raise ValueError("missing_dataset_primary_key_column") from exc


def _validated_batch_rows(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("invalid_batch_rows")
    return value


def _freeze_filter_value(value: Any) -> Any:
    if isinstance(value, PreparedMembershipValues):
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_filter_value(item) for item in value)
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


def _filter_value_is_representable(field: pa.Field, value: Any) -> bool:
    source = _physical_filter_scalar(field, value)
    try:
        converted = source.cast(field.type, safe=True)
    except pa.ArrowInvalid:
        return False
    return bool(pc.equal(source, converted).as_py())


def _prepare_in_filter(
    field: pa.Field, values: tuple[Any, ...]
) -> PreparedMembershipValues:
    for value in values:
        _validated_physical_filter_value(field, value)
    try:
        selected = pa.array(
            [value for value in values if _filter_value_is_representable(field, value)],
            type=field.type,
        )
        unique = pc.unique(selected)
        return PreparedMembershipValues(pc.take(unique, pc.sort_indices(unique)))
    except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError):
        raise ValueError("invalid_dataset_physical_filter_value") from None


def _validated_physical_filter_value(field: pa.Field, value: Any) -> Any:
    frozen = _freeze_filter_value(value)
    data_type = field.type
    if frozen is pd.NaT or isinstance(frozen, np.generic):
        raise ValueError("invalid_dataset_physical_filter_value")
    if isinstance(frozen, (float, Decimal)) and (not math.isfinite(frozen)):
        raise ValueError("invalid_dataset_physical_filter_value")
    if pa.types.is_boolean(data_type):
        valid = type(frozen) is bool
    elif pa.types.is_integer(data_type):
        valid = type(frozen) is int
    elif pa.types.is_floating(data_type):
        valid = type(frozen) is float
    elif pa.types.is_decimal(data_type):
        valid = type(frozen) is Decimal
    elif pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        valid = type(frozen) is str
        if valid:
            try:
                frozen.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError("invalid_dataset_physical_filter_value") from None
    elif (
        pa.types.is_binary(data_type)
        or pa.types.is_large_binary(data_type)
        or pa.types.is_fixed_size_binary(data_type)
    ):
        valid = type(frozen) is bytes
    elif pa.types.is_timestamp(data_type):
        valid = type(frozen) in (pd.Timestamp, datetime)
    elif pa.types.is_date(data_type):
        valid = type(frozen) is date
    elif pa.types.is_time32(data_type) or pa.types.is_time64(data_type):
        valid = type(frozen) is time
    elif pa.types.is_duration(data_type):
        valid = type(frozen) in (pd.Timedelta, timedelta)
    else:
        valid = False
    if not valid:
        raise ValueError("invalid_dataset_physical_filter_value")
    try:
        pa.scalar(frozen, type=data_type)
    except (
        pa.ArrowInvalid,
        pa.ArrowNotImplementedError,
        pa.ArrowTypeError,
        OverflowError,
    ):
        raise ValueError("invalid_dataset_physical_filter_value") from None
    return frozen


def _empty_frame(context: DatasetSession, columns: tuple[str, ...]) -> pd.DataFrame:
    selected = pa.schema([context.arrow_schema.field(column) for column in columns])
    overrides = pandas_dtype_overrides_from_logical_payload(
        context.logical_dtypes, columns=columns
    )
    frame = _arrow_slice_to_pandas(
        pa.RecordBatch.from_arrays(
            [pa.array([], type=field.type) for field in selected], schema=selected
        ),
        dtype_overrides=overrides,
    )
    output = frame.astype(overrides) if overrides else frame
    return output


def _arrow_batch_to_bounded_pandas(
    batch: pa.RecordBatch,
    *,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    dtype_overrides: Mapping[str, object] | None = None,
    persisted_frame_observer: Callable[[pd.DataFrame], None] | None = None,
    batch_observer: Callable[[pa.RecordBatch, pd.DataFrame], None] | None = None,
) -> Iterator[pd.DataFrame]:
    if batch.num_rows == 0:
        return
    target = _validated_batch_rows(batch_rows)
    rows = target
    overrides = dict(dtype_overrides or {})
    categorical = {
        name: dtype
        for name, value in overrides.items()
        if isinstance((dtype := pd.api.types.pandas_dtype(value)), pd.CategoricalDtype)
    }
    ordinary = {
        name: value for name, value in overrides.items() if name not in categorical
    }
    for start in range(0, batch.num_rows, rows):
        arrow_slice = batch.slice(start, rows)
        frame = _arrow_slice_to_pandas(arrow_slice, dtype_overrides=overrides)
        if persisted_frame_observer is not None:
            persisted_frame_observer(frame)
        output = frame.astype(ordinary) if ordinary else frame
        for name, dtype in categorical.items():
            codes = dtype.categories.get_indexer(frame[name])
            if np.any((codes == -1) & frame[name].notna().to_numpy()):
                raise ValueError("dataset_categorical_domain_mismatch")
            output[name] = pd.Categorical.from_codes(codes, dtype=dtype, validate=False)
        if batch_observer is not None:
            batch_observer(arrow_slice, output)
        yield output


def _arrow_slice_to_pandas(
    arrow_slice: pa.RecordBatch, *, dtype_overrides: Mapping[str, object] | None = None
) -> pd.DataFrame:
    native_types = set()
    nullable_float_columns = []
    for name, dtype in (dtype_overrides or {}).items():
        dtype = pd.api.types.pandas_dtype(dtype)
        data_type = arrow_slice.schema.field(name).type
        if pa.types.is_timestamp(data_type) and pd.api.types.is_datetime64_any_dtype(
            dtype
        ):
            expected = pa.Array.from_pandas(pd.Series([], dtype=dtype)).type
            if data_type != expected:
                try:
                    values = arrow_slice.column(name).cast(expected, safe=True)
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    raise ValueError("dataset_logical_dtype_mismatch") from None
                arrow_slice = arrow_slice.set_column(
                    arrow_slice.schema.get_field_index(name), name, values
                )
        if pa.types.is_integer(data_type) or pa.types.is_boolean(data_type):
            native_types.add(data_type)
        if isinstance(dtype, (pd.Float32Dtype, pd.Float64Dtype)):
            native_types.add(data_type)
            nullable_float_columns.append(name)
        if (
            isinstance(dtype, np.dtype)
            and dtype.kind in "biu"
            and arrow_slice.column(name).null_count
        ):
            raise ValueError("dataset_nonnullable_value_missing")
    frame = arrow_slice.to_pandas(
        use_threads=True,
        integer_object_nulls=True,
        types_mapper=(
            lambda data_type: (
                pd.ArrowDtype(data_type) if data_type in native_types else None
            )
        )
        if native_types
        else None,
    )
    for name in nullable_float_columns:
        values = arrow_slice.column(name)
        frame[name] = pd.arrays.FloatingArray(
            values.to_numpy(zero_copy_only=False),
            values.is_null().to_numpy(zero_copy_only=False),
            copy=False,
        )
    return frame


def iter_frame_chunks(
    frame: pd.DataFrame, *, batch_rows: int = DEFAULT_BATCH_ROWS
) -> Iterator[pd.DataFrame]:
    rows = _validated_batch_rows(batch_rows)
    for start in range(0, len(frame), rows):
        yield frame.iloc[start : start + rows]


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
        format_year_partition_label(year)
        for year in range(int(start.year), int(end.year) + 1)
    }
    return tuple(
        path
        for path in paths
        if path.parent == Path(data_root)
        or format_year_partition_label(
            parse_year_partition_label(relative_partition_label(data_root, path))
        )
        in years
    )


def parquet_filters_for_time_window(
    *, timestamp_column: str | None, time_window: TimeWindow | None
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
        requested = tuple(spec["required_columns"])
    else:
        if type(columns) not in (tuple, list):
            raise ValueError("dataset_columns_invalid")
        if not columns:
            raise ValueError("dataset_columns_empty")
        if any(type(column) is not str or not column for column in columns) or len(
            columns
        ) != len(set(columns)):
            raise ValueError("dataset_columns_invalid")
        requested = tuple(columns)
    missing = [column for column in requested if column not in schema_columns]
    if missing:
        raise ValueError("dataset_column_not_found")
    return requested


def read_columns_for_time_window(
    columns: tuple[str, ...], *, spec: Mapping[str, Any], time_window: TimeWindow | None
) -> tuple[str, ...]:
    if time_window is None:
        return columns
    timestamp_column = timestamp_column_from_spec_payload(spec)
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    return tuple(dict.fromkeys((*columns, timestamp_column)))
