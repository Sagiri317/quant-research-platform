"""Dataset writing implementation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.json_codec import (
    canonical_identity_bytes,
    canonical_json_hash,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    DEFAULT_BATCH_ROWS,
)
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.datasets.observation import DatasetSummaryObserver
from quant_research_platform.storage.datasets.paths import (
    partition_frames,
    partition_part_path,
    relative_partition_label,
    timestamp_column_from_spec_payload,
    timestamp_primary_key_column_from_spec,
)
from quant_research_platform.storage.datasets.reading import (
    _arrow_batch_to_bounded_pandas,
    iter_frame_chunks,
)
from quant_research_platform.storage.datasets.schema import (
    arrow_type_has_null,
    arrow_type_resolves_null,
    data_spec_payload,
    encode_logical_category_value,
    logical_dtype_metadata_bytes,
    logical_dtype_payload_from_frame,
    merge_dataset_metadata,
    merge_logical_dtype_payloads,
    pandas_dtype_overrides_from_logical_payload,
    parse_logical_dtype_metadata,
    physical_schema_columns,
    physical_schema_from_schema,
    validate_frame_objects,
    validate_persisted_logical_schema,
)
from quant_research_platform.storage.datasets.types import DatasetInspection
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    filesystem_mkdir,
    native_filesystem_path,
    require_artifact_root,
)
from quant_research_platform.storage.manifests import write_internal_artifact_manifest
from quant_research_platform.storage.parquet import (
    inspect_dataset_part,
    parquet_table_writer,
)


@dataclass
class _WriteStats:
    spec: DataSpec
    data_root: Path
    key_tracker: ExactPrimaryKeyIndex
    row_count: int = 0
    file_count: int = 0
    file_size_bytes: int = 0
    partitions: set[str] = field(default_factory=set)
    partition_row_counts: dict[str, int] = field(default_factory=dict)
    columns: tuple[str, ...] | None = None
    schema_hash: str | None = None
    logical_dtypes: dict[str, Any] | None = None
    _schema: list[dict[str, Any]] | None = None
    _arrow_schema: pa.Schema | None = None
    _logical_dtype_plan: dict[str, Any] | None = None
    _dtype_plans: dict[bytes, tuple[dict[str, Any], dict[str, object]]] = field(
        default_factory=dict
    )
    _dtype_schemas: set[tuple[bytes, pa.Schema]] = field(default_factory=set)
    _frame_dtype_bytes: dict[bytes, bytes] = field(default_factory=dict)
    _summary_observer: DatasetSummaryObserver = field(init=False)

    def __post_init__(self) -> None:
        self._summary_observer = DatasetSummaryObserver(
            primary_key=tuple(self.spec.primary_key),
            timestamp_column=timestamp_primary_key_column_from_spec(self.spec),
            column_roles={
                column: role.value for column, role in self.spec.column_roles.items()
            },
            exact_key_index=self.key_tracker,
        )

    def declared_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate the input declaration without copying or Arrow conversion."""
        if type(frame) is not pd.DataFrame:
            raise TypeError("invalid_dataset_chunk")
        labels = list(frame.columns)
        if any(type(column) is not str or not column for column in labels) or len(
            labels
        ) != len(set(labels)):
            raise ValueError("invalid_dataset_column_labels")
        unregistered = sorted(set(frame.columns).difference(self.spec.column_roles))
        if unregistered:
            raise ValueError(f"unregistered_output_columns: {', '.join(unregistered)}")
        missing = sorted(set(self.spec.required_columns).difference(frame.columns))
        if missing:
            raise ValueError(f"missing_required_columns: {', '.join(missing)}")
        return frame

    def prepare_frame(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
        output = frame.loc[:, list(self.spec.required_columns)].copy()
        for column in self.spec.primary_key:
            if self.spec.column_roles.get(column) is ColumnRole.TIMESTAMP:
                output[column] = strict_timestamp_series(
                    output[column],
                    error="invalid_dataset_timestamp_value",
                    allow_null=True,
                )
        logical_dtypes = logical_dtype_payload_from_frame(output)
        return (output, logical_dtypes)

    def table_for_frame(
        self,
        frame: pd.DataFrame,
        *,
        schema: pa.Schema | None,
        object_string_columns: tuple[str, ...],
    ) -> pa.Table:
        try:
            table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as exc:
            raise ValueError("unsupported_dataset_logical_dtype") from exc
        for name in frame:
            dtype = frame[name].dtype
            if isinstance(dtype, pd.CategoricalDtype):
                values_type = pa.Array.from_pandas(dtype.categories).type
                if not (
                    pa.types.is_string(values_type)
                    or pa.types.is_large_string(values_type)
                ):
                    position = table.schema.get_field_index(name)
                    table = table.set_column(
                        position, name, table.column(position).cast(values_type)
                    )
        physical_schema = _canonical_storage_schema(table.schema.remove_metadata())
        if self._arrow_schema is None:
            for name in object_string_columns:
                field = physical_schema.field(name)
                if pa.types.is_null(field.type):
                    physical_schema = physical_schema.set(
                        physical_schema.get_field_index(name),
                        field.with_type(pa.string()),
                    )
            self._arrow_schema = physical_schema
        elif physical_schema != self._arrow_schema:
            try:
                unified = pa.unify_schemas(
                    [self._arrow_schema, physical_schema], promote_options="default"
                )
            except (pa.ArrowInvalid, pa.ArrowTypeError) as error:
                raise ValueError("dataset_schema_mismatch") from error
            if any(
                not arrow_type_resolves_null(old.type, new.type)
                for source in (self._arrow_schema, physical_schema)
                for old, new in zip(source, unified, strict=True)
            ):
                raise ValueError("dataset_schema_mismatch")
            self._arrow_schema = unified
            physical_schema = self._arrow_schema
        return table.cast(physical_schema, safe=True)

    def observe_file(
        self, path: Path, table: pa.Table, *, batch_rows: int = DEFAULT_BATCH_ROWS
    ) -> None:
        """Check the canonical stream and closed footer without decoding it again."""
        metadata = inspect_dataset_part(path)
        if metadata.schema != table.schema.remove_metadata():
            raise ValueError("dataset_schema_mismatch")
        if metadata.row_count != table.num_rows:
            raise ValueError("dataset_row_count_mismatch")
        raw = metadata.logical_dtype_payload
        if raw != (table.schema.metadata or {}).get(b"qrp.logical-dtypes"):
            raise ValueError("dataset_logical_dtype_mismatch")
        if raw not in self._dtype_plans:
            logical_payload = parse_logical_dtype_metadata(raw)
            self.observe_logical_dtype_payload(logical_payload)
            self._dtype_plans[raw] = (
                logical_payload,
                pandas_dtype_overrides_from_logical_payload(
                    logical_payload, columns=tuple(metadata.schema.names)
                ),
            )
        logical_payload, overrides = self._dtype_plans[raw]
        dtype_schema = (raw, metadata.schema)
        if dtype_schema not in self._dtype_schemas:
            validate_persisted_logical_schema(metadata.schema, logical_payload)
            self._dtype_schemas.add(dtype_schema)
        physical_schema = physical_schema_from_schema(metadata.schema)
        if self._schema is None:
            self._schema = physical_schema
            self.columns = physical_schema_columns(physical_schema)
            self.schema_hash = canonical_json_hash(physical_schema)
        elif physical_schema != self._schema:
            raise ValueError("dataset_schema_mismatch")
        for batch in table.to_batches(max_chunksize=batch_rows):
            for _ in _arrow_batch_to_bounded_pandas(
                batch,
                batch_rows=batch_rows,
                dtype_overrides=overrides,
                persisted_frame_observer=lambda frame: (
                    _validate_persisted_categorical_values(frame, logical_payload)
                ),
                batch_observer=lambda observed, frame: (
                    self._summary_observer.observe_batch(observed, frame=frame)
                ),
            ):
                pass
            _raise_accumulated_failures(self)
        self.row_count = self._summary_observer.row_count
        partition = relative_partition_label(self.data_root, path)
        self.file_count += 1
        self.file_size_bytes += metadata.file_size_bytes
        self.partitions.add(partition)
        self.partition_row_counts[partition] = (
            self.partition_row_counts.get(partition, 0) + metadata.row_count
        )

    def dtype_metadata(self, payload: dict[str, Any]) -> bytes:
        identity = canonical_identity_bytes(payload)
        if identity not in self._frame_dtype_bytes:
            self._frame_dtype_bytes[identity] = logical_dtype_metadata_bytes(payload)
        return self._frame_dtype_bytes[identity]

    def observe_logical_dtype_payload(self, logical_payload: Mapping[str, Any]) -> None:
        category_plan = {"schema_id": logical_payload["schema_id"], "columns": []}
        for descriptor in logical_payload["columns"]:
            current = dict(descriptor)
            if current["kind"] == "categorical":
                encoded_values = (
                    encode_logical_category_value(item)
                    for item in current["categories"]
                )
            if current["kind"] == "categorical" and (not current["ordered"]):
                self.key_tracker.add_identity_payloads(
                    namespace=_category_namespace(current["name"]),
                    payloads=encoded_values,
                )
                current["categories"] = []
            elif current["kind"] == "categorical":
                for _ in encoded_values:
                    pass
            category_plan["columns"].append(current)
        self._logical_dtype_plan = (
            category_plan
            if self._logical_dtype_plan is None
            else merge_logical_dtype_payloads([self._logical_dtype_plan, category_plan])
        )

    def payload(self) -> dict[str, Any]:
        if self.columns is None or self._schema is None or self.schema_hash is None:
            raise ValueError("dataset_schema_missing")
        self.logical_dtypes = self.materialized_logical_dtypes()
        payload = {}
        summary = self._summary_observer.summary_payload()
        payload.update(
            {
                "row_count": int(self.row_count),
                "file_count": int(self.file_count),
                "file_size_bytes": int(self.file_size_bytes),
                "columns": list(self.columns),
                "physical_schema": self._schema,
                "schema_hash": self.schema_hash,
                "logical_dtypes": self.logical_dtypes,
                "partition_row_counts": {
                    key: int(value)
                    for key, value in sorted(self.partition_row_counts.items())
                },
                "partition_count": len(self.partitions),
                "partitions": sorted(self.partitions),
                **summary,
            }
        )
        return payload

    def materialized_logical_dtypes(self) -> dict[str, Any]:
        if self._logical_dtype_plan is None:
            raise ValueError("dataset_logical_dtypes_missing")
        logical_dtypes = merge_logical_dtype_payloads([self._logical_dtype_plan])
        for descriptor in logical_dtypes["columns"]:
            if descriptor["kind"] == "categorical" and (not descriptor["ordered"]):
                descriptor["categories"] = [
                    parse_qrp_json_object(raw)["value"]
                    for raw in self.key_tracker.identity_payload_values(
                        namespace=_category_namespace(descriptor["name"])
                    )
                ]
        return logical_dtypes


def write_dataset(
    dataset_root: Path,
    data: pd.DataFrame | Iterable[pd.DataFrame],
    *,
    spec: DataSpec,
    metadata: Mapping[str, Any] | None = None,
) -> DatasetInspection:
    spec_payload = data_spec_payload(spec)
    final_root = Path(dataset_root)
    final_root = require_artifact_root(final_root)
    input_chunks = iter((data,)) if isinstance(data, pd.DataFrame) else iter(data)
    with closing_iterator(ExitStack()) as input_owner:
        close = getattr(input_chunks, "close", None)
        if close is not None:
            input_owner.callback(close)
        chunks = input_chunks

        def writer(temp_root: Path) -> DatasetInspection:
            key_tracker = ExactPrimaryKeyIndex(primary_key=spec.primary_key)
            accumulator = _write_contents(
                chunks,
                temp_root / "data",
                spec=spec,
                timestamp_column=timestamp_column_from_spec_payload(spec_payload),
                batch_rows=DEFAULT_BATCH_ROWS,
                key_tracker=key_tracker,
            )
            write_internal_artifact_manifest(temp_root / "spec.json", spec_payload)
            payload = accumulator.payload()
            payload["schema_id"] = "qrp-dataset-manifest"
            manifest = merge_dataset_metadata(payload, metadata)
            write_internal_artifact_manifest(
                temp_root / "_manifest" / "manifest.json", manifest
            )
            result = DatasetInspection(
                root=final_root,
                data_root=final_root / "data",
                spec=spec_payload,
                columns=tuple(manifest["columns"]),
                schema_hash=manifest["schema_hash"],
                file_count=manifest["file_count"],
                row_count=manifest["row_count"],
                partitions=tuple(manifest["partitions"]),
                partition_row_counts=manifest["partition_row_counts"],
                date_min=manifest["date_min"],
                date_max=manifest["date_max"],
                manifest=manifest,
            )
            input_owner.close()
            return result

        try:
            return atomic_write_directory(final_root, writer)
        except OutputAlreadyExistsError as exc:
            raise ValueError("dataset_exists") from exc


def _write_contents(
    chunks: Iterator[pd.DataFrame],
    root: Path,
    *,
    spec: DataSpec,
    timestamp_column: str | None,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    key_tracker: ExactPrimaryKeyIndex,
) -> _WriteStats:
    accumulator = _WriteStats(spec=spec, data_root=root, key_tracker=key_tracker)
    part_indices: dict[str, int] = {}
    pending_parts: list[Path] = []

    def observe_pending() -> None:
        schema = accumulator._arrow_schema
        assert schema is not None
        if any(arrow_type_has_null(field.type) for field in schema):
            return
        for pending in pending_parts:
            with closing_iterator(
                pq.ParquetFile(native_filesystem_path(pending))
            ) as parquet:
                table = parquet.read()
            resolved = table.cast(
                schema.with_metadata(table.schema.metadata), safe=True
            )
            write_parquet_file(resolved, pending, timestamp_column=timestamp_column)
            accumulator.observe_file(pending, resolved, batch_rows=batch_rows)
        pending_parts.clear()

    saw_empty_schema = False
    for chunk in chunks:
        declared = pd.DataFrame(accumulator.declared_frame(chunk), copy=False)
        object_string_columns = validate_frame_objects(declared)
        for bounded_frame in (
            (declared,)
            if declared.empty
            else iter_frame_chunks(declared, batch_rows=batch_rows)
        ):
            frame, logical_dtypes = accumulator.prepare_frame(bounded_frame)
            if frame.empty:
                accumulator.table_for_frame(
                    frame, schema=None, object_string_columns=object_string_columns
                )
                accumulator.observe_logical_dtype_payload(logical_dtypes)
                saw_empty_schema = True
                continue
            for partition, partition_frame in partition_frames(
                frame,
                timestamp_column=timestamp_column,
                partition_columns=spec.partition_columns,
            ):
                table = accumulator.table_for_frame(
                    partition_frame,
                    schema=None,
                    object_string_columns=object_string_columns,
                )
                persisted = table.replace_schema_metadata(
                    {b"qrp.logical-dtypes": accumulator.dtype_metadata(logical_dtypes)}
                )
                part_index = part_indices.get(partition, 0)
                part_indices[partition] = part_index + 1
                path = partition_part_path(root, partition, part_index)
                write_parquet_file(persisted, path, timestamp_column=timestamp_column)
                assert accumulator._arrow_schema is not None
                if any(
                    arrow_type_has_null(field.type)
                    for field in accumulator._arrow_schema
                ):
                    pending_parts.append(path)
                else:
                    observe_pending()
                    accumulator.observe_file(path, persisted, batch_rows=batch_rows)
                _raise_accumulated_failures(accumulator)
    if pending_parts:
        observe_pending()
    if pending_parts:
        raise ValueError("unsupported_dataset_arrow_type")
    if accumulator.row_count == 0 and (not saw_empty_schema):
        raise ValueError("empty_dataset_schema_missing")
    if saw_empty_schema:
        logical_dtypes = accumulator.materialized_logical_dtypes()
        partition = "" if accumulator.row_count == 0 else min(part_indices)
        path = partition_part_path(root, partition, part_indices.get(partition, 0))
        assert accumulator._arrow_schema is not None
        table = pa.Table.from_batches([], schema=accumulator._arrow_schema)
        table = table.replace_schema_metadata(
            {b"qrp.logical-dtypes": logical_dtype_metadata_bytes(logical_dtypes)}
        )
        write_parquet_file(table, path, timestamp_column=timestamp_column)
        accumulator.observe_file(path, table, batch_rows=batch_rows)
    return accumulator


def _raise_accumulated_failures(accumulator: _WriteStats) -> None:
    summary = accumulator._summary_observer
    if summary.duplicate_primary_key_count:
        raise ValueError("duplicate_dataset_primary_key")
    if any(int(count) > 0 for count in summary.null_count_summary.values()):
        raise ValueError("dataset_primary_key_null")
    if any(int(count) > 0 for count in summary.non_finite_count_summary.values()):
        raise ValueError("dataset_infinite_value")


def _category_namespace(column: str) -> str:
    return f"logical-category:{column}"


def _canonical_storage_schema(schema: pa.Schema) -> pa.Schema:
    return pa.schema(
        [
            pa.field(
                field.name,
                pa.dictionary(
                    pa.int32(),
                    pa.string()
                    if pa.types.is_large_string(field.type.value_type)
                    else field.type.value_type,
                    ordered=field.type.ordered,
                )
                if pa.types.is_dictionary(field.type)
                else pa.timestamp("ms", tz=field.type.tz)
                if pa.types.is_timestamp(field.type) and field.type.unit == "s"
                else field.type,
                nullable=field.nullable,
            )
            for field in schema
        ]
    )


def _validate_persisted_categorical_values(
    frame: pd.DataFrame, logical_dtypes: Mapping[str, Any]
) -> None:
    actual = logical_dtype_payload_from_frame(frame)
    actual_by_name = {item["name"]: item for item in actual["columns"]}
    for expected in logical_dtypes["columns"]:
        if expected["kind"] != "categorical":
            continue
        observed = actual_by_name.get(expected["name"])
        if observed is not None and observed["kind"] != "categorical":
            # Parquet decodes non-string dictionary values as their primitive type.
            # The shared batch converter checks their domain before casting.
            continue
        if (
            observed is None
            or observed["kind"] != "categorical"
            or observed["ordered"] != expected["ordered"]
        ):
            raise ValueError("dataset_categorical_domain_mismatch")
        expected_values = {
            canonical_identity_bytes(value) for value in expected["categories"]
        }
        observed_values = {
            canonical_identity_bytes(value) for value in observed["categories"]
        }
        if not observed_values.issubset(expected_values):
            raise ValueError("dataset_categorical_domain_mismatch")


def write_parquet_file(
    table: pa.Table, path: Path, *, timestamp_column: str | None = None
) -> Path:
    filesystem_mkdir(path.parent, parents=True, exist_ok=True)
    with parquet_table_writer(
        native_filesystem_path(path),
        compression="zstd",
        statistics=(timestamp_column,) if timestamp_column is not None else (),
    ) as write:
        write(table)
    return path
