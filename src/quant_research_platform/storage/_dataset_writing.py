"""Dataset writing implementation."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_platform.contracts._json import (
    canonical_identity_bytes,
    canonical_json_hash,
    parse_qrp_json_object,
)
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage._dataset_inspection import (
    _inspection_from_persisted_session,
)
from quant_research_platform.storage._dataset_inventory import DatasetSessionInventory
from quant_research_platform.storage._dataset_key_tracking import (
    EXACT_KEY_SQLITE_PAGE_BYTES,
    ExactPrimaryKeyIndex,
)
from quant_research_platform.storage._dataset_observation import (
    DatasetSummaryObserver,
)
from quant_research_platform.storage._dataset_paths import (
    partition_frames,
    partition_part_path,
    relative_partition_label,
    timestamp_primary_key_column_from_spec,
    validate_dataset_artifact_root,
)
from quant_research_platform.storage._dataset_physical import (
    DatasetPartMetadata,
    dataset_file_identity,
)
from quant_research_platform.storage._dataset_reading import (
    DatasetSession,
    _bounded_frame_chunks,
    dataset_session_from_inventory,
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage._dataset_schema import (
    _encoded_logical_category_value,
    data_spec_payload,
    logical_dtype_metadata_bytes,
    logical_dtype_payload_from_frame,
    merge_dataset_instance_metadata,
    merge_logical_dtype_payloads,
    pandas_dtype_overrides_from_logical_payload,
    parse_logical_dtype_metadata,
    physical_schema_columns,
    physical_schema_from_schema,
    require_frame_logical_dtype_metadata_within_target,
    require_logical_dtype_metadata_within_target,
    validate_dataset_spec,
)
from quant_research_platform.storage._dataset_types import DatasetInspection
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.manifests import write_internal_artifact_manifest


@dataclass(frozen=True, slots=True)
class DatasetWriteReceipt:
    root: Path
    session: DatasetSession
    exact_primary_key_index: ExactPrimaryKeyIndex


@dataclass
class _WriteStats:
    spec: DataSpec
    data_root: Path
    key_tracker: ExactPrimaryKeyIndex
    inventory: DatasetSessionInventory
    row_count: int = 0
    file_count: int = 0
    file_size_bytes: int = 0
    partitions: set[str] = field(default_factory=set)
    partition_row_counts: dict[str, int] = field(default_factory=dict)
    columns: tuple[str, ...] | None = None
    schema_hash: str | None = None
    logical_dtypes: dict[str, Any] | None = None
    logical_dtype_hash: str | None = None
    _schema: list[dict[str, Any]] | None = None
    _arrow_schema: pa.Schema | None = None
    _logical_dtype_plan: dict[str, Any] | None = None
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

    def declared_frame(
        self,
        frame: pd.DataFrame,
        *,
        target_chunk_bytes: int,
    ) -> pd.DataFrame:
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
        require_frame_logical_dtype_metadata_within_target(
            frame,
            columns=tuple(self.spec.required_columns),
            target_chunk_bytes=target_chunk_bytes,
        )
        return frame

    def prepare_frame(
        self,
        frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        output = frame.loc[:, list(self.spec.required_columns)].copy()
        for column in self.spec.primary_key:
            if self.spec.column_roles.get(column) is ColumnRole.TIMESTAMP:
                output[column] = strict_timestamp_series(
                    output[column],
                    error="invalid_dataset_timestamp_value",
                    allow_null=True,
                )
        logical_dtypes = logical_dtype_payload_from_frame(output)
        return output, logical_dtypes

    def table_for_frame(self, frame: pd.DataFrame) -> pa.Table:
        try:
            table = pa.Table.from_pandas(frame, preserve_index=False)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as exc:
            raise ValueError("unsupported_dataset_logical_dtype") from exc
        physical_schema = _canonical_storage_schema(table.schema.remove_metadata())
        if self._arrow_schema is None:
            self._arrow_schema = physical_schema
        elif physical_schema != self._arrow_schema:
            raise ValueError("dataset_schema_mismatch")
        return table.cast(physical_schema, safe=True)

    def observe_file(
        self,
        path: Path,
        *,
        target_chunk_bytes: int,
        operation_headroom_bytes: int,
    ) -> None:
        observed_metadata: DatasetPartMetadata | None = None
        observed_logical_dtypes: dict[str, Any] | None = None
        dtype_overrides: dict[str, object] = {}
        observed_rows = 0

        def observe_metadata(metadata: DatasetPartMetadata) -> None:
            nonlocal observed_metadata, observed_logical_dtypes
            observed_metadata = metadata
            logical_payload = parse_logical_dtype_metadata(
                metadata.logical_dtype_payload
            )
            observed_logical_dtypes = logical_payload
            self.observe_logical_dtype_payload(
                logical_payload,
                target_chunk_bytes=target_chunk_bytes,
            )
            dtype_overrides.update(
                pandas_dtype_overrides_from_logical_payload(
                    logical_payload,
                    columns=tuple(metadata.schema.names),
                )
            )
            physical_schema = physical_schema_from_schema(metadata.schema)
            _validate_persisted_logical_schema(metadata.schema, logical_payload)
            if self._schema is None:
                self._schema = physical_schema
                self.columns = physical_schema_columns(physical_schema)
                self.schema_hash = canonical_json_hash(physical_schema)
            elif physical_schema != self._schema:
                raise ValueError("dataset_schema_mismatch")

        def observe_persisted_frame(persisted_frame: pd.DataFrame) -> None:
            assert observed_logical_dtypes is not None
            _validate_persisted_categorical_values(
                persisted_frame,
                observed_logical_dtypes,
            )

        def observe_batch(batch: pa.RecordBatch, logical_frame: pd.DataFrame) -> None:
            nonlocal observed_rows
            observed_rows += int(batch.num_rows)
            self._summary_observer.observe_batch(batch, frame=logical_frame)
            self.row_count = self._summary_observer.row_count
            _raise_accumulated_failures(self)

        for _ in iter_declared_parquet_member_batches(
            path,
            expected_identity=dataset_file_identity(path),
            target_chunk_bytes=target_chunk_bytes,
            operation_headroom_bytes=operation_headroom_bytes,
            metadata_observer=observe_metadata,
            dtype_overrides=dtype_overrides,
            persisted_frame_observer=observe_persisted_frame,
            batch_observer=observe_batch,
        ):
            pass
        if observed_metadata is None:
            raise AssertionError("dataset_part_metadata_not_observed")
        if observed_rows != observed_metadata.row_count:
            raise ValueError("dataset_row_count_mismatch")
        partition = relative_partition_label(self.data_root, path)
        self.file_count += 1
        self.file_size_bytes += observed_metadata.file_size_bytes
        self.partitions.add(partition)
        self.partition_row_counts[partition] = (
            self.partition_row_counts.get(partition, 0) + observed_rows
        )
        self.inventory.add(path, observed_metadata, selected=True)

    def observe_logical_dtype_payload(
        self,
        logical_payload: Mapping[str, Any],
        *,
        target_chunk_bytes: int,
    ) -> None:
        spillable_payload = {
            "schema_id": logical_payload["schema_id"],
            "columns": [],
        }
        for descriptor in logical_payload["columns"]:
            current = dict(descriptor)
            if current["kind"] == "categorical":
                encoded_values = (
                    _encoded_logical_category_value(
                        item,
                        target_chunk_bytes=target_chunk_bytes,
                    )
                    for item in current["categories"]
                )
            if current["kind"] == "categorical" and not current["ordered"]:
                self.key_tracker.add_identity_payloads(
                    namespace=_category_namespace(current["name"]),
                    payloads=encoded_values,
                )
                current["categories"] = []
            elif current["kind"] == "categorical":
                for _ in encoded_values:
                    pass
            spillable_payload["columns"].append(current)
        self._logical_dtype_plan = (
            spillable_payload
            if self._logical_dtype_plan is None
            else merge_logical_dtype_payloads(
                [self._logical_dtype_plan, spillable_payload]
            )
        )

    def payload(self, *, target_chunk_bytes: int) -> dict[str, Any]:
        if self.columns is None or self._schema is None or self.schema_hash is None:
            raise ValueError("dataset_schema_missing")
        self.logical_dtypes = self.materialized_logical_dtypes(
            target_chunk_bytes=target_chunk_bytes
        )
        self.logical_dtype_hash = canonical_json_hash(self.logical_dtypes)
        payload = data_spec_payload(self.spec)
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
                "logical_dtype_hash": self.logical_dtype_hash,
                "partition_row_counts": {
                    key: int(value)
                    for key, value in sorted(self.partition_row_counts.items())
                },
                "partition_count": len(self.partitions),
                "partitions": sorted(self.partitions),
                **summary,
                "validation_status": "PASS",
            }
        )
        return payload

    def materialized_logical_dtypes(
        self,
        *,
        target_chunk_bytes: int,
    ) -> dict[str, Any]:
        if self._logical_dtype_plan is None:
            raise ValueError("dataset_logical_dtypes_missing")
        logical_dtypes = merge_logical_dtype_payloads([self._logical_dtype_plan])

        def streamed_category_payloads(
            descriptor: Mapping[str, Any],
        ) -> Iterable[Mapping[str, Any]]:
            if descriptor["ordered"]:
                return iter(descriptor["categories"])
            return (
                parse_qrp_json_object(raw)["value"]
                for raw in self.key_tracker.identity_payload_values(
                    namespace=_category_namespace(str(descriptor["name"]))
                )
            )

        exact_metadata_bytes = require_logical_dtype_metadata_within_target(
            logical_dtypes,
            category_payloads=streamed_category_payloads,
            target_chunk_bytes=target_chunk_bytes,
        )
        for descriptor in logical_dtypes["columns"]:
            if descriptor["kind"] != "categorical" or descriptor["ordered"]:
                continue
            descriptor["categories"] = [
                parse_qrp_json_object(raw)["value"]
                for raw in self.key_tracker.identity_payload_values(
                    namespace=_category_namespace(descriptor["name"])
                )
            ]
        if len(logical_dtype_metadata_bytes(logical_dtypes)) != exact_metadata_bytes:
            raise AssertionError("dataset_logical_dtype_metadata_size_mismatch")
        return logical_dtypes


def write_dataset(
    dataset_root: Path,
    data: pd.DataFrame | Iterable[pd.DataFrame],
    *,
    spec: DataSpec,
    instance_metadata: Mapping[str, Any] | None = None,
    staged_validator: Callable[[DatasetWriteReceipt], None] | None = None,
    resource_policy: RuntimeResourcePolicy | None,
) -> DatasetInspection:
    final_root = Path(dataset_root)
    validate_dataset_spec(spec)
    validate_dataset_artifact_root(final_root, spec)
    issued_policy = resource_policy or active_runtime_resource_policy()
    input_chunks = iter((data,)) if isinstance(data, pd.DataFrame) else iter(data)
    close = getattr(input_chunks, "close", None)
    input_closed = close is None

    def close_input_once() -> None:
        nonlocal input_closed
        if input_closed:
            return
        input_closed = True
        if sys.exception() is None:
            close()
            return
        try:
            close()
        except BaseException:
            pass

    def _writer(temp_root: Path) -> DatasetInspection:
        try:
            timestamp_columns = tuple(
                column
                for column in spec.primary_key
                if spec.column_roles.get(column) is ColumnRole.TIMESTAMP
            )
            with (
                ExactPrimaryKeyIndex(
                    primary_key=spec.primary_key,
                    timestamp_columns=timestamp_columns,
                    scratch_parent=temp_root,
                    sqlite_cache_bytes=EXACT_KEY_SQLITE_PAGE_BYTES,
                ) as key_tracker,
                DatasetSessionInventory(
                    data_root=temp_root / "data",
                    scratch_parent=temp_root,
                    task_memory_budget_bytes=issued_policy.task_memory_budget_bytes,
                ) as inventory,
            ):
                working_headroom = issued_policy.task_memory_budget_bytes - (
                    key_tracker.retained_memory_bytes + inventory.retained_memory_bytes
                )
                if working_headroom <= 0:
                    raise ValueError("dataset_working_set_exceeds_task_memory_budget")
                accumulator = _write_contents(
                    input_chunks,
                    temp_root / "data",
                    spec=spec,
                    target_chunk_bytes=issued_policy.payload_target_bytes,
                    task_memory_budget_bytes=working_headroom,
                    key_tracker=key_tracker,
                    inventory=inventory,
                )
                write_internal_artifact_manifest(
                    temp_root / "spec.json",
                    data_spec_payload(spec),
                )
                manifest = merge_dataset_instance_metadata(
                    {
                        **accumulator.payload(
                            target_chunk_bytes=issued_policy.payload_target_bytes
                        ),
                        "schema_id": "qrp-dataset-manifest",
                    },
                    instance_metadata,
                )
                write_internal_artifact_manifest(
                    temp_root / "_manifest" / "manifest.json",
                    manifest,
                )
                session = dataset_session_from_inventory(
                    temp_root,
                    inventory=inventory,
                    columns=None,
                    time_window=None,
                    resource_policy=issued_policy,
                )
                if staged_validator is not None:
                    staged_validator(
                        DatasetWriteReceipt(
                            root=temp_root,
                            session=session,
                            exact_primary_key_index=key_tracker,
                        )
                    )
                session.verify()
                return _inspection_from_persisted_session(
                    session,
                    published_root=final_root,
                )
        finally:
            close_input_once()

    try:
        try:
            return atomic_write_directory(final_root, _writer)
        except OutputAlreadyExistsError as exc:
            raise ValueError("dataset_exists") from exc
    finally:
        close_input_once()


def _write_contents(
    chunks: Iterator[pd.DataFrame],
    root: Path,
    *,
    spec: DataSpec,
    target_chunk_bytes: int,
    task_memory_budget_bytes: int,
    key_tracker: ExactPrimaryKeyIndex,
    inventory: DatasetSessionInventory,
) -> _WriteStats:
    timestamp_column = validate_dataset_spec(spec)
    accumulator = _WriteStats(
        spec=spec,
        data_root=root,
        key_tracker=key_tracker,
        inventory=inventory,
    )
    part_indices: dict[str, int] = {}
    saw_empty_schema = False
    for chunk in chunks:
        declared = accumulator.declared_frame(
            chunk,
            target_chunk_bytes=target_chunk_bytes,
        )
        if declared.empty:
            frame, logical_dtypes = accumulator.prepare_frame(declared)
            table = accumulator.table_for_frame(frame)
            _require_writer_headroom(
                task_memory_budget_bytes,
                declared,
                frame,
                arrow_bytes=table.nbytes,
            )
            accumulator.observe_logical_dtype_payload(
                logical_dtypes,
                target_chunk_bytes=target_chunk_bytes,
            )
            saw_empty_schema = True
            del table, frame, logical_dtypes, declared, chunk
            continue
        for bounded_frame in _bounded_frame_chunks(
            declared,
            target_chunk_bytes=target_chunk_bytes,
        ):
            prepared_frame, bounded_logical_dtypes = accumulator.prepare_frame(
                bounded_frame
            )
            _require_writer_headroom(
                task_memory_budget_bytes,
                bounded_frame,
                prepared_frame,
            )
            for partition, partition_frame in partition_frames(
                prepared_frame,
                timestamp_column=timestamp_column,
                partition_columns=spec.partition_columns,
            ):
                for bounded_partition in _bounded_frame_chunks(
                    partition_frame,
                    target_chunk_bytes=target_chunk_bytes,
                ):
                    table = accumulator.table_for_frame(bounded_partition)
                    remaining_headroom = _require_writer_headroom(
                        task_memory_budget_bytes,
                        bounded_frame,
                        prepared_frame,
                        bounded_partition,
                        arrow_bytes=table.nbytes,
                    )
                    if remaining_headroom <= 0:
                        raise ValueError(
                            "dataset_working_set_exceeds_task_memory_budget"
                        )
                    for persisted_table in _bounded_persisted_tables(
                        table,
                        logical_dtypes=bounded_logical_dtypes,
                        target_chunk_bytes=target_chunk_bytes,
                    ):
                        part_index = part_indices.get(partition, 0)
                        part_indices[partition] = part_index + 1
                        output_path = partition_part_path(
                            root,
                            partition,
                            part_index,
                        )
                        write_parquet_file(persisted_table, output_path)
                        accumulator.observe_file(
                            output_path,
                            target_chunk_bytes=target_chunk_bytes,
                            operation_headroom_bytes=remaining_headroom,
                        )
                        _raise_accumulated_failures(accumulator)
                        del persisted_table, output_path
                    del table, remaining_headroom, bounded_partition
                del partition, partition_frame
            del prepared_frame, bounded_logical_dtypes, bounded_frame
        del declared, chunk
    if accumulator.row_count == 0 and not saw_empty_schema:
        raise ValueError("empty_dataset_schema_missing")
    if saw_empty_schema:
        logical_dtypes = accumulator.materialized_logical_dtypes(
            target_chunk_bytes=target_chunk_bytes
        )
        partition = "" if accumulator.row_count == 0 else min(part_indices)
        part_index = part_indices.get(partition, 0)
        part_indices[partition] = part_index + 1
        output_path = partition_part_path(root, partition, part_index)
        assert accumulator._arrow_schema is not None
        empty_table = pa.Table.from_batches([], schema=accumulator._arrow_schema)
        remaining_headroom = _require_writer_headroom(
            task_memory_budget_bytes,
            arrow_bytes=empty_table.nbytes,
        )
        if remaining_headroom <= 0:
            raise ValueError("dataset_working_set_exceeds_task_memory_budget")
        persisted_table = next(
            _bounded_persisted_tables(
                empty_table,
                logical_dtypes=logical_dtypes,
                target_chunk_bytes=target_chunk_bytes,
            )
        )
        write_parquet_file(persisted_table, output_path)
        accumulator.observe_file(
            output_path,
            target_chunk_bytes=target_chunk_bytes,
            operation_headroom_bytes=remaining_headroom,
        )
    return accumulator


def _require_writer_headroom(
    task_memory_budget_bytes: int,
    *frames: pd.DataFrame,
    arrow_bytes: int = 0,
) -> int:
    if (
        type(task_memory_budget_bytes) is not int
        or task_memory_budget_bytes <= 0
        or type(arrow_bytes) is not int
        or arrow_bytes < 0
    ):
        raise ValueError("invalid_dataset_task_memory_budget")
    retained_bytes = arrow_bytes + sum(
        pandas_frame_payload_bytes(frame)
        for frame in {id(item): item for item in frames}.values()
    )
    if retained_bytes > task_memory_budget_bytes:
        raise ValueError("dataset_working_set_exceeds_task_memory_budget")
    return task_memory_budget_bytes - retained_bytes


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
                    field.type.value_type,
                    ordered=field.type.ordered,
                )
                if pa.types.is_dictionary(field.type)
                else field.type,
                nullable=field.nullable,
            )
            for field in schema
        ]
    )


def _validate_persisted_logical_schema(
    schema: pa.Schema,
    logical_dtypes: Mapping[str, Any],
) -> None:
    for descriptor in logical_dtypes["columns"]:
        if descriptor["kind"] != "categorical":
            continue
        data_type = schema.field(descriptor["name"]).type
        if not pa.types.is_dictionary(data_type) or bool(data_type.ordered) != bool(
            descriptor["ordered"]
        ):
            raise ValueError("dataset_categorical_domain_mismatch")


def _validate_persisted_categorical_values(
    frame: pd.DataFrame,
    logical_dtypes: Mapping[str, Any],
) -> None:
    actual = logical_dtype_payload_from_frame(frame)
    actual_by_name = {item["name"]: item for item in actual["columns"]}
    for expected in logical_dtypes["columns"]:
        if expected["kind"] != "categorical":
            continue
        observed = actual_by_name.get(expected["name"])
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


def _bounded_persisted_tables(
    table: pa.Table,
    *,
    logical_dtypes: Mapping[str, Any],
    target_chunk_bytes: int,
) -> Iterator[pa.Table]:
    persisted = table.replace_schema_metadata(
        {b"qrp.logical-dtypes": logical_dtype_metadata_bytes(logical_dtypes)}
    )
    if persisted.num_rows == 0:
        yield persisted
        return
    pending = [persisted]
    while pending:
        candidate = pending.pop()
        if candidate.nbytes <= target_chunk_bytes:
            yield candidate
            continue
        if candidate.num_rows == 1:
            raise ValueError("dataset_row_exceeds_target_chunk_bytes")
        left_rows = candidate.num_rows // 2
        pending.append(candidate.slice(left_rows))
        pending.append(candidate.slice(0, left_rows))


def write_parquet_file(table: pa.Table, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path
