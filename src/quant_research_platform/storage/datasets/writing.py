"""Dataset writing implementation."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, nullcontext
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
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    RuntimeResourcePolicy,
    _close_lease_preserving_primary,
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
    require_free_disk_headroom,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.atomic import atomic_write_directory
from quant_research_platform.storage.datasets.inspection import (
    _inspection_result_bound,
    dataset_inspection_retained_memory_bytes,
    inspection_from_persisted_session,
)
from quant_research_platform.storage.datasets.inventory import (
    DATASET_INVENTORY_RESERVE_BYTES,
    DatasetSessionInventory,
)
from quant_research_platform.storage.datasets.observation import (
    DatasetSummaryObserver,
)
from quant_research_platform.storage.datasets.paths import (
    partition_frames,
    partition_part_path,
    relative_partition_label,
    timestamp_primary_key_column_from_spec,
    validate_dataset_artifact_root,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    dataset_session_from_inventory,
    iter_bounded_frame_chunks,
    iter_declared_parquet_member_batches,
)
from quant_research_platform.storage.datasets.schema import (
    data_spec_payload,
    encode_logical_category_value,
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
    validate_persisted_logical_schema,
)
from quant_research_platform.storage.datasets.types import DatasetInspection
from quant_research_platform.storage.errors import OutputAlreadyExistsError
from quant_research_platform.storage.exact_keys import (
    EXACT_KEY_SQLITE_CACHE_BYTES,
    ExactPrimaryKeyIndex,
)
from quant_research_platform.storage.identity import (
    dataset_file_identity,
    native_filesystem_path,
    require_artifact_root,
)
from quant_research_platform.storage.manifests import write_internal_artifact_manifest
from quant_research_platform.storage.parquet import (
    _PARQUET_NATIVE_FIXED_WORKSPACE_BYTES,
    _PHYSICAL_PAGE_ROWS_KEY,
    DatasetPartMetadata,
    _parquet_row_group_footer_upper_bound,
    _parquet_row_group_size,
    _table_parquet_page_working_set_bytes,
    table_with_decoded_row_bounds,
)

_TIMESTAMP_PREPARATION_BYTES_PER_ROW = 256
_ARROW_CELL_OVERHEAD_BYTES = 9
_ARROW_FRAME_FIXED_BYTES = 4 * 1024
_DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES = (
    EXACT_KEY_SQLITE_CACHE_BYTES + DATASET_INVENTORY_RESERVE_BYTES
)


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
                target_chunk_bytes=operation_headroom_bytes,
            )
            dtype_overrides.update(
                pandas_dtype_overrides_from_logical_payload(
                    logical_payload,
                    columns=tuple(metadata.schema.names),
                )
            )
            physical_schema = physical_schema_from_schema(metadata.schema)
            validate_persisted_logical_schema(metadata.schema, logical_payload)
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
                    encode_logical_category_value(
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
    data: pd.DataFrame | Iterable[object],
    *,
    spec: DataSpec,
    instance_metadata: Mapping[str, Any] | None = None,
    staged_validator: Callable[[DatasetWriteReceipt], None] | None = None,
    resource_policy: RuntimeResourcePolicy | None,
    borrowed_input: bool = False,
) -> DatasetInspection:
    with runtime_resource_policy_scope(resource_policy):
        owned = _write_dataset_owned(
            dataset_root,
            data,
            spec=spec,
            instance_metadata=instance_metadata,
            staged_validator=staged_validator,
            borrowed_input=borrowed_input,
        )
        result = owned.value
        owned.release(owner="storage.dataset.write.result")
        return result


def _write_dataset_owned(
    dataset_root: Path,
    data: pd.DataFrame | Iterable[object],
    *,
    spec: DataSpec,
    instance_metadata: Mapping[str, Any] | None = None,
    staged_validator: Callable[[DatasetWriteReceipt], None] | None = None,
    borrowed_input: bool = False,
) -> OwnedValue[DatasetInspection]:
    resource_policy = current_runtime_resource_policy()
    if resource_policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    final_root = Path(dataset_root)
    validate_dataset_spec(spec)
    validate_dataset_artifact_root(final_root, spec)
    final_root = require_artifact_root(final_root)
    owner_reserve_bytes = _DATASET_WRITE_FIXED_OWNER_RESERVE_BYTES
    with resource_policy.lease("storage.dataset.write", owner_reserve_bytes):
        return _write_dataset_body(
            final_root,
            data,
            spec=spec,
            instance_metadata=instance_metadata,
            staged_validator=staged_validator,
            borrowed_input=borrowed_input,
        )


def _write_dataset_body(
    final_root: Path,
    data: pd.DataFrame | Iterable[object],
    *,
    spec: DataSpec,
    instance_metadata: Mapping[str, Any] | None,
    staged_validator: Callable[[DatasetWriteReceipt], None] | None,
    borrowed_input: bool,
) -> OwnedValue[DatasetInspection]:
    resource_policy = current_runtime_resource_policy()
    if resource_policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    input_chunks = iter((data,)) if isinstance(data, pd.DataFrame) else iter(data)
    close = getattr(input_chunks, "close", None)
    input_closed = close is None
    input_cleanup_failed = False
    lease_cleanup_failed = False
    active_producer_leases: list[MemoryLease] = []
    active_producer_values: list[OwnedValue[pd.DataFrame]] = []
    pending_result: OwnedValue[DatasetInspection] | None = None

    def close_input_once() -> None:
        nonlocal input_cleanup_failed, input_closed
        if input_closed or input_cleanup_failed:
            return
        try:
            close()
        except BaseException:
            input_cleanup_failed = True
            raise
        input_closed = True

    def close_input_and_active_leases() -> None:
        nonlocal lease_cleanup_failed
        primary = sys.exception()
        first_failure: BaseException | None = None
        try:
            close_input_once()
        except BaseException as cleanup_error:
            if primary is None:
                first_failure = cleanup_error
            else:
                primary.add_note(
                    "dataset producer close also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        active_producer_values.clear()
        if lease_cleanup_failed:
            return
        while active_producer_leases:
            lease = active_producer_leases.pop()
            try:
                lease.close()
            except BaseException as cleanup_error:
                lease_cleanup_failed = True
                if primary is None:
                    if first_failure is None:
                        first_failure = cleanup_error
                    else:
                        first_failure.add_note(
                            "dataset producer lease close also failed: "
                            f"{type(cleanup_error).__name__}"
                        )
                else:
                    primary.add_note(
                        "dataset producer lease close also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
        if first_failure is not None:
            raise first_failure

    def _writer(temp_root: Path) -> OwnedValue[DatasetInspection]:
        nonlocal pending_result
        try:
            if resource_policy.operation_headroom_bytes <= 0:
                raise ValueError("dataset_working_set_exceeds_task_memory_budget")
            timestamp_columns = tuple(
                column
                for column in spec.primary_key
                if spec.column_roles.get(column) is ColumnRole.TIMESTAMP
            )
            with (
                DatasetSessionInventory(
                    data_root=temp_root / "data",
                    resource_policy=resource_policy,
                ) as inventory,
                ExactPrimaryKeyIndex(
                    primary_key=spec.primary_key,
                    timestamp_columns=timestamp_columns,
                ) as key_tracker,
            ):
                accumulator = _write_contents(
                    input_chunks,
                    temp_root / "data",
                    spec=spec,
                    target_chunk_bytes=resource_policy.operation_headroom_bytes,
                    key_tracker=key_tracker,
                    inventory=inventory,
                    borrowed_input=borrowed_input,
                    active_producer_leases=active_producer_leases,
                    active_producer_values=active_producer_values,
                )
                write_internal_artifact_manifest(
                    temp_root / "spec.json",
                    data_spec_payload(spec),
                )
                manifest = merge_dataset_instance_metadata(
                    {
                        **accumulator.payload(
                            target_chunk_bytes=(
                                resource_policy.operation_headroom_bytes
                            )
                        ),
                        "schema_id": "qrp-dataset-manifest",
                    },
                    instance_metadata,
                    spec=data_spec_payload(spec),
                )
                write_internal_artifact_manifest(
                    temp_root / "_manifest" / "manifest.json",
                    manifest,
                )
                with dataset_session_from_inventory(
                    temp_root,
                    inventory=inventory,
                    columns=None,
                    time_window=None,
                    resource_policy=resource_policy,
                ) as session:
                    if staged_validator is not None:
                        staged_validator(
                            DatasetWriteReceipt(
                                root=temp_root,
                                session=session,
                                exact_primary_key_index=key_tracker,
                            )
                        )
                    session.verify()
                    result_bound = _inspection_result_bound(session)
                    try:
                        result_lease = resource_policy.lease(
                            "storage.dataset.write.result",
                            result_bound,
                        )
                    except MemoryError:
                        raise ValueError(
                            "dataset_working_set_exceeds_task_memory_budget"
                        ) from None
                    try:
                        result = inspection_from_persisted_session(
                            session,
                            published_root=final_root,
                        )
                        result_bytes = dataset_inspection_retained_memory_bytes(result)
                        if result_bytes > result_lease.reserved_bytes:
                            raise ValueError("dataset_write_result_bound_invalid")
                        result_lease.shrink_to(result_bytes)
                        pending_result = OwnedValue(result, result_lease)
                    except BaseException:
                        result = None
                        _close_lease_preserving_primary(
                            result_lease,
                            owner="storage.dataset.write.result",
                        )
                        raise
                return pending_result
        finally:
            close_input_and_active_leases()

    try:
        try:
            published = atomic_write_directory(final_root, _writer)
            pending_result = None
            return published
        except OutputAlreadyExistsError as exc:
            raise ValueError("dataset_exists") from exc
    finally:
        result_to_release = pending_result
        pending_result = None
        try:
            if result_to_release is not None:
                result_to_release.release(owner="storage.dataset.write.result")
        finally:
            close_input_and_active_leases()


def _write_contents(
    chunks: Iterator[Any],
    root: Path,
    *,
    spec: DataSpec,
    target_chunk_bytes: int,
    key_tracker: ExactPrimaryKeyIndex,
    inventory: DatasetSessionInventory,
    borrowed_input: bool,
    active_producer_leases: list[MemoryLease],
    active_producer_values: list[OwnedValue[pd.DataFrame]],
) -> _WriteStats:
    resource_policy = current_runtime_resource_policy()
    if resource_policy is None:
        raise RuntimeError("runtime_resource_policy_not_active")
    timestamp_column = validate_dataset_spec(spec)
    timestamp_key_count = sum(
        spec.column_roles.get(column) is ColumnRole.TIMESTAMP
        for column in spec.primary_key
    )
    accumulator = _WriteStats(
        spec=spec,
        data_root=root,
        key_tracker=key_tracker,
        inventory=inventory,
    )
    part_indices: dict[str, int] = {}
    saw_empty_schema = False
    producer_bound: int | tuple[int, MemoryLease] | None = None
    chunk: pd.DataFrame | None = None
    declared: pd.DataFrame | None = None
    frame: pd.DataFrame | None = None
    bounded_frame: pd.DataFrame | None = None
    prepared_frame: pd.DataFrame | None = None
    partition_frame: pd.DataFrame | None = None
    bounded_partition: pd.DataFrame | None = None
    table: pa.Table | None = None
    persisted_table: pa.Table | None = None

    def clear_current_chunk() -> None:
        nonlocal bounded_frame, bounded_partition, chunk, declared, frame
        nonlocal partition_frame, persisted_table, prepared_frame, table
        chunk = None
        declared = None
        frame = None
        bounded_frame = None
        prepared_frame = None
        partition_frame = None
        bounded_partition = None
        table = None
        persisted_table = None

    while True:
        producer_lease: MemoryLease | None = None
        if borrowed_input:
            try:
                chunk = next(chunks)
            except StopIteration:
                break
        else:
            if producer_bound is None:
                try:
                    producer_bound = next(chunks)
                except StopIteration:
                    break
            if isinstance(producer_bound, tuple):
                if len(producer_bound) != 2:
                    raise ValueError("invalid_dataset_producer_handoff")
                handoff_bound, handoff_lease = producer_bound
                if (
                    isinstance(handoff_bound, bool)
                    or type(handoff_bound) is not int
                    or handoff_bound <= 0
                    or not isinstance(handoff_lease, MemoryLease)
                ):
                    raise ValueError("invalid_dataset_producer_handoff")
                if (
                    handoff_lease._root is not resource_policy
                    or not handoff_lease.active
                ):
                    raise ValueError("invalid_dataset_producer_handoff")
                if handoff_lease.reserved_bytes < handoff_bound:
                    handoff_lease.grow(handoff_bound - handoff_lease.reserved_bytes)
                handoff_lease.transfer("storage.dataset.producer")
                producer_lease = handoff_lease
            else:
                if (
                    isinstance(producer_bound, bool)
                    or type(producer_bound) is not int
                    or producer_bound <= 0
                ):
                    raise ValueError("invalid_dataset_producer_bound")
                producer_lease = resource_policy.lease(
                    "storage.dataset.producer",
                    producer_bound,
                )
            active_producer_leases.append(producer_lease)
            send = getattr(chunks, "send", None)
            if not callable(send):
                raise TypeError("invalid_dataset_producer")
            try:
                owned = send(producer_lease)
            except StopIteration:
                producer_lease.close()
                active_producer_leases.pop()
                producer_lease = None
            if producer_lease is None:
                break
            if not isinstance(owned, OwnedValue):
                raise TypeError("invalid_dataset_owned_frame")
            if owned.lease is not producer_lease:
                raise RuntimeError("runtime_memory_lease_transfer_mismatch")
            if type(owned.value) is not pd.DataFrame:
                raise TypeError("invalid_dataset_chunk")
            producer_lease.transfer("storage.dataset.input")
            chunk = owned.value
            active_producer_values.append(owned)
            del owned
            producer_bound = None
        if type(chunk) is not pd.DataFrame:
            raise TypeError("invalid_dataset_chunk")
        chunk_bytes = pandas_frame_payload_bytes(chunk)
        if producer_lease is not None and chunk_bytes > producer_lease.reserved_bytes:
            raise ValueError("dataset_producer_bound_invalid")
        with ExitStack() as publication_state:
            publication_state.callback(clear_current_chunk)
            metadata_budget = resource_policy.operation_headroom_bytes
            if metadata_budget <= 0:
                raise ValueError("dataset_working_set_exceeds_task_memory_budget")
            operator_target_bytes = _dataset_write_operator_target_bytes(
                chunk,
                operation_headroom_bytes=metadata_budget,
                timestamp_key_count=timestamp_key_count,
                partitioned=bool(spec.partition_columns),
            )
            declared = accumulator.declared_frame(
                chunk,
                target_chunk_bytes=metadata_budget,
            )
            if declared.empty:
                prepared_bound = (
                    chunk_bytes
                    + len(declared)
                    * timestamp_key_count
                    * _TIMESTAMP_PREPARATION_BYTES_PER_ROW
                )
                with resource_policy.lease(
                    "storage.dataset.prepare",
                    prepared_bound,
                ):
                    frame, logical_dtypes = accumulator.prepare_frame(declared)
                    frame_bytes = pandas_frame_payload_bytes(frame)
                    arrow_bound = (
                        _ARROW_FRAME_FIXED_BYTES
                        + frame_bytes
                        + len(frame) * len(frame.columns) * _ARROW_CELL_OVERHEAD_BYTES
                    )
                    arrow_cast_bound = arrow_bound
                    with resource_policy.lease(
                        "storage.dataset.arrow",
                        arrow_bound + arrow_cast_bound,
                    ):
                        table = accumulator.table_for_frame(frame)
                        del table
                    metadata_budget = resource_policy.operation_headroom_bytes
                    accumulator.observe_logical_dtype_payload(
                        logical_dtypes,
                        target_chunk_bytes=metadata_budget,
                    )
                saw_empty_schema = True
                del frame, logical_dtypes, declared, chunk
                if producer_lease is not None:
                    active_producer_values.pop()
                    try:
                        producer_bound = send(None)
                    except StopIteration:
                        producer_bound = None
                        producer_finished = True
                    else:
                        producer_finished = False
                    producer_lease.close()
                    active_producer_leases.pop()
                    producer_lease = None
                    if producer_finished:
                        break
                continue
            for bounded_frame in iter_bounded_frame_chunks(
                declared,
                target_chunk_bytes=operator_target_bytes,
            ):
                bounded_frame_bytes = pandas_frame_payload_bytes(bounded_frame)
                prepared_bound = (
                    bounded_frame_bytes
                    + len(bounded_frame)
                    * timestamp_key_count
                    * _TIMESTAMP_PREPARATION_BYTES_PER_ROW
                )
                with resource_policy.lease(
                    "storage.dataset.prepare",
                    prepared_bound,
                ):
                    prepared_frame, bounded_logical_dtypes = accumulator.prepare_frame(
                        bounded_frame
                    )
                    prepared_frame_bytes = pandas_frame_payload_bytes(prepared_frame)
                    if prepared_frame_bytes > prepared_bound:
                        raise ValueError(
                            "dataset_working_set_exceeds_task_memory_budget"
                        )
                    partition_mask_bytes = (
                        len(prepared_frame) if spec.partition_columns else 0
                    )
                    lower_bound_mask_bytes = partition_mask_bytes
                    upper_bound_mask_bytes = partition_mask_bytes
                    combined_mask_bytes = partition_mask_bytes
                    partition_output_bound = (
                        prepared_frame_bytes if spec.partition_columns else 0
                    )
                    partition_bound = sum(
                        (
                            lower_bound_mask_bytes,
                            upper_bound_mask_bytes,
                            combined_mask_bytes,
                            partition_output_bound,
                        )
                    )
                    with resource_policy.lease(
                        "storage.dataset.partition",
                        partition_bound,
                    ):
                        for partition, partition_frame in partition_frames(
                            prepared_frame,
                            timestamp_column=timestamp_column,
                            partition_columns=spec.partition_columns,
                        ):
                            for bounded_partition in iter_bounded_frame_chunks(
                                partition_frame,
                                target_chunk_bytes=operator_target_bytes,
                            ):
                                bounded_partition_bytes = pandas_frame_payload_bytes(
                                    bounded_partition
                                )
                                arrow_bound = (
                                    _ARROW_FRAME_FIXED_BYTES
                                    + bounded_partition_bytes
                                    + len(bounded_partition)
                                    * len(bounded_partition.columns)
                                    * _ARROW_CELL_OVERHEAD_BYTES
                                )
                                arrow_cast_bound = arrow_bound
                                written_paths: list[Path] = []
                                with resource_policy.lease(
                                    "storage.dataset.arrow",
                                    arrow_bound + arrow_cast_bound,
                                ):
                                    table = accumulator.table_for_frame(
                                        bounded_partition
                                    )
                                    if table.nbytes > arrow_bound:
                                        raise ValueError(
                                            "dataset_working_set_exceeds_task_memory_budget"
                                        )
                                    for persisted_table in _bounded_persisted_tables(
                                        table,
                                        logical_dtypes=bounded_logical_dtypes,
                                        target_chunk_bytes=operator_target_bytes,
                                    ):
                                        part_index = part_indices.get(partition, 0)
                                        part_indices[partition] = part_index + 1
                                        output_path = partition_part_path(
                                            root,
                                            partition,
                                            part_index,
                                        )
                                        write_parquet_file(
                                            persisted_table,
                                            output_path,
                                        )
                                        written_paths.append(output_path)
                                        del persisted_table, output_path
                                    del table, bounded_partition
                                for output_path in written_paths:
                                    readback_headroom = (
                                        resource_policy.operation_headroom_bytes
                                    )
                                    if readback_headroom <= 0:
                                        raise ValueError(
                                            "dataset_working_set_exceeds_task_memory_budget"
                                        )
                                    accumulator.observe_file(
                                        output_path,
                                        target_chunk_bytes=operator_target_bytes,
                                        operation_headroom_bytes=readback_headroom,
                                    )
                                    _raise_accumulated_failures(accumulator)
                                del written_paths
                            del partition, partition_frame
                    del prepared_frame, bounded_logical_dtypes, bounded_frame
            del declared, chunk
        if producer_lease is not None:
            active_producer_values.pop()
            try:
                producer_bound = send(None)
            except StopIteration:
                producer_bound = None
                producer_finished = True
            else:
                producer_finished = False
            producer_lease.close()
            active_producer_leases.pop()
            producer_lease = None
            if producer_finished:
                break
    if accumulator.row_count == 0 and not saw_empty_schema:
        raise ValueError("empty_dataset_schema_missing")
    if saw_empty_schema:
        metadata_budget = resource_policy.operation_headroom_bytes
        logical_dtypes = accumulator.materialized_logical_dtypes(
            target_chunk_bytes=metadata_budget
        )
        partition = "" if accumulator.row_count == 0 else min(part_indices)
        part_index = part_indices.get(partition, 0)
        part_indices[partition] = part_index + 1
        output_path = partition_part_path(root, partition, part_index)
        assert accumulator._arrow_schema is not None
        empty_table_bound = _ARROW_FRAME_FIXED_BYTES
        empty_cast_bound = _ARROW_FRAME_FIXED_BYTES
        with resource_policy.lease(
            "storage.dataset.arrow",
            empty_table_bound + empty_cast_bound,
        ):
            empty_table = pa.Table.from_batches([], schema=accumulator._arrow_schema)
            persisted_table = next(
                _bounded_persisted_tables(
                    empty_table,
                    logical_dtypes=logical_dtypes,
                    target_chunk_bytes=target_chunk_bytes,
                )
            )
            write_parquet_file(persisted_table, output_path)
            readback_headroom = resource_policy.operation_headroom_bytes
            accumulator.observe_file(
                output_path,
                target_chunk_bytes=target_chunk_bytes,
                operation_headroom_bytes=readback_headroom,
            )
    return accumulator


def _raise_accumulated_failures(accumulator: _WriteStats) -> None:
    summary = accumulator._summary_observer
    if summary.duplicate_primary_key_count:
        raise ValueError("duplicate_dataset_primary_key")
    if any(int(count) > 0 for count in summary.null_count_summary.values()):
        raise ValueError("dataset_primary_key_null")
    if any(int(count) > 0 for count in summary.non_finite_count_summary.values()):
        raise ValueError("dataset_infinite_value")


def _dataset_write_operator_target_bytes(
    frame: pd.DataFrame,
    *,
    operation_headroom_bytes: int,
    timestamp_key_count: int,
    partitioned: bool,
) -> int:
    if frame.empty:
        return operation_headroom_bytes
    fixed_bytes = 2 * _ARROW_FRAME_FIXED_BYTES + _PARQUET_NATIVE_FIXED_WORKSPACE_BYTES
    available = operation_headroom_bytes - fixed_bytes
    if available <= 0:
        raise ValueError("dataset_working_set_exceeds_task_memory_budget")
    payload_per_row = max(
        1,
        (pandas_frame_payload_bytes(frame) + len(frame) - 1) // len(frame),
    )
    payload_overlaps = 5 if partitioned else 4
    row_overlap_bytes = (
        timestamp_key_count * _TIMESTAMP_PREPARATION_BYTES_PER_ROW
        + (3 if partitioned else 0)
        + 2 * len(frame.columns) * _ARROW_CELL_OVERHEAD_BYTES
    )
    rows = available // (payload_overlaps * payload_per_row + row_overlap_bytes)
    if rows <= 0:
        raise ValueError("dataset_working_set_exceeds_task_memory_budget")
    return rows * payload_per_row


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
            yield candidate
            continue
        left_rows = candidate.num_rows // 2
        pending.append(candidate.slice(left_rows))
        pending.append(candidate.slice(0, left_rows))


def write_parquet_file(table: pa.Table, path: Path) -> Path:
    Path(native_filesystem_path(path.parent)).mkdir(parents=True, exist_ok=True)
    persisted = table_with_decoded_row_bounds(table)
    metadata = dict(persisted.schema.metadata or {})
    metadata[_PHYSICAL_PAGE_ROWS_KEY] = str(max(1, persisted.num_rows)).encode("ascii")
    persisted = persisted.replace_schema_metadata(metadata)
    policy = current_runtime_resource_policy()
    if policy is None:
        batch_rows = max(1, persisted.num_rows)
    else:
        remaining = policy.operation_headroom_bytes
        footer_bytes = _parquet_row_group_footer_upper_bound(
            persisted.schema,
            rows=persisted.num_rows,
        )
        page_headroom = remaining - footer_bytes
        minimum_rows = 0 if persisted.num_rows == 0 else 1
        minimum_bytes = _table_parquet_page_working_set_bytes(
            persisted,
            rows=minimum_rows,
        )
        if minimum_bytes > page_headroom:
            raise ValueError("dataset_working_set_exceeds_task_memory_budget")
        batch_rows = 1
        page_working_bytes = minimum_bytes
        if persisted.num_rows:
            low = 1
            high = _parquet_row_group_size(persisted.num_rows)
            while low <= high:
                candidate = (low + high) // 2
                candidate_bytes = _table_parquet_page_working_set_bytes(
                    persisted,
                    rows=candidate,
                )
                if candidate_bytes <= page_headroom:
                    batch_rows = candidate
                    page_working_bytes = candidate_bytes
                    low = candidate + 1
                else:
                    high = candidate - 1
    metadata = dict(persisted.schema.metadata or {})
    metadata[_PHYSICAL_PAGE_ROWS_KEY] = str(batch_rows).encode("ascii")
    persisted = persisted.replace_schema_metadata(metadata)
    require_free_disk_headroom(
        path.parent,
        reserve_bytes=(
            _table_parquet_page_working_set_bytes(
                persisted,
                rows=persisted.num_rows,
            )
            + _parquet_row_group_footer_upper_bound(
                persisted.schema,
                rows=persisted.num_rows,
            )
        ),
    )
    if policy is None:
        reservation = nullcontext()
    else:
        reservation = policy.lease(
            "storage.parquet.native",
            page_working_bytes + footer_bytes,
        )
    with reservation:
        pq.write_table(
            persisted,
            native_filesystem_path(path),
            compression="zstd",
            row_group_size=_parquet_row_group_size(persisted.num_rows),
            write_batch_size=batch_rows,
            max_rows_per_page=batch_rows,
            use_dictionary=False,
            write_statistics=False,
            write_page_index=False,
            write_page_checksum=False,
        )
    return path
