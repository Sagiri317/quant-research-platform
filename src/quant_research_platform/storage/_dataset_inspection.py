"""Dataset inspection implementation."""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Any

from quant_research_platform.contracts._json import render_qrp_json_object
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import (
    MemoryLease,
    RuntimeResourcePolicy,
    _OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_key_tracking import (
    EXACT_KEY_SQLITE_PAGE_BYTES,
    ExactPrimaryKeyIndex,
)
from quant_research_platform.storage._dataset_observation import (
    DatasetSummaryObserver,
)
from quant_research_platform.storage._dataset_paths import relative_partition_label
from quant_research_platform.storage._dataset_reading import (
    DatasetSession,
    dataset_partition_state_reserve_bytes,
    dataset_primary_key_schema,
    iter_dataset_file_batches,
    open_dataset_session,
)
from quant_research_platform.storage._dataset_schema import (
    date_string,
    physical_schema_columns,
)
from quant_research_platform.storage._dataset_types import DatasetInspection


def _close_lease_preserving_primary(lease: MemoryLease, *, owner: str) -> None:
    primary = sys.exception()
    try:
        lease.close()
    except BaseException as cleanup_error:
        if primary is None:
            raise
        primary.add_note(
            f"{owner} lease close also failed: {type(cleanup_error).__name__}"
        )


def _release_owned_value(
    owned: _OwnedValue[Any],
    *,
    owner: str,
) -> None:
    owned.value = None
    _close_lease_preserving_primary(owned.lease, owner=owner)


def inspect_dataset(
    dataset_root: Path,
    *,
    time_window: TimeWindow | None = None,
    resource_policy: RuntimeResourcePolicy | None,
) -> DatasetInspection:
    with open_dataset_session(
        dataset_root,
        columns=None,
        time_window=time_window,
        require_consumable=False,
        resource_policy=resource_policy,
    ) as context:
        return inspect_dataset_session(
            context,
            resource_policy=context.resource_policy,
        )


def inspect_dataset_session(
    context: DatasetSession,
    *,
    resource_policy: RuntimeResourcePolicy,
) -> DatasetInspection:
    """Inspect one already frozen Storage session without reopening its dataset."""

    if context.resource_policy is not resource_policy:
        raise RuntimeError("runtime_resource_policy_conflict")
    with runtime_resource_policy_scope(resource_policy):
        owned = _inspect_dataset_session_owned(
            context,
            resource_policy=resource_policy,
        )
        result = owned.value
        _release_owned_value(owned, owner="storage.dataset.inspection.result")
        return result


def _inspect_dataset_owned(
    dataset_root: Path,
    *,
    time_window: TimeWindow | None = None,
    resource_policy: RuntimeResourcePolicy,
) -> _OwnedValue[DatasetInspection]:
    with open_dataset_session(
        dataset_root,
        columns=None,
        time_window=time_window,
        require_consumable=False,
        resource_policy=resource_policy,
    ) as context:
        return _inspect_dataset_session_owned(
            context,
            resource_policy=resource_policy,
        )


def _inspect_dataset_session_owned(
    context: DatasetSession,
    *,
    resource_policy: RuntimeResourcePolicy,
) -> _OwnedValue[DatasetInspection]:
    try:
        result_lease = resource_policy.lease(
            "storage.dataset.inspection.result",
            _inspection_operation_bound(context),
        )
    except MemoryError:
        raise ValueError("dataset_working_set_exceeds_task_memory_budget") from None
    runtime: dict[str, Any] | None = None
    columns: tuple[str, ...] | list[str] = ()
    try:
        runtime = _inspection_runtime_metadata(
            context,
            target_chunk_bytes=resource_policy.available_memory_bytes,
        )
        columns = physical_schema_columns(context.schema)
    except BaseException:
        runtime = None
        columns = ()
        _close_lease_preserving_primary(
            result_lease,
            owner="storage.dataset.inspection.result",
        )
        raise
    try:
        result = DatasetInspection(
            root=context.root,
            data_root=context.data_root,
            spec=context.spec,
            columns=columns,
            schema_hash=context.schema_hash,
            file_count=int(runtime["file_count"]),
            row_count=int(runtime["row_count"]),
            partitions=runtime["partitions"],
            partition_row_counts=runtime["partition_row_counts"],
            date_min=runtime["date_min"],
            date_max=runtime["date_max"],
            manifest=context.manifest,
        )
        result_bytes = dataset_inspection_retained_memory_bytes(result)
        if result_bytes > result_lease.reserved_bytes:
            raise ValueError("dataset_inspection_result_bound_invalid")
        result_lease.shrink_to(result_bytes)
        return _OwnedValue(result, result_lease)
    except BaseException:
        result = None
        runtime = None
        columns = ()
        _close_lease_preserving_primary(
            result_lease,
            owner="storage.dataset.inspection.result",
        )
        raise


def _inspection_operation_bound(context: DatasetSession) -> int:
    path_count = len(context.all_paths)
    columns = getattr(context, "physical_columns", context.arrow_schema.names)
    result_bound = _inspection_result_bound(
        context,
        path_count=path_count,
        columns=columns,
    )
    workspace_bound = (
        dataset_partition_state_reserve_bytes(context)
        + sum(
            len(str(column).encode("utf-8")) + 32
            for column in columns
        )
        + 4096
    )
    return EXACT_KEY_SQLITE_PAGE_BYTES + workspace_bound + result_bound


def _inspection_result_bound(
    context: DatasetSession,
    *,
    path_count: int | None = None,
    columns: tuple[str, ...] | list[str] | None = None,
) -> int:
    pointer_bytes = struct.calcsize("P")
    if path_count is None:
        path_count = len(context.all_paths)
    if columns is None:
        columns = getattr(context, "physical_columns", context.arrow_schema.names)
    metadata_bytes = len(getattr(context, "spec_bytes", b"")) + len(
        getattr(context, "manifest_bytes", b"")
    )
    result_bound = metadata_bytes + sum(
        len(str(column).encode("utf-8")) + pointer_bytes for column in columns
    )
    return result_bound + path_count * (8 * pointer_bytes + 64) + 1024


def dataset_inspection_retained_memory_bytes(value: DatasetInspection) -> int:
    return _inspection_retained_memory_bytes(
        spec=value.spec,
        manifest=value.manifest,
        columns=value.columns,
        partitions=value.partitions,
    )


def _inspection_retained_memory_bytes(
    *,
    spec: dict[str, Any],
    manifest: dict[str, Any],
    columns: tuple[str, ...],
    partitions: tuple[str, ...],
) -> int:
    retained = len(render_qrp_json_object(spec).encode("utf-8"))
    retained += len(render_qrp_json_object(manifest).encode("utf-8"))
    retained += sum(len(item.encode("utf-8")) for item in columns)
    retained += sum(len(item.encode("utf-8")) for item in partitions)
    return retained


def inspection_from_persisted_session(
    context: DatasetSession,
    *,
    published_root: Path,
) -> DatasetInspection:
    """Build the writer receipt from the already validated persisted session."""

    root = Path(published_root)
    manifest = context.manifest
    return DatasetInspection(
        root=root,
        data_root=root / "data",
        spec=context.spec,
        columns=physical_schema_columns(context.schema),
        schema_hash=context.schema_hash,
        file_count=int(manifest["file_count"]),
        row_count=int(manifest["row_count"]),
        partitions=tuple(str(item) for item in manifest["partitions"]),
        partition_row_counts={
            str(key): int(value)
            for key, value in manifest["partition_row_counts"].items()
        },
        date_min=manifest["date_min"],
        date_max=manifest["date_max"],
        manifest=manifest,
    )


def _inspection_runtime_metadata(
    context: DatasetSession,
    *,
    target_chunk_bytes: int,
) -> dict[str, Any]:
    spec = context.spec
    primary_key = tuple(str(column) for column in spec["primary_key"])
    column_roles = {
        str(column): str(role) for column, role in spec["column_roles"].items()
    }
    remaining_headroom = (
        context.resource_policy.memory_budget_bytes
        - context.resource_policy.used_memory_bytes
    )
    if remaining_headroom <= 0:
        raise ValueError("dataset_working_set_exceeds_task_memory_budget")
    target_chunk_bytes = min(target_chunk_bytes, remaining_headroom)
    reader_upstream_headroom = remaining_headroom + context.retained_memory_bytes
    timestamp_column = context.timestamp_column
    if context.time_window is not None and timestamp_column is None:
        raise ValueError("time_window_not_supported")
    timestamp_columns = tuple(
        column
        for column in primary_key
        if spec["column_roles"].get(column) == "timestamp"
    )
    del spec
    partition_row_counts: dict[str, int] = {}
    selected_file_count = 0
    last_observed_path: Path | None = None
    with ExactPrimaryKeyIndex(
        primary_key=primary_key,
        timestamp_columns=timestamp_columns,
        scratch_parent=context.root.parent,
        key_schema=dataset_primary_key_schema(context, primary_key),
        sqlite_cache_bytes=EXACT_KEY_SQLITE_PAGE_BYTES,
    ) as key_index:
        observer = DatasetSummaryObserver(
            primary_key=primary_key,
            timestamp_column=timestamp_column,
            column_roles=column_roles,
            exact_key_index=key_index,
        )
        for path, frame in iter_dataset_file_batches(
            context,
            columns=tuple(context.arrow_schema.names),
            target_chunk_bytes=target_chunk_bytes,
            operation_headroom_bytes=reader_upstream_headroom,
            batch_observer=lambda batch, logical_frame: observer.observe_batch(
                batch,
                frame=logical_frame,
            ),
        ):
            if frame.empty:
                continue
            if path != last_observed_path:
                selected_file_count += 1
                last_observed_path = path
            partition = relative_partition_label(context.data_root, path)
            partition_row_counts[partition] = partition_row_counts.get(
                partition, 0
            ) + len(frame)
    if context.time_window is None:
        for path in context.all_paths:
            partition_row_counts.setdefault(
                relative_partition_label(context.data_root, path),
                0,
            )
        file_count = len(context.all_paths)
    else:
        file_count = selected_file_count
    partitions = tuple(sorted(partition_row_counts))
    runtime_summary = {
        "file_count": file_count,
        "row_count": int(observer.row_count),
        "partitions": partitions,
        "partition_row_counts": partition_row_counts,
        "date_min": date_string(observer.date_min),
        "date_max": date_string(observer.date_max),
    }
    if context.time_window is None:
        persisted_summary = {
            **runtime_summary,
            "partitions": list(partitions),
            **observer.summary_payload(),
        }
        for key, value in persisted_summary.items():
            if context.manifest.get(key) != value:
                raise ValueError("dataset_manifest_summary_mismatch")
    return runtime_summary
