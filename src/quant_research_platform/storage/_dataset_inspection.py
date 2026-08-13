"""Dataset inspection implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_research_platform.contracts._json import render_qrp_json_object
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
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


def inspect_dataset(
    dataset_root: Path,
    *,
    time_window: TimeWindow | None = None,
    resource_policy: RuntimeResourcePolicy | None,
    retained_payload_bytes: int = 0,
) -> DatasetInspection:
    with open_dataset_session(
        dataset_root,
        columns=None,
        time_window=time_window,
        require_consumable=False,
        resource_policy=resource_policy,
        retained_payload_bytes=retained_payload_bytes,
    ) as context:
        return inspect_dataset_session(
            context,
            resource_policy=context.resource_policy,
            retained_payload_bytes=retained_payload_bytes,
        )


def inspect_dataset_session(
    context: DatasetSession,
    *,
    resource_policy: RuntimeResourcePolicy,
    retained_payload_bytes: int = 0,
) -> DatasetInspection:
    """Inspect one already frozen Storage session without reopening its dataset."""

    if context.resource_policy is not resource_policy:
        raise RuntimeError("runtime_resource_policy_conflict")
    with runtime_resource_policy_scope(resource_policy):
        runtime = _inspection_runtime_metadata(
            context,
            target_chunk_bytes=resource_policy.remaining_memory_bytes,
            retained_payload_bytes=retained_payload_bytes,
        )
        result = DatasetInspection(
            root=context.root,
            data_root=context.data_root,
            spec=context.spec,
            columns=physical_schema_columns(context.schema),
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
        if (
            retained_payload_bytes + context.retained_memory_bytes + result_bytes
            > resource_policy.task_memory_budget_bytes
        ):
            raise ValueError("dataset_working_set_exceeds_task_memory_budget")
        return result


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
    retained_payload_bytes: int,
) -> dict[str, Any]:
    if type(retained_payload_bytes) is not int or retained_payload_bytes < 0:
        raise ValueError("invalid_dataset_inspection_retained_payload_bytes")
    spec = context.spec
    primary_key = tuple(str(column) for column in spec["primary_key"])
    column_roles = {
        str(column): str(role) for column, role in spec["column_roles"].items()
    }
    observer_state_bytes = sum(
        len(column.encode("utf-8")) + len(role.encode("utf-8")) + 24
        for column, role in column_roles.items()
    )
    accumulated_state_bytes = (
        dataset_partition_state_reserve_bytes(context) + observer_state_bytes
    )
    reader_upstream_headroom = context.resource_policy.task_memory_budget_bytes
    reader_upstream_headroom -= retained_payload_bytes
    reader_upstream_headroom -= EXACT_KEY_SQLITE_PAGE_BYTES
    reader_upstream_headroom -= accumulated_state_bytes
    remaining_headroom = reader_upstream_headroom - context.retained_memory_bytes
    if remaining_headroom <= 0:
        raise ValueError("dataset_working_set_exceeds_task_memory_budget")
    target_chunk_bytes = min(target_chunk_bytes, remaining_headroom)
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
    projected_result_bytes = _inspection_retained_memory_bytes(
        spec=context.spec,
        manifest=context.manifest,
        columns=physical_schema_columns(context.schema),
        partitions=partitions,
    )
    if (
        retained_payload_bytes
        + context.retained_memory_bytes
        + accumulated_state_bytes
        + projected_result_bytes
        > context.resource_policy.task_memory_budget_bytes
    ):
        raise ValueError("dataset_working_set_exceeds_task_memory_budget")
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
