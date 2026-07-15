"""Dataset inspection implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage._dataset_key_tracking import (
    ExactPrimaryKeyIndex,
)
from quant_research_platform.storage._dataset_observation import (
    DatasetSummaryObserver,
)
from quant_research_platform.storage._dataset_paths import relative_partition_label
from quant_research_platform.storage._dataset_reading import (
    DatasetSession,
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
    resource_policy: RuntimeResourcePolicy,
) -> DatasetInspection:
    context = open_dataset_session(
        dataset_root,
        columns=None,
        time_window=time_window,
        require_consumable=False,
    )
    return inspect_dataset_session(context, resource_policy=resource_policy)


def inspect_dataset_session(
    context: DatasetSession,
    *,
    resource_policy: RuntimeResourcePolicy,
) -> DatasetInspection:
    """Inspect one already frozen Storage session without reopening its dataset."""

    runtime = _inspection_runtime_metadata(
        context,
        target_chunk_bytes=resource_policy.payload_target_bytes,
    )
    return DatasetInspection(
        root=context.root,
        data_root=context.data_root,
        spec=context.spec,
        columns=physical_schema_columns(context.schema),
        schema_hash=context.schema_hash,
        file_count=int(runtime["file_count"]),
        row_count=int(runtime["row_count"]),
        partitions=tuple(runtime["partitions"]),
        partition_row_counts=dict(runtime["partition_row_counts"]),
        date_min=runtime["date_min"],
        date_max=runtime["date_max"],
        manifest=context.manifest,
    )


def _inspection_from_persisted_session(
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
    timestamp_column = context.timestamp_column
    if context.time_window is not None and timestamp_column is None:
        raise ValueError("time_window_not_supported")
    primary_key = tuple(str(column) for column in context.spec["primary_key"])
    timestamp_columns = tuple(
        column
        for column in primary_key
        if context.spec["column_roles"].get(column) == "timestamp"
    )
    partition_row_counts: dict[str, int] = {}
    observed_paths: set[Path] = set()
    with ExactPrimaryKeyIndex(
        primary_key=primary_key,
        timestamp_columns=timestamp_columns,
        scratch_parent=context.root.parent,
        key_schema=dataset_primary_key_schema(context, primary_key),
    ) as key_index:
        observer = DatasetSummaryObserver(
            primary_key=primary_key,
            timestamp_column=timestamp_column,
            column_roles={
                str(column): str(role)
                for column, role in context.spec["column_roles"].items()
            },
            exact_key_index=key_index,
        )
        for path, frame in iter_dataset_file_batches(
            context,
            columns=tuple(context.arrow_schema.names),
            target_chunk_bytes=target_chunk_bytes,
        ):
            if frame.empty:
                continue
            observed_paths.add(path)
            partition = relative_partition_label(context.data_root, path)
            partition_row_counts[partition] = (
                partition_row_counts.get(partition, 0) + len(frame)
            )
            observer.observe_frame(frame)
    if context.time_window is None:
        for path in context.all_paths:
            partition_row_counts.setdefault(
                relative_partition_label(context.data_root, path),
                0,
            )
        file_count = len(context.all_paths)
    else:
        file_count = len(observed_paths)
    runtime_summary = {
        "file_count": file_count,
        "row_count": int(observer.row_count),
        "partitions": sorted(partition_row_counts),
        "partition_row_counts": dict(sorted(partition_row_counts.items())),
        "date_min": date_string(observer.date_min),
        "date_max": date_string(observer.date_max),
    }
    if context.time_window is None:
        persisted_summary = {
            **runtime_summary,
            **observer.summary_payload(),
        }
        for key, value in persisted_summary.items():
            if context.manifest.get(key) != value:
                raise ValueError("dataset_manifest_summary_mismatch")
    return runtime_summary
