"""Dataset inspection implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import DEFAULT_BATCH_ROWS
from quant_research_platform.storage.datasets.observation import DatasetSummaryObserver
from quant_research_platform.storage.datasets.paths import relative_partition_label
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    dataset_primary_key_schema,
    iter_dataset_file_batches,
    open_dataset_session,
)
from quant_research_platform.storage.datasets.schema import (
    date_string,
    physical_schema_columns,
)
from quant_research_platform.storage.datasets.types import DatasetInspection
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex


def inspect_dataset(
    dataset_root: Path, *, time_window: TimeWindow | None = None
) -> DatasetInspection:
    context = open_dataset_session(dataset_root, columns=None, time_window=time_window)
    return inspect_dataset_session(context)


def inspect_dataset_session(context: DatasetSession) -> DatasetInspection:
    """Inspect an existing frozen session through the shared summary observer."""
    runtime = _inspection_runtime_metadata(context, batch_rows=DEFAULT_BATCH_ROWS)
    return DatasetInspection(
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


def _inspection_runtime_metadata(
    context: DatasetSession,
    *,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    frame_observer=None,
) -> dict[str, Any]:
    spec = context.spec
    primary_key = tuple(str(column) for column in spec["primary_key"])
    column_roles = {
        str(column): str(role) for column, role in spec["column_roles"].items()
    }
    timestamp_column = context.timestamp_column
    if context.time_window is not None and timestamp_column is None:
        raise ValueError("time_window_not_supported")
    del spec
    partition_row_counts: dict[str, int] = {}
    selected_file_count = 0
    last_observed_path: Path | None = None
    key_index = ExactPrimaryKeyIndex(
        primary_key=primary_key,
        key_schema=dataset_primary_key_schema(context, primary_key),
    )
    observer = DatasetSummaryObserver(
        primary_key=primary_key,
        timestamp_column=timestamp_column,
        column_roles=column_roles,
        exact_key_index=key_index,
    )
    with closing_iterator(
        iter_dataset_file_batches(
            context,
            columns=tuple(context.arrow_schema.names),
            batch_rows=batch_rows,
            batch_observer=lambda batch, logical_frame: observer.observe_batch(
                batch, frame=logical_frame
            ),
        )
    ) as batches:
        for path, frame in batches:
            if frame.empty:
                continue
            if path != last_observed_path:
                selected_file_count += 1
                last_observed_path = path
            partition = relative_partition_label(context.data_root, path)
            partition_row_counts[partition] = partition_row_counts.get(
                partition, 0
            ) + len(frame)
            if frame_observer is not None:
                frame_observer(frame)
    if context.time_window is None:
        for path in context.all_paths:
            partition_row_counts.setdefault(
                relative_partition_label(context.data_root, path), 0
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
        manifest = context.manifest
        for key, value in persisted_summary.items():
            if manifest.get(key) != value:
                raise ValueError("dataset_manifest_summary_mismatch")
    return runtime_summary
