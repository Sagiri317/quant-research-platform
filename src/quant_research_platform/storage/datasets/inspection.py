"""Inspect selected dataset structure and row coverage through the native reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.datasets.paths import relative_partition_label
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_file_batches,
    open_dataset_session,
)
from quant_research_platform.storage.datasets.schema import (
    date_string,
    physical_schema_columns,
)
from quant_research_platform.storage.datasets.types import DatasetInspection
from quant_research_platform.storage.parquet import DEFAULT_BATCH_ROWS


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
) -> dict[str, Any]:
    if context.native and context.time_window is None:
        manifest = context.manifest
        summary = {
            name: manifest[name]
            for name in (
                "file_count",
                "row_count",
                "partitions",
                "partition_row_counts",
                "date_min",
                "date_max",
            )
        }
        summary["partitions"] = tuple(summary["partitions"])
        return summary
    timestamp = context.timestamp_column
    columns = tuple(
        dict.fromkeys(
            (
                *(
                    (timestamp,)
                    if timestamp is not None
                    else context.logical_columns[:1]
                ),
                *(name for name, _, _ in context.parquet_filters or ()),
            )
        )
    )
    partition_rows = {}
    files = set()
    rows = 0
    minimum = maximum = None
    with closing_iterator(
        iter_dataset_file_batches(
            context,
            columns=columns,
            batch_rows=batch_rows,
        )
    ) as batches:
        for path, frame in batches:
            if frame.empty:
                continue
            files.add(path)
            rows += len(frame)
            partition = relative_partition_label(context.data_root, path)
            partition_rows[partition] = partition_rows.get(partition, 0) + len(frame)
            if timestamp is not None:
                current_min, current_max = (
                    frame[timestamp].min(),
                    frame[timestamp].max(),
                )
                minimum = current_min if minimum is None else min(minimum, current_min)
                maximum = current_max if maximum is None else max(maximum, current_max)
    return {
        "file_count": len(files),
        "row_count": rows,
        "partitions": tuple(sorted(partition_rows)),
        "partition_row_counts": partition_rows,
        "date_min": date_string(minimum),
        "date_max": date_string(maximum),
    }
