"""Dataset inspection implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage._dataset_manifest_validation import (
    _manifest_partition_row_counts,
)
from quant_research_platform.storage._dataset_paths import _relative_partition_label
from quant_research_platform.storage._dataset_reading import (
    _build_dataset_read_context,
    _DatasetReadContext,
    _filter_frame_by_time_window,
)
from quant_research_platform.storage._dataset_schema import _date_string
from quant_research_platform.storage._dataset_types import DatasetInspection


def inspect_dataset(
    dataset_root: Path,
    *,
    time_window: TimeWindow | None = None,
) -> DatasetInspection:
    context = _build_dataset_read_context(
        dataset_root,
        columns=None,
        time_window=time_window,
    )
    runtime = _inspection_runtime_metadata(
        context,
    )
    return DatasetInspection(
        root=context.root,
        data_root=context.data_root,
        spec=context.spec,
        columns=tuple(name for name, _ in context.schema),
        schema_hash=context.schema_hash,
        file_count=int(runtime["file_count"]),
        row_count=int(runtime["row_count"]),
        partitions=tuple(runtime["partitions"]),
        partition_row_counts=dict(runtime["partition_row_counts"]),
        date_min=runtime["date_min"],
        date_max=runtime["date_max"],
        manifest=context.manifest,
    )


def _inspection_runtime_metadata(
    context: _DatasetReadContext,
) -> dict[str, Any]:
    if context.time_window is None:
        partition_row_counts = _manifest_partition_row_counts(context.manifest)
        return {
            "file_count": int(context.manifest["file_count"]),
            "row_count": int(context.manifest["row_count"]),
            "partitions": tuple(sorted(partition_row_counts)),
            "partition_row_counts": partition_row_counts,
            "date_min": context.manifest.get("date_min"),
            "date_max": context.manifest.get("date_max"),
        }
    timestamp_column = context.timestamp_column
    if timestamp_column is None:
        raise ValueError("time_window_not_supported")
    row_count = 0
    file_count = 0
    partition_row_counts: dict[str, int] = {}
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    for path in context.read_paths:
        frame = pd.read_parquet(
            path,
            columns=[timestamp_column],
            engine="pyarrow",
            filters=context.parquet_filters,
        )
        filtered = _filter_frame_by_time_window(
            frame,
            spec=context.spec,
            time_window=context.time_window,
        )
        if filtered.empty:
            continue
        file_count += 1
        partition = _relative_partition_label(context.data_root, path)
        partition_row_counts[partition] = partition_row_counts.get(partition, 0) + len(
            filtered
        )
        row_count += len(filtered)
        values = pd.to_datetime(filtered[timestamp_column], errors="coerce").dropna()
        if not values.empty:
            current_min = values.min()
            current_max = values.max()
            date_min = current_min if date_min is None else min(date_min, current_min)
            date_max = current_max if date_max is None else max(date_max, current_max)
    return {
        "file_count": file_count,
        "row_count": int(row_count),
        "partitions": tuple(sorted(partition_row_counts)),
        "partition_row_counts": dict(sorted(partition_row_counts.items())),
        "date_min": _date_string(date_min),
        "date_max": _date_string(date_max),
    }
