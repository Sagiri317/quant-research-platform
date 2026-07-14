"""Dataset inspection implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.storage._dataset_manifest_validation import (
    manifest_partition_row_counts,
)
from quant_research_platform.storage._dataset_paths import relative_partition_label
from quant_research_platform.storage._dataset_reading import (
    DatasetReadContext,
    build_dataset_read_context,
    iter_dataset_file_batches,
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
) -> DatasetInspection:
    context = build_dataset_read_context(
        dataset_root,
        columns=None,
        time_window=time_window,
        require_consumable=False,
    )
    runtime = _inspection_runtime_metadata(
        context,
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


def _inspection_runtime_metadata(
    context: DatasetReadContext,
) -> dict[str, Any]:
    if context.time_window is None:
        partition_row_counts = manifest_partition_row_counts(context.manifest)
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
    partition_row_counts: dict[str, int] = {}
    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    observed_paths: set[Path] = set()
    for path, frame in iter_dataset_file_batches(
        context,
        columns=(timestamp_column,),
    ):
        if frame.empty:
            continue
        observed_paths.add(path)
        partition = relative_partition_label(context.data_root, path)
        partition_row_counts[partition] = partition_row_counts.get(partition, 0) + len(
            frame
        )
        row_count += len(frame)
        values = strict_timestamp_series(
            frame[timestamp_column],
            error="invalid_time_window_timestamp_value",
        ).dropna()
        if not values.empty:
            current_min = values.min()
            current_max = values.max()
            date_min = current_min if date_min is None else min(date_min, current_min)
            date_max = current_max if date_max is None else max(date_max, current_max)
    return {
        "file_count": len(observed_paths),
        "row_count": int(row_count),
        "partitions": tuple(sorted(partition_row_counts)),
        "partition_row_counts": dict(sorted(partition_row_counts.items())),
        "date_min": date_string(date_min),
        "date_max": date_string(date_max),
    }
