"""Dataset comparison implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage._dataset_fingerprinting import (
    fingerprint_dataset,
    partition_keys,
)
from quant_research_platform.storage._dataset_types import DatasetComparison


def compare_datasets(
    left_root: Path,
    right_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
) -> DatasetComparison:
    left_path = Path(left_root)
    right_path = Path(right_root)
    if not left_path.exists():
        return _comparison("MISSING_LEFT")
    if not right_path.exists():
        return _comparison("MISSING_RIGHT")
    try:
        left = fingerprint_dataset(
            left_path,
            columns=columns,
            time_window=time_window,
        )
        right = fingerprint_dataset(
            right_path,
            columns=columns,
            time_window=time_window,
        )
    except Exception as exc:
        return _comparison("INSPECTION_FAILED", summary={"error": str(exc)})

    layout_delta = {
        "file_count": {"left": left.file_count, "right": right.file_count},
    }
    summary = {
        "schema_hash": {"left": left.schema_hash, "right": right.schema_hash},
        "row_count": {"left": left.row_count, "right": right.row_count},
        "key_columns": {"left": left.key_columns, "right": right.key_columns},
        "content_hash": {"left": left.content_hash, "right": right.content_hash},
        "partitions": {
            "left": left.partition_fingerprints,
            "right": right.partition_fingerprints,
        },
    }
    if left.schema_hash != right.schema_hash or left.column_names != right.column_names:
        status = "SCHEMA_MISMATCH"
    elif left.row_count != right.row_count:
        status = "ROW_COUNT_MISMATCH"
    elif left.key_columns != right.key_columns:
        status = "KEY_MISMATCH"
    elif partition_keys(left) != partition_keys(right):
        status = "PARTITION_MISMATCH"
    elif left.content_hash != right.content_hash:
        status = "CONTENT_MISMATCH"
    else:
        status = "MATCH"
    return _comparison(status, summary=summary, layout_delta=layout_delta)


def _comparison(
    status: str,
    *,
    summary: dict[str, Any] | None = None,
    layout_delta: dict[str, Any] | None = None,
) -> DatasetComparison:
    return DatasetComparison(
        status=status,
        matches=status == "MATCH",
        summary={} if summary is None else summary,
        layout_delta={} if layout_delta is None else layout_delta,
    )
