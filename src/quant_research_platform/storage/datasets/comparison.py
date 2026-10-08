"""Dataset comparison implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.datasets.fingerprinting import (
    comparison_schema,
    fingerprint_dataset_session,
)
from quant_research_platform.storage.datasets.reading import (
    open_dataset_session,
)
from quant_research_platform.storage.datasets.types import DatasetComparison
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    filesystem_exists,
    require_artifact_root,
)


def compare_datasets(
    left_root: Path,
    right_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
) -> DatasetComparison:
    try:
        left_path = require_artifact_root(Path(left_root))
        if not filesystem_exists(left_path):
            return _comparison("MISSING_LEFT")
        right_path = require_artifact_root(Path(right_root))
        if not filesystem_exists(right_path):
            return _comparison("MISSING_RIGHT")
        left_context = open_dataset_session(
            left_path, columns=columns, time_window=time_window
        )
        right_context = open_dataset_session(
            right_path, columns=columns, time_window=time_window
        )
        key_columns = tuple(left_context.spec["primary_key"])
        same_schema = (
            left_context.logical_columns == right_context.logical_columns
            and comparison_schema(left_context) == comparison_schema(right_context)
            and key_columns == tuple(right_context.spec["primary_key"])
        )
        index = ExactPrimaryKeyIndex(primary_key=key_columns)
        changed = 0

        def fingerprint(context, namespace):
            def index_batch(batch, _frame):
                nonlocal changed
                table = pa.Table.from_batches([batch])
                values = tuple(
                    column
                    for column in context.logical_columns
                    if column not in key_columns
                )
                if namespace == "left":
                    duplicates = index.add_comparison_table(
                        table, namespace=namespace, value_columns=values
                    )
                else:
                    duplicates, different = index._compare_table(
                        table,
                        expected_namespace="left",
                        actual_namespace=namespace,
                        value_columns=values,
                    )
                    changed += different
                if duplicates:
                    raise ValueError("dataset_primary_key_duplicate")

            return fingerprint_dataset_session(
                context, batch_observer=index_batch if same_schema else None
            )

        left = fingerprint(left_context, "left")
        right = fingerprint(right_context, "right")
        layout_delta = {
            "file_count": {"left": left.file_count, "right": right.file_count},
            "partitions": {
                "left": tuple(
                    {
                        "partition": partition["partition"],
                        "row_count": partition["row_count"],
                    }
                    for partition in left.partition_fingerprints
                ),
                "right": tuple(
                    {
                        "partition": partition["partition"],
                        "row_count": partition["row_count"],
                    }
                    for partition in right.partition_fingerprints
                ),
            },
        }
        summary = {
            "schema_hash": {"left": left.schema_hash, "right": right.schema_hash},
            "row_count": {"left": left.row_count, "right": right.row_count},
            "key_columns": {"left": left.key_columns, "right": right.key_columns},
            "content_hash": {"left": left.content_hash, "right": right.content_hash},
        }
        if (
            left.schema_hash != right.schema_hash
            or left.column_names != right.column_names
            or comparison_schema(left_context) != comparison_schema(right_context)
        ):
            status = "SCHEMA_MISMATCH"
        elif left.row_count != right.row_count:
            status = "ROW_COUNT_MISMATCH"
        elif left.key_columns != right.key_columns:
            status = "KEY_MISMATCH"
        else:
            missing, extra = index.difference_counts(
                expected_namespace="left", actual_namespace="right"
            )
            exact = {
                "missing_row_count": missing,
                "extra_row_count": extra,
                "changed_row_count": changed,
            }
            summary["exact"] = exact
            status = "CONTENT_MISMATCH" if any(exact.values()) else "MATCH"
        left_context.verify()
        right_context.verify()
        return _comparison(status, summary=summary, layout_delta=layout_delta)
    except (MemoryError, OSError, ValueError):
        return _comparison(
            "INSPECTION_FAILED", summary={"error": "dataset_inspection_failed"}
        )


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
