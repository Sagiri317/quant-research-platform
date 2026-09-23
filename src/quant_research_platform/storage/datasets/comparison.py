"""Dataset comparison implementation."""

from __future__ import annotations

import sqlite3
import struct
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pyarrow as pa

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    _close_lease_preserving_primary,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets.fingerprinting import (
    _fingerprint_dataset_session_owned,
    dataset_fingerprint_retained_memory_bytes,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetSession,
    iter_dataset_file_batches,
    open_dataset_session,
)
from quant_research_platform.storage.datasets.types import DatasetComparison
from quant_research_platform.storage.exact_keys import (
    EXACT_KEY_SQLITE_CACHE_BYTES,
    ExactPrimaryKeyIndex,
)
from quant_research_platform.storage.identity import require_artifact_root


def compare_datasets(
    left_root: Path,
    right_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
    resource_policy: RuntimeResourcePolicy | None,
) -> DatasetComparison:
    try:
        left_path = require_artifact_root(Path(left_root))
    except ValueError:
        return _comparison(
            "INSPECTION_FAILED",
            summary={"error": "dataset_inspection_failed"},
        )
    if not left_path.exists():
        return _comparison("MISSING_LEFT")
    try:
        right_path = require_artifact_root(Path(right_root))
    except ValueError:
        return _comparison(
            "INSPECTION_FAILED",
            summary={"error": "dataset_inspection_failed"},
        )
    if not right_path.exists():
        return _comparison("MISSING_RIGHT")
    requested_policy = resource_policy or current_runtime_resource_policy()
    with runtime_resource_policy_scope(requested_policy) as issued_policy:
        return _compare_datasets_under_policy(
            left_path,
            right_path,
            columns=columns,
            time_window=time_window,
            resource_policy=issued_policy,
        )


def _compare_datasets_under_policy(
    left_root: Path,
    right_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
    time_window: TimeWindow | None = None,
    resource_policy: RuntimeResourcePolicy,
) -> DatasetComparison:
    try:
        left_path = require_artifact_root(Path(left_root))
        if not left_path.exists():
            return _comparison("MISSING_LEFT")
        right_path = require_artifact_root(Path(right_root))
        if not right_path.exists():
            return _comparison("MISSING_RIGHT")
        issued_policy = resource_policy
        with ExitStack() as retained:
            left_context = retained.enter_context(
                open_dataset_session(
                    left_path,
                    columns=columns,
                    time_window=time_window,
                    resource_policy=issued_policy,
                )
            )
            right_context = retained.enter_context(
                open_dataset_session(
                    right_path,
                    columns=columns,
                    time_window=time_window,
                    resource_policy=issued_policy,
                )
            )
            left_owned = _fingerprint_dataset_session_owned(
                left_context, resource_policy=issued_policy
            )
            left = left_owned.value

            def release_left() -> None:
                nonlocal left
                left = None
                left_owned.release(owner="storage.dataset.comparison.left.fingerprint")

            retained.callback(release_left)
            right_owned = _fingerprint_dataset_session_owned(
                right_context, resource_policy=issued_policy
            )
            right = right_owned.value

            def release_right() -> None:
                nonlocal right
                right = None
                right_owned.release(
                    owner="storage.dataset.comparison.right.fingerprint"
                )

            retained.callback(release_right)
            comparison_result_lease = issued_policy.lease(
                "storage.dataset.comparison.result",
                _comparison_result_bound(left, right),
            )
            layout_delta: dict[str, Any] | None = None
            summary: dict[str, Any] | None = None

            def release_comparison_result() -> None:
                if sys.exception() is not None:
                    if summary is not None:
                        summary.clear()
                    if layout_delta is not None:
                        layout_delta.clear()
                _close_lease_preserving_primary(
                    comparison_result_lease,
                    owner="storage.dataset.comparison.result",
                )

            retained.callback(release_comparison_result)
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
                "key_columns": {
                    "left": left.key_columns,
                    "right": right.key_columns,
                },
                "content_hash": {
                    "left": left.content_hash,
                    "right": right.content_hash,
                },
            }
            if (
                left.schema_hash != right.schema_hash
                or left.column_names != right.column_names
            ):
                status = "SCHEMA_MISMATCH"
            elif left.row_count != right.row_count:
                status = "ROW_COUNT_MISMATCH"
            elif left.key_columns != right.key_columns:
                status = "KEY_MISMATCH"
            elif left.content_hash != right.content_hash:
                status = "CONTENT_MISMATCH"
            else:
                try:
                    exact = _exact_comparison_counts(
                        left_context,
                        right_context,
                        resource_policy=issued_policy,
                        primary_key=left.key_columns,
                    )
                except (MemoryError, OSError, ValueError, sqlite3.Error):
                    return _comparison(
                        "INSPECTION_FAILED",
                        summary={"error": "dataset_inspection_failed"},
                    )
                summary["exact"] = exact
                status = "CONTENT_MISMATCH" if any(exact.values()) else "MATCH"
            left_context.verify()
            right_context.verify()
            return _comparison(status, summary=summary, layout_delta=layout_delta)
    except (MemoryError, OSError, ValueError, sqlite3.Error):
        return _comparison(
            "INSPECTION_FAILED",
            summary={"error": "dataset_inspection_failed"},
        )


def _comparison_result_bound(left: Any, right: Any) -> int:
    pointer_bytes = struct.calcsize("P")
    return (
        dataset_fingerprint_retained_memory_bytes(left)
        + dataset_fingerprint_retained_memory_bytes(right)
        + pointer_bytes * 32
        + 1024
    )


def _exact_comparison_counts(
    left: DatasetSession,
    right: DatasetSession,
    *,
    resource_policy: RuntimeResourcePolicy,
    primary_key: tuple[str, ...],
) -> dict[str, int]:
    with resource_policy.lease(
        "storage.dataset.comparison.index", EXACT_KEY_SQLITE_CACHE_BYTES
    ):
        with ExactPrimaryKeyIndex(primary_key=primary_key) as index:
            for context, namespace in ((left, "left"), (right, "right")):
                _index_dataset_rows(
                    context,
                    resource_policy=resource_policy,
                    index=index,
                    namespace=namespace,
                )
            missing, extra, changed = index.comparison_counts(
                expected_namespace="left",
                actual_namespace="right",
            )
    return {
        "missing_row_count": missing,
        "extra_row_count": extra,
        "changed_row_count": changed,
    }


def _index_dataset_rows(
    context: DatasetSession,
    *,
    resource_policy: RuntimeResourcePolicy,
    index: ExactPrimaryKeyIndex,
    namespace: str,
) -> None:
    value_columns = tuple(
        column for column in context.logical_columns if column not in index.primary_key
    )
    comparison_columns = (*index.primary_key, *value_columns)
    headroom = resource_policy.operation_headroom_bytes
    if headroom <= 0:
        raise ValueError("dataset_working_set_exceeds_task_memory_budget")
    exact_payload_target = headroom
    reader_headroom = headroom
    duplicates = 0

    def index_batch(batch: pa.RecordBatch, _frame: Any) -> None:
        nonlocal duplicates
        callback_headroom = resource_policy.operation_headroom_bytes
        callback_target = min(exact_payload_target, callback_headroom)
        if callback_target <= 0:
            raise ValueError("dataset_working_set_exceeds_task_memory_budget")
        duplicates += index.add_comparison_table(
            pa.Table.from_batches([batch]),
            namespace=namespace,
            value_columns=value_columns,
            payload_target_bytes=callback_target,
        )

    for _path, _frame in iter_dataset_file_batches(
        context,
        columns=comparison_columns,
        target_chunk_bytes=max(
            1,
            min(resource_policy.planning_headroom_bytes, reader_headroom),
        ),
        operation_headroom_bytes=reader_headroom,
        batch_observer=index_batch,
    ):
        pass
    if duplicates:
        raise ValueError("dataset_primary_key_duplicate")


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
