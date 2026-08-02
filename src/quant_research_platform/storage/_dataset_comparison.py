"""Dataset comparison implementation."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import pyarrow as pa

from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage._dataset_fingerprinting import (
    dataset_fingerprint_retained_memory_bytes,
    fingerprint_dataset_session,
)
from quant_research_platform.storage._dataset_key_tracking import (
    EXACT_KEY_SQLITE_PAGE_BYTES,
    ExactPrimaryKeyIndex,
)
from quant_research_platform.storage._dataset_paths import require_artifact_root
from quant_research_platform.storage._dataset_reading import (
    DatasetSession,
    iter_dataset_file_batches,
    open_dataset_session,
)
from quant_research_platform.storage._dataset_types import DatasetComparison


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
    issued_policy = resource_policy or active_runtime_resource_policy()
    with runtime_resource_policy_scope(issued_policy):
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
        with open_dataset_session(
            left_path,
            columns=columns,
            time_window=time_window,
            resource_policy=issued_policy,
            retained_payload_bytes=0,
        ) as left_context:
            left = fingerprint_dataset_session(
                left_context, resource_policy=issued_policy
            )
            left_identity = _session_identity_token(left_context)
            left_retained_bytes = dataset_fingerprint_retained_memory_bytes(left) + len(
                left_identity
            )
            if (
                left_context.retained_memory_bytes + left_retained_bytes
                > issued_policy.task_memory_budget_bytes
            ):
                raise ValueError("dataset_working_set_exceeds_task_memory_budget")
        with open_dataset_session(
            right_path,
            columns=columns,
            time_window=time_window,
            resource_policy=issued_policy,
            retained_payload_bytes=left_retained_bytes,
        ) as right_context:
            right = fingerprint_dataset_session(
                right_context,
                resource_policy=issued_policy,
                retained_payload_bytes=left_retained_bytes,
            )
            right_identity = _session_identity_token(right_context)
            right_retained_bytes = dataset_fingerprint_retained_memory_bytes(
                right
            ) + len(right_identity)
            if (
                left_retained_bytes
                + right_context.retained_memory_bytes
                + right_retained_bytes
                > issued_policy.task_memory_budget_bytes
            ):
                raise ValueError("dataset_working_set_exceeds_task_memory_budget")
    except (OSError, ValueError, sqlite3.Error):
        return _comparison(
            "INSPECTION_FAILED",
            summary={"error": "dataset_inspection_failed"},
        )

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
    comparison_retained_bytes = 2 * (left_retained_bytes + right_retained_bytes)
    if comparison_retained_bytes > issued_policy.task_memory_budget_bytes:
        return _comparison(
            "INSPECTION_FAILED",
            summary={"error": "dataset_inspection_failed"},
        )
    if left.schema_hash != right.schema_hash or left.column_names != right.column_names:
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
                left_path,
                right_path,
                columns=columns,
                time_window=time_window,
                resource_policy=issued_policy,
                primary_key=left.key_columns,
                left_identity=left_identity,
                right_identity=right_identity,
                retained_payload_bytes=comparison_retained_bytes,
            )
        except (OSError, ValueError, sqlite3.Error):
            return _comparison(
                "INSPECTION_FAILED",
                summary={"error": "dataset_inspection_failed"},
            )
        summary["exact"] = exact
        status = "CONTENT_MISMATCH" if any(exact.values()) else "MATCH"
    return _comparison(status, summary=summary, layout_delta=layout_delta)


def _exact_comparison_counts(
    left_root: Path,
    right_root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
    resource_policy: RuntimeResourcePolicy,
    primary_key: tuple[str, ...],
    left_identity: bytes,
    right_identity: bytes,
    retained_payload_bytes: int,
) -> dict[str, int]:
    with ExactPrimaryKeyIndex(
        primary_key=primary_key,
        timestamp_columns=(),
        scratch_parent=left_root.parent,
        sqlite_cache_bytes=EXACT_KEY_SQLITE_PAGE_BYTES,
    ) as index:
        _index_dataset_rows(
            left_root,
            columns=columns,
            time_window=time_window,
            resource_policy=resource_policy,
            index=index,
            namespace="left",
            expected_identity=left_identity,
            retained_payload_bytes=retained_payload_bytes,
        )
        _index_dataset_rows(
            right_root,
            columns=columns,
            time_window=time_window,
            resource_policy=resource_policy,
            index=index,
            namespace="right",
            expected_identity=right_identity,
            retained_payload_bytes=retained_payload_bytes,
        )
        for root, expected_identity in (
            (left_root, left_identity),
            (right_root, right_identity),
        ):
            with open_dataset_session(
                root,
                columns=columns,
                time_window=time_window,
                resource_policy=resource_policy,
                retained_payload_bytes=(
                    retained_payload_bytes + index.retained_memory_bytes
                ),
            ) as context:
                if _session_identity_token(context) != expected_identity:
                    raise ValueError("dataset_identity_changed_during_comparison")
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
    root: Path,
    *,
    columns: tuple[str, ...] | list[str] | None,
    time_window: TimeWindow | None,
    resource_policy: RuntimeResourcePolicy,
    index: ExactPrimaryKeyIndex,
    namespace: str,
    expected_identity: bytes,
    retained_payload_bytes: int,
) -> None:
    with open_dataset_session(
        root,
        columns=columns,
        time_window=time_window,
        resource_policy=resource_policy,
        retained_payload_bytes=(retained_payload_bytes + index.retained_memory_bytes),
    ) as context:
        if _session_identity_token(context) != expected_identity:
            raise ValueError("dataset_identity_changed_during_comparison")
        value_columns = tuple(
            column
            for column in context.logical_columns
            if column not in index.primary_key
        )
        comparison_columns = (*index.primary_key, *value_columns)
        retained_before_session = retained_payload_bytes + index.retained_memory_bytes
        headroom = resource_policy.task_memory_budget_bytes
        headroom -= retained_before_session + context.retained_memory_bytes
        if headroom <= 0:
            raise ValueError("dataset_working_set_exceeds_task_memory_budget")
        exact_payload_target = min(
            resource_policy.payload_target_bytes,
            headroom // 3,
        )
        sqlite_binding_headroom = exact_payload_target
        reader_headroom = (
            resource_policy.task_memory_budget_bytes
            - retained_before_session
            - exact_payload_target
            - sqlite_binding_headroom
        )
        if exact_payload_target <= 0 or reader_headroom <= 0:
            raise ValueError("dataset_working_set_exceeds_task_memory_budget")
        duplicates = 0

        def index_batch(batch: pa.RecordBatch, _frame: Any) -> None:
            nonlocal duplicates
            duplicates += index.add_comparison_table(
                pa.Table.from_batches([batch]),
                namespace=namespace,
                value_columns=value_columns,
                payload_target_bytes=exact_payload_target,
            )

        for _path, _frame in iter_dataset_file_batches(
            context,
            columns=comparison_columns,
            target_chunk_bytes=min(
                resource_policy.payload_target_bytes,
                reader_headroom,
            ),
            operation_headroom_bytes=reader_headroom,
            batch_observer=index_batch,
        ):
            pass
        if duplicates:
            raise ValueError("dataset_primary_key_duplicate")
        if _session_identity_token(context) != expected_identity:
            raise ValueError("dataset_identity_changed_during_comparison")


def _session_identity_token(context: DatasetSession) -> bytes:
    digest = hashlib.sha256()

    def update(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    update(context.spec_bytes)
    update(context.spec_identity_bytes)
    update(context.manifest_bytes)
    update(context.manifest_identity_bytes)
    for path in context.all_paths:
        metadata = context.metadata_for(path)
        identity = metadata.identity
        update(path.relative_to(context.data_root).as_posix().encode("utf-8"))
        for value in (
            identity.device,
            identity.inode,
            identity.size,
            identity.modified_ns,
        ):
            update(str(value).encode("ascii"))
    return digest.digest()


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
