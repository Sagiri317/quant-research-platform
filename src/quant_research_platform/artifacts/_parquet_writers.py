"""Parquet writer helpers for local outputs."""

from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import quant_research_platform.storage._scratch_ownership as scratch_owner
from quant_research_platform.artifacts._output_common_validation import (
    validate_output_frame,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage._dataset_key_tracking import ExactPrimaryKeyIndex
from quant_research_platform.storage._dataset_paths import require_unaliased_path
from quant_research_platform.storage._dataset_reading import closing_dataset_iterator
from quant_research_platform.storage._scratch_ownership import require_owned_scratch
from quant_research_platform.storage.atomic import publish_no_replace, reject_if_exists


@dataclass(frozen=True)
class _PredictionOutputWriteSummary:
    row_count: int
    effective_test_end: str


def write_parquet_chunks(
    frames: Iterable[pd.DataFrame],
    path: Path,
    *,
    required: tuple[str, ...],
    key_columns: tuple[str, ...],
    compression: str,
) -> _PredictionOutputWriteSummary:
    final_path = Path(path)
    reject_if_exists(final_path)
    require_unaliased_path(final_path.parent, error="invalid_output_write_path")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = final_path.parent.lstat()
    require_unaliased_path(final_path, error="invalid_output_write_path")
    scratch_owner.cleanup_stale_owned_scratch(
        final_path.parent,
        parent_identity=parent_identity,
        prefix=".f-",
        directory=False,
    )
    temp_path = final_path.with_name(
        scratch_owner.owned_scratch_name(".f-", uuid4().hex[:12])
    )
    require_unaliased_path(temp_path, error="invalid_output_write_path")
    writer: pq.ParquetWriter | None = None
    claimed_temp: os.stat_result | None = None
    row_count = 0
    effective_test_end: str | None = None
    try:
        with _artifact_writer_reservation():
            handle, claimed_temp = scratch_owner.open_owned_scratch_file(temp_path)
            with scratch_owner.handle_preserving_primary(handle):
                require_owned_scratch(final_path.parent, parent_identity)
                try:
                    with closing_dataset_iterator(iter(frames)) as active_frames:
                        with ExactPrimaryKeyIndex(
                            primary_key=key_columns,
                            timestamp_columns=tuple(
                                column
                                for column in key_columns
                                if column == "timestamp"
                            ),
                            scratch_parent=temp_path.parent,
                        ) as key_index:
                            for frame in active_frames:
                                with _artifact_frame_reservation(
                                    frame,
                                    key_columns=key_columns,
                                ):
                                    selected = _prediction_output_chunk_frame(
                                        frame,
                                        required=required,
                                        key_columns=key_columns,
                                    )
                                    if selected.empty:
                                        continue
                                    timestamps = strict_timestamp_series(
                                        selected["timestamp"],
                                        error="prediction_metadata_inconsistent",
                                    )
                                    current_end = (
                                        pd.Timestamp(timestamps.max())
                                        .date()
                                        .isoformat()
                                    )
                                    if effective_test_end is None or pd.Timestamp(
                                        current_end
                                    ) > pd.Timestamp(effective_test_end):
                                        effective_test_end = current_end
                                    table = pa.Table.from_pandas(
                                        selected,
                                        preserve_index=False,
                                    )
                                    try:
                                        duplicates = key_index.add_table(table)
                                    except ValueError as exc:
                                        raise ValueError(
                                            "invalid_prediction_keys"
                                        ) from exc
                                    if duplicates:
                                        raise ValueError("duplicate_prediction_keys")
                                    if writer is None:
                                        writer = pq.ParquetWriter(
                                            handle,
                                            table.schema,
                                            compression=compression,
                                        )
                                    writer.write_table(table)
                                    row_count += len(selected)
                except BaseException:
                    if writer is not None:
                        try:
                            writer.close()
                        except BaseException:
                            pass
                        writer = None
                    raise
                if writer is not None:
                    active_writer = writer
                    writer = None
                    active_writer.close()
                handle.flush()
                os.fsync(handle.fileno())
        if row_count == 0 or effective_test_end is None:
            raise ValueError("empty_prediction_window")
        reject_if_exists(final_path)
        require_unaliased_path(temp_path, error="invalid_output_write_path")
        require_owned_scratch(temp_path, claimed_temp)
        publish_no_replace(temp_path, final_path, source_identity=claimed_temp)
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except BaseException:
                pass
        if claimed_temp is not None:
            try:
                require_unaliased_path(temp_path, error="invalid_output_write_path")
                scratch_owner.remove_owned_scratch(temp_path, claimed_temp)
            except BaseException:
                pass
        raise
    return _PredictionOutputWriteSummary(
        row_count=row_count,
        effective_test_end=effective_test_end,
    )


def _prediction_output_chunk_frame(
    frame: pd.DataFrame,
    *,
    required: tuple[str, ...],
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    validate_output_frame(
        frame,
        required=required,
        error_name="prediction_output",
    )
    selected = frame.loc[:, list(required)]
    if bool(selected.duplicated(list(key_columns)).any()):
        raise ValueError("duplicate_prediction_keys")
    return selected


def write_parquet(
    frame: pd.DataFrame,
    path: Path,
    **kwargs: object,
) -> Path:
    final_path = Path(path)
    reject_if_exists(final_path)
    require_unaliased_path(final_path.parent, error="invalid_output_write_path")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    parent_identity = final_path.parent.lstat()
    require_unaliased_path(final_path, error="invalid_output_write_path")
    scratch_owner.cleanup_stale_owned_scratch(
        final_path.parent,
        parent_identity=parent_identity,
        prefix=".f-",
        directory=False,
    )
    temp_path = final_path.with_name(
        scratch_owner.owned_scratch_name(".f-", uuid4().hex[:12])
    )
    require_unaliased_path(temp_path, error="invalid_output_write_path")
    claimed_temp: os.stat_result | None = None
    try:
        with _artifact_writer_reservation(), _artifact_frame_reservation(frame):
            handle, claimed_temp = scratch_owner.open_owned_scratch_file(temp_path)
            with scratch_owner.handle_preserving_primary(handle):
                require_owned_scratch(final_path.parent, parent_identity)
                frame.to_parquet(handle, **kwargs)
                handle.flush()
                os.fsync(handle.fileno())
        reject_if_exists(final_path)
        require_unaliased_path(temp_path, error="invalid_output_write_path")
        require_owned_scratch(temp_path, claimed_temp)
        publish_no_replace(temp_path, final_path, source_identity=claimed_temp)
    except BaseException:
        if claimed_temp is not None:
            try:
                require_unaliased_path(temp_path, error="invalid_output_write_path")
                scratch_owner.remove_owned_scratch(temp_path, claimed_temp)
            except BaseException:
                pass
        raise
    return final_path


def _artifact_writer_reservation() -> ExitStack:
    stack = ExitStack()
    policy = current_runtime_resource_policy()
    if policy is not None:
        stack.enter_context(
            policy.reserve(
                "artifacts.parquet.native",
                max(64 * 1024, policy.payload_target_bytes // 8),
            )
        )
    return stack


def _artifact_frame_reservation(
    frame: pd.DataFrame | None,
    *,
    key_columns: tuple[str, ...] = (),
) -> ExitStack:
    stack = ExitStack()
    policy = current_runtime_resource_policy()
    if policy is not None:
        frame_bytes = 0 if frame is None else pandas_frame_payload_bytes(frame)
        key_bytes = 0 if frame is None else len(frame) * (64 + 32 * len(key_columns))
        stack.enter_context(
            policy.reserve(
                "artifacts.parquet.frame",
                3 * frame_bytes + key_bytes,
            )
        )
    return stack
