"""Parquet writer helpers for local outputs."""

from __future__ import annotations

import os
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import quant_research_platform.storage._scratch_ownership as scratch_owner
from quant_research_platform.artifacts._output_common_validation import (
    validate_output_frame,
)
from quant_research_platform.contracts.predictions import (
    _validate_prediction_output_values,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
)
from quant_research_platform.storage._dataset_key_tracking import (
    EXACT_KEY_SQLITE_PAGE_BYTES,
    ExactPrimaryKeyIndex,
)
from quant_research_platform.storage._dataset_paths import require_unaliased_path
from quant_research_platform.storage._dataset_physical import (
    _PARQUET_NATIVE_FIXED_WORKSPACE_BYTES,
    _parquet_row_group_footer_upper_bound,
    _parquet_row_group_size,
    _table_parquet_page_working_set_bytes,
    table_with_decoded_row_bounds,
)
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
    decoded_metadata: dict[bytes, bytes] | None = None
    writer_state = ExitStack()
    writer_memory = (
        (_PARQUET_NATIVE_FIXED_WORKSPACE_BYTES, 0)
        if current_runtime_resource_policy() is not None
        else (0, 0)
    )
    try:
        with _artifact_writer_fixed_reservation(
            writer_count=1,
            index_bytes=EXACT_KEY_SQLITE_PAGE_BYTES,
        ):
            with closing_dataset_iterator(iter(frames)) as active_frames:
                try:
                    first_frame = next(active_frames)
                except StopIteration:
                    raise ValueError("empty_prediction_window") from None
                with ExitStack() as first_frame_state:
                    first_frame_state.enter_context(
                        _artifact_frame_reservation(
                            first_frame,
                            key_columns=key_columns,
                        )
                    )
                    handle, claimed_temp = scratch_owner.open_owned_scratch_file(
                        temp_path
                    )
                    with scratch_owner.handle_preserving_primary(handle):
                        require_owned_scratch(final_path.parent, parent_identity)
                        try:
                            with ExactPrimaryKeyIndex(
                                primary_key=key_columns,
                                timestamp_columns=tuple(
                                    column
                                    for column in key_columns
                                    if column == "timestamp"
                                ),
                                scratch_parent=temp_path.parent,
                            ) as key_index:
                                first_pending = True
                                for frame in chain((first_frame,), active_frames):
                                    is_first = first_pending
                                    first_pending = False
                                    try:
                                        with ExitStack() as frame_state:
                                            if not is_first:
                                                frame_state.enter_context(
                                                    _artifact_frame_reservation(
                                                        frame,
                                                        key_columns=key_columns,
                                                    )
                                                )
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
                                            if (
                                                effective_test_end is None
                                                or pd.Timestamp(current_end)
                                                > pd.Timestamp(effective_test_end)
                                            ):
                                                effective_test_end = current_end
                                            table = pa.Table.from_pandas(
                                                selected,
                                                preserve_index=False,
                                            )
                                            (
                                                persisted,
                                                table_state,
                                                batch_rows,
                                                writer_memory,
                                            ) = (
                                                _artifact_table_write_state(
                                                    table,
                                                    previous_metadata=decoded_metadata,
                                                    retained_memory=writer_memory,
                                                )
                                            )
                                            writer_state.enter_context(table_state)
                                            decoded_metadata = dict(
                                                persisted.schema.metadata or {}
                                            )
                                            try:
                                                duplicates = key_index.add_table(
                                                    persisted
                                                )
                                            except ValueError as exc:
                                                raise ValueError(
                                                    "invalid_prediction_keys"
                                                ) from exc
                                            if duplicates:
                                                raise ValueError(
                                                    "duplicate_prediction_keys"
                                                )
                                            if writer is None:
                                                writer = pq.ParquetWriter(
                                                    handle,
                                                    persisted.schema,
                                                    compression=compression,
                                                    write_batch_size=batch_rows,
                                                    max_rows_per_page=batch_rows,
                                                    use_dictionary=False,
                                                    write_statistics=False,
                                                    write_page_index=False,
                                                    write_page_checksum=False,
                                                )
                                            writer.write_table(
                                                persisted,
                                                row_group_size=(
                                                    _parquet_row_group_size(
                                                        persisted.num_rows
                                                    )
                                                ),
                                            )
                                            row_count += len(selected)
                                    finally:
                                        if is_first:
                                            first_frame_state.close()
                                            del first_frame
                        except BaseException:
                            if writer is not None:
                                try:
                                    writer.close()
                                except BaseException:
                                    pass
                                writer = None
                            writer_state.close()
                            raise
                        if writer is not None:
                            writer.add_key_value_metadata(decoded_metadata or {})
                            active_writer = writer
                            writer = None
                            active_writer.close()
                        writer_state.close()
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
        writer_state.close()
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
    if not selected.empty:
        _validate_prediction_output_values(
            selected,
            primary_key=key_columns,
            prediction_column=next(
                column for column in required if column not in key_columns
            ),
        )
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
        with _artifact_frame_reservation(frame):
            handle, claimed_temp = scratch_owner.open_owned_scratch_file(temp_path)
            with scratch_owner.handle_preserving_primary(handle):
                require_owned_scratch(final_path.parent, parent_identity)
                options = dict(kwargs)
                engine = options.pop("engine", None)
                if engine not in (None, "pyarrow"):
                    raise ValueError("unsupported_parquet_engine")
                preserve_index = options.pop("index", None)
                compression = options.pop("compression", "snappy")
                table = pa.Table.from_pandas(
                    frame,
                    preserve_index=preserve_index,
                )
                persisted, table_state, batch_rows, _retained = (
                    _artifact_table_write_state(table)
                )
                with table_state:
                    pq.write_table(
                        persisted,
                        handle,
                        compression=compression,
                        row_group_size=_parquet_row_group_size(persisted.num_rows),
                        write_batch_size=batch_rows,
                        max_rows_per_page=batch_rows,
                        use_dictionary=False,
                        write_statistics=False,
                        write_page_index=False,
                        write_page_checksum=False,
                        **options,
                    )
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


def _artifact_writer_fixed_reservation(
    *,
    writer_count: int,
    index_bytes: int,
) -> ExitStack:
    stack = ExitStack()
    policy = current_runtime_resource_policy()
    if policy is not None:
        stack.enter_context(
            policy.reserve(
                "artifacts.parquet.fixed",
                index_bytes
                + writer_count * _PARQUET_NATIVE_FIXED_WORKSPACE_BYTES,
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
        row_count = 0 if frame is None else len(frame)
        column_count = 0 if frame is None else len(frame.columns)
        index_bytes = (
            0 if frame is None else int(frame.index.memory_usage(deep=True))
        )
        arrow_bytes = frame_bytes + row_count * 9 * column_count
        timestamp_bytes = (
            index_bytes + 9 * row_count if "timestamp" in key_columns else 0
        )
        key_mask_bytes = index_bytes + row_count if key_columns else 0
        stack.enter_context(
            policy.reserve(
                "artifacts.parquet.frame",
                frame_bytes + arrow_bytes + timestamp_bytes + key_mask_bytes,
            )
        )
    return stack


def _artifact_table_write_state(
    table: pa.Table,
    *,
    previous_metadata: dict[bytes, bytes] | None = None,
    retained_memory: tuple[int, int] = (0, 0),
) -> tuple[pa.Table, ExitStack, int, tuple[int, int]]:
    persisted = table_with_decoded_row_bounds(
        table,
        previous_metadata=previous_metadata,
    )
    stack = ExitStack()
    policy = current_runtime_resource_policy()
    if policy is None:
        return persisted, stack, max(1, persisted.num_rows), (0, 0)
    retained_page_bytes, retained_footer_bytes = retained_memory
    retained_bytes = retained_page_bytes + retained_footer_bytes
    footer_increment = _parquet_row_group_footer_upper_bound(
        persisted.schema,
        rows=persisted.num_rows,
    )
    page_headroom = (
        policy.remaining_memory_bytes + retained_page_bytes - footer_increment
    )
    minimum_rows = 0 if persisted.num_rows == 0 else 1
    minimum_bytes = _table_parquet_page_working_set_bytes(
        persisted,
        rows=minimum_rows,
    )
    if minimum_bytes > page_headroom:
        raise MemoryError("runtime_memory_budget_exceeded")
    batch_rows = max(1, persisted.num_rows)
    working_bytes = minimum_bytes
    low = 1
    high = _parquet_row_group_size(persisted.num_rows)
    while low <= high:
        candidate = (low + high) // 2
        candidate_bytes = _table_parquet_page_working_set_bytes(
            persisted,
            rows=candidate,
        )
        if candidate_bytes <= page_headroom:
            batch_rows = candidate
            working_bytes = candidate_bytes
            low = candidate + 1
        else:
            high = candidate - 1
    next_memory = (
        max(retained_page_bytes, working_bytes),
        retained_footer_bytes + footer_increment,
    )
    next_retained = sum(next_memory)
    stack.enter_context(
        policy.reserve(
            "artifacts.parquet.native",
            next_retained - retained_bytes,
        )
    )
    return persisted, stack, batch_rows, next_memory
