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
from quant_research_platform.artifacts.outputs.common_validation import (
    validate_output_frame,
)
from quant_research_platform.contracts.predictions import (
    RAW_PREDICTION_COLUMN,
    _validate_prediction_output_values,
)
from quant_research_platform.contracts.time import time_value_isoformat
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.runtime.resources import (
    MemoryLease,
    OwnedValue,
    pandas_frame_payload_bytes,
    require_free_disk_headroom,
)
from quant_research_platform.storage.atomic import publish_no_replace, reject_if_exists
from quant_research_platform.storage.exact_keys import (
    EXACT_KEY_SQLITE_CACHE_BYTES,
    ExactPrimaryKeyIndex,
)
from quant_research_platform.storage.identity import (
    native_filesystem_path,
    require_unaliased_path,
)
from quant_research_platform.storage.parquet import (
    _PARQUET_NATIVE_FIXED_WORKSPACE_BYTES,
    _parquet_frame_reservation,
    _parquet_row_group_size,
    _parquet_table_write_state,
    _parquet_writer_fixed_reservation,
    parquet_table_writer,
)


@dataclass(frozen=True)
class _PredictionOutputWriteSummary:
    row_count: int
    effective_test_end: str


def _unwrap_frame_input(
    item: pd.DataFrame | OwnedValue[pd.DataFrame],
) -> tuple[pd.DataFrame, MemoryLease | None]:
    if isinstance(item, OwnedValue):
        if not isinstance(item.value, pd.DataFrame):
            raise TypeError("invalid_artifact_output_frame")
        return item.value, item.lease
    if not isinstance(item, pd.DataFrame):
        raise TypeError("invalid_artifact_output_frame")
    return item, None


def write_parquet_chunks(
    frames: Iterable[pd.DataFrame | OwnedValue[pd.DataFrame]],
    path: Path,
    *,
    required: tuple[str, ...],
    key_columns: tuple[str, ...],
    compression: str,
    date_only: bool = False,
    borrowed_input: bool = False,
    validate_frame=None,
) -> _PredictionOutputWriteSummary:
    final_path = Path(path)
    reject_if_exists(final_path)
    require_unaliased_path(final_path.parent, error="invalid_output_write_path")
    require_free_disk_headroom(
        final_path.parent, reserve_bytes=_PARQUET_NATIVE_FIXED_WORKSPACE_BYTES
    )
    final_path.parent.mkdir(parents=True, exist_ok=True)
    require_unaliased_path(final_path, error="invalid_output_write_path")
    temp_path = final_path.with_name(
        scratch_owner.owned_scratch_name(".f-", uuid4().hex[:12])
    )
    require_unaliased_path(temp_path, error="invalid_output_write_path")
    claimed_temp = False
    row_count = 0
    effective_test_end = None
    try:
        with (
            _parquet_writer_fixed_reservation(
                writer_count=0, index_bytes=EXACT_KEY_SQLITE_CACHE_BYTES
            ),
            closing_iterator(iter(frames)) as active_frames,
        ):
            try:
                first_item = next(active_frames)
            except StopIteration:
                raise ValueError("empty_prediction_window") from None
            first_frame, first_lease = _unwrap_frame_input(first_item)
            with ExitStack() as first_frame_state:
                first_frame_state.enter_context(
                    _parquet_frame_reservation(
                        first_frame,
                        key_columns=key_columns,
                        owned_lease=first_lease,
                        borrowed_input=borrowed_input,
                    )
                )
                handle = scratch_owner.open_owned_scratch_file(temp_path)
                claimed_temp = True
                with scratch_owner.handle_preserving_primary(handle):
                    require_unaliased_path(
                        final_path.parent, error="invalid_output_write_path"
                    )
                    with (
                        parquet_table_writer(
                            handle, parent=final_path.parent, compression=compression
                        ) as write_table,
                        ExactPrimaryKeyIndex(
                            primary_key=key_columns,
                            timestamp_columns=tuple(
                                column
                                for column in key_columns
                                if column == "timestamp"
                            ),
                        ) as key_index,
                    ):
                        first_pending = True
                        for item in chain((first_item,), active_frames):
                            frame, frame_lease = _unwrap_frame_input(item)
                            is_first = first_pending
                            first_pending = False
                            try:
                                with ExitStack() as frame_state:
                                    if not is_first:
                                        frame_state.enter_context(
                                            _parquet_frame_reservation(
                                                frame,
                                                key_columns=key_columns,
                                                owned_lease=frame_lease,
                                                borrowed_input=borrowed_input,
                                            )
                                        )
                                    selected = _prediction_output_chunk_frame(
                                        frame,
                                        required=required,
                                        key_columns=key_columns,
                                    )
                                    if selected.empty:
                                        continue
                                    if validate_frame is not None:
                                        validate_frame(selected)
                                    timestamps = strict_timestamp_series(
                                        selected["timestamp"],
                                        error="prediction_metadata_inconsistent",
                                    )
                                    current_end = time_value_isoformat(
                                        pd.Timestamp(timestamps.max()),
                                        date_only=date_only,
                                    )
                                    if effective_test_end is None or pd.Timestamp(
                                        current_end
                                    ) > pd.Timestamp(effective_test_end):
                                        effective_test_end = current_end
                                    table = pa.Table.from_pandas(
                                        selected, preserve_index=False
                                    )
                                    try:
                                        duplicates = key_index.add_table(table)
                                    except ValueError as exc:
                                        raise ValueError(
                                            "invalid_prediction_keys"
                                        ) from exc
                                    if duplicates:
                                        raise ValueError("duplicate_prediction_keys")
                                    write_table(table)
                                    row_count += len(selected)
                            finally:
                                frame = selected = table = timestamps = item = None
                                if is_first:
                                    first_frame_state.close()
                                    first_frame = first_item = None
                    handle.flush()
                    os.fsync(handle.fileno())
        if row_count == 0 or effective_test_end is None:
            raise ValueError("empty_prediction_window")
        reject_if_exists(final_path)
        require_unaliased_path(temp_path, error="invalid_output_write_path")
        publish_no_replace(temp_path, final_path)
    except BaseException as primary:
        if claimed_temp:
            try:
                require_unaliased_path(temp_path, error="invalid_output_write_path")
                scratch_owner.remove_owned_scratch(temp_path)
            except BaseException as cleanup_error:
                primary.add_note(
                    "prediction temporary cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        raise
    return _PredictionOutputWriteSummary(
        row_count=row_count, effective_test_end=effective_test_end
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
            prediction_column=(
                RAW_PREDICTION_COLUMN
                if RAW_PREDICTION_COLUMN in required
                else next(column for column in required if column not in key_columns)
            ),
        )
        if RAW_PREDICTION_COLUMN in required:
            if any(
                type(value) is not str or not value for value in selected["model_id"]
            ):
                raise ValueError("invalid_prediction_model_id")
            alpha = strict_finite_numeric_series(
                selected["alpha_score"],
                error="invalid_prediction_alpha_scores",
            )
            alpha_mean = alpha.groupby(
                [selected["timestamp"], selected["family_id"]], sort=False
            ).mean()
            if bool(alpha_mean.abs().gt(1e-12).any()):
                raise ValueError("invalid_prediction_alpha_scores")
    return selected


def write_parquet(
    frame: pd.DataFrame,
    path: Path,
    **kwargs: object,
) -> Path:
    final_path = Path(path)
    reject_if_exists(final_path)
    require_unaliased_path(final_path.parent, error="invalid_output_write_path")
    payload_bytes = pandas_frame_payload_bytes(frame)
    require_free_disk_headroom(
        final_path.parent,
        reserve_bytes=max(
            _PARQUET_NATIVE_FIXED_WORKSPACE_BYTES,
            payload_bytes * 2,
        ),
    )
    Path(native_filesystem_path(final_path.parent)).mkdir(parents=True, exist_ok=True)
    require_unaliased_path(final_path, error="invalid_output_write_path")
    temp_path = final_path.with_name(
        scratch_owner.owned_scratch_name(".f-", uuid4().hex[:12])
    )
    require_unaliased_path(temp_path, error="invalid_output_write_path")
    claimed_temp = False
    try:
        with _parquet_frame_reservation(
            frame,
        ):
            handle = scratch_owner.open_owned_scratch_file(temp_path)
            claimed_temp = True
            with scratch_owner.handle_preserving_primary(handle):
                require_unaliased_path(
                    final_path.parent,
                    error="invalid_output_write_path",
                )
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
                    _parquet_table_write_state(table)
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
        publish_no_replace(temp_path, final_path)
    except BaseException as primary:
        if claimed_temp:
            try:
                require_unaliased_path(temp_path, error="invalid_output_write_path")
                scratch_owner.remove_owned_scratch(temp_path)
            except BaseException as cleanup_error:
                primary.add_note(
                    "prediction temporary cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        raise
    return final_path
