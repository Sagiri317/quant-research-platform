"""Parquet writer helpers for local outputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_platform.artifacts._output_common_validation import (
    validate_output_frame,
)
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.storage._dataset_key_tracking import ExactPrimaryKeyIndex
from quant_research_platform.storage._dataset_reading import closing_dataset_iterator
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
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_name(f".{final_path.name}.tmp-{uuid4().hex[:12]}")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    claimed_temp = False
    row_count = 0
    effective_test_end: str | None = None
    try:
        temp_path.touch(exist_ok=False)
        claimed_temp = True
        with closing_dataset_iterator(iter(frames)) as active_frames:
            with ExactPrimaryKeyIndex(
                primary_key=key_columns,
                timestamp_columns=tuple(
                    column for column in key_columns if column == "timestamp"
                ),
                scratch_parent=temp_path.parent,
            ) as key_index:
                for frame in active_frames:
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
                    current_end = pd.Timestamp(timestamps.max()).date().isoformat()
                    if effective_test_end is None or pd.Timestamp(
                        current_end
                    ) > pd.Timestamp(effective_test_end):
                        effective_test_end = current_end
                    table = pa.Table.from_pandas(selected, preserve_index=False)
                    try:
                        duplicates = key_index.add_table(table)
                    except ValueError as exc:
                        raise ValueError("invalid_prediction_keys") from exc
                    if duplicates:
                        raise ValueError("duplicate_prediction_keys")
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temp_path,
                            table.schema,
                            compression=compression,
                        )
                    writer.write_table(table)
                    row_count += len(selected)
                if writer is not None:
                    active_writer = writer
                    writer = None
                    active_writer.close()
        if row_count == 0 or effective_test_end is None:
            raise ValueError("empty_prediction_window")
        reject_if_exists(final_path)
        publish_no_replace(temp_path, final_path)
    except BaseException:
        if writer is not None:
            try:
                writer.close()
            except BaseException:
                pass
        if claimed_temp:
            try:
                temp_path.unlink(missing_ok=True)
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
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_name(f".{final_path.name}.tmp-{uuid4().hex[:12]}")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    claimed_temp = False
    try:
        temp_path.touch(exist_ok=False)
        claimed_temp = True
        frame.to_parquet(temp_path, **kwargs)
        reject_if_exists(final_path)
        publish_no_replace(temp_path, final_path)
    except BaseException:
        if claimed_temp:
            try:
                temp_path.unlink(missing_ok=True)
            except BaseException:
                pass
        raise
    return final_path
