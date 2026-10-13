"""Parquet writer helpers for local outputs."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from quant_research_platform.artifacts.outputs.prediction_validation import (
    validate_prediction_frame,
)
from quant_research_platform.contracts.time import time_value_isoformat
from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.policy.lifecycle import closing_iterator
from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex
from quant_research_platform.storage.identity import (
    open_filesystem_file,
    require_unaliased_path,
)
from quant_research_platform.storage.parquet import (
    parquet_table_writer,
)


@dataclass(frozen=True)
class _PredictionOutputWriteSummary:
    row_count: int
    effective_prediction_end: str


def write_parquet_chunks(
    frames: Iterable[pd.DataFrame],
    path: Path,
    *,
    metadata: dict,
) -> _PredictionOutputWriteSummary:
    path = Path(path)
    require_unaliased_path(path, error="invalid_output_write_path")
    row_count = 0
    effective_prediction_end = None
    with closing_iterator(iter(frames)) as active_frames:
        handle = open_filesystem_file(path, "xb")
        with closing_iterator(handle):
            require_unaliased_path(path.parent, error="invalid_output_write_path")
            with parquet_table_writer(handle, compression="zstd") as write_table:
                key_index = ExactPrimaryKeyIndex(primary_key=("timestamp", "asset_id"))
                for frame in active_frames:
                    if not isinstance(frame, pd.DataFrame):
                        raise TypeError("invalid_artifact_output_frame")
                    validate_prediction_frame(frame, metadata)
                    if frame.empty:
                        continue
                    timestamps = strict_timestamp_series(
                        frame["timestamp"],
                        error="prediction_metadata_inconsistent",
                    )
                    selected = frame.assign(timestamp=timestamps)
                    current_end = time_value_isoformat(pd.Timestamp(timestamps.max()))
                    if effective_prediction_end is None or pd.Timestamp(
                        current_end
                    ) > pd.Timestamp(effective_prediction_end):
                        effective_prediction_end = current_end
                    table = pa.Table.from_pandas(selected, preserve_index=False)
                    try:
                        duplicates = key_index.add_table(table)
                    except ValueError as exc:
                        raise ValueError("invalid_prediction_keys") from exc
                    if duplicates:
                        raise ValueError("duplicate_prediction_keys")
                    write_table(table)
                    row_count += len(selected)
            handle.flush()
            os.fsync(handle.fileno())
    if row_count == 0 or effective_prediction_end is None:
        raise ValueError("empty_prediction_window")
    return _PredictionOutputWriteSummary(
        row_count=row_count, effective_prediction_end=effective_prediction_end
    )


def write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """Write one member inside its caller-owned atomic bundle staging directory."""
    path = Path(path)
    require_unaliased_path(path, error="invalid_output_write_path")
    with closing_iterator(open_filesystem_file(path, "xb")) as handle:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        with parquet_table_writer(handle, compression="zstd") as write_table:
            write_table(table)
        handle.flush()
        os.fsync(handle.fileno())
    return path
