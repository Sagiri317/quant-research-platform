"""Parquet writer helpers for local outputs."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

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
    effective_test_end: str


def write_parquet_chunks(
    frames: Iterable[pd.DataFrame],
    path: Path,
    *,
    required: tuple[str, ...],
    key_columns: tuple[str, ...],
    compression: str,
    date_only: bool = False,
    validate_frame=None,
) -> _PredictionOutputWriteSummary:
    path = Path(path)
    require_unaliased_path(path, error="invalid_output_write_path")
    row_count = 0
    effective_test_end = None
    with closing_iterator(iter(frames)) as active_frames:
        handle = open_filesystem_file(path, "xb")
        with closing_iterator(handle):
            require_unaliased_path(path.parent, error="invalid_output_write_path")
            with parquet_table_writer(handle, compression=compression) as write_table:
                key_index = ExactPrimaryKeyIndex(primary_key=key_columns)
                for frame in active_frames:
                    if not isinstance(frame, pd.DataFrame):
                        raise TypeError("invalid_artifact_output_frame")
                    selected = _prediction_output_chunk_frame(
                        frame, required=required, key_columns=key_columns
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
                        pd.Timestamp(timestamps.max()), date_only=date_only
                    )
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
                    write_table(table)
                    row_count += len(selected)
            handle.flush()
            os.fsync(handle.fileno())
    if row_count == 0 or effective_test_end is None:
        raise ValueError("empty_prediction_window")
    return _PredictionOutputWriteSummary(
        row_count=row_count, effective_test_end=effective_test_end
    )


def _prediction_output_chunk_frame(
    frame: pd.DataFrame, *, required: tuple[str, ...], key_columns: tuple[str, ...]
) -> pd.DataFrame:
    validate_output_frame(frame, required=required, error_name="prediction_output")
    selected = frame.loc[:, list(required)]
    if not selected.empty:
        _validate_prediction_output_values(
            selected,
            primary_key=key_columns,
            prediction_column=RAW_PREDICTION_COLUMN
            if RAW_PREDICTION_COLUMN in required
            else next(column for column in required if column not in key_columns),
        )
        if RAW_PREDICTION_COLUMN in required:
            if any(
                type(value) is not str or not value for value in selected["model_id"]
            ):
                raise ValueError("invalid_prediction_model_id")
            alpha = strict_finite_numeric_series(
                selected["alpha_score"], error="invalid_prediction_alpha_scores"
            )
            alpha_mean = alpha.groupby(
                [selected["timestamp"], selected["family_id"]], sort=False
            ).mean()
            if bool(alpha_mean.abs().gt(1e-12).any()):
                raise ValueError("invalid_prediction_alpha_scores")
    return selected


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
