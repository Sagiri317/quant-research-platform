"""Parquet writer helpers for local outputs."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_research_platform.contracts.validation import strict_timestamp_series
from quant_research_platform.storage.atomic import reject_if_exists


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
    seen_keys: set[tuple[object, ...]] = set()
    row_count = 0
    effective_test_end: str | None = None
    try:
        temp_path.touch(exist_ok=False)
        claimed_temp = True
        for frame in frames:
            selected = _prediction_output_chunk_frame(
                frame,
                required=required,
                key_columns=key_columns,
                seen_keys=seen_keys,
            )
            if selected.empty:
                continue
            timestamps = strict_timestamp_series(
                selected["timestamp"], error="prediction_metadata_inconsistent"
            )
            current_end = pd.Timestamp(timestamps.max()).date().isoformat()
            if effective_test_end is None or pd.Timestamp(current_end) > pd.Timestamp(
                effective_test_end
            ):
                effective_test_end = current_end
            table = pa.Table.from_pandas(selected, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temp_path,
                    table.schema,
                    compression=compression,
                )
            writer.write_table(table)
            row_count += len(selected)
        if writer is None or row_count == 0 or effective_test_end is None:
            raise ValueError("empty_prediction_window")
        writer.close()
        writer = None
        reject_if_exists(final_path)
        _rename_with_transient_permission_retry(temp_path, final_path)
    except Exception:
        if writer is not None:
            writer.close()
        if claimed_temp and temp_path.exists():
            temp_path.unlink()
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
    seen_keys: set[tuple[object, ...]],
) -> pd.DataFrame:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"missing_prediction_output_columns: {', '.join(missing)}")
    extra = sorted(set(frame.columns).difference(required))
    if extra:
        raise ValueError(f"unregistered_prediction_output_columns: {', '.join(extra)}")
    selected = frame.loc[:, list(required)]
    if bool(selected.duplicated(list(key_columns)).any()):
        raise ValueError("duplicate_prediction_keys")
    for key in selected.loc[:, list(key_columns)].itertuples(index=False, name=None):
        normalized_key = tuple(key)
        if normalized_key in seen_keys:
            raise ValueError("duplicate_prediction_keys")
        seen_keys.add(normalized_key)
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
        _rename_with_transient_permission_retry(temp_path, final_path)
    except Exception:
        if claimed_temp and temp_path.exists():
            temp_path.unlink()
        raise
    return final_path


def _rename_with_transient_permission_retry(
    source: Path,
    target: Path,
    *,
    attempts: int = 3,
    delay_seconds: float = 0.10,
) -> None:
    for attempt in range(1, attempts + 1):
        try:
            source.rename(target)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            time.sleep(delay_seconds)
