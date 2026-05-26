"""Metadata-driven artifact table preview service."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from fastapi import HTTPException

from quant_research_platform.artifacts.display_tables import (
    display_table_artifact_path,
    display_table_preview_limits,
)
from quant_research_platform.interfaces.api.run_index import require_index_file


def table_preview_payload(
    output_dir: Path,
    manifest: dict[str, object],
    table_key: str,
    *,
    combination_id: str | None,
    limit: int | None,
    offset: int,
) -> dict[str, object]:
    """Return a bounded preview for a manifest.display table."""

    try:
        preview = display_table_preview_limits(manifest, table_key)
        relative_path = display_table_artifact_path(manifest, table_key)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if combination_id and not preview.supports_combination_filter:
        raise HTTPException(
            status_code=422,
            detail="table does not support combination_id filtering",
        )
    effective_limit = limit if limit is not None else preview.default_limit
    effective_limit = _validate_preview_limit(
        effective_limit,
        max_limit=preview.max_limit,
    )
    if offset and not preview.supports_offset:
        raise HTTPException(status_code=422, detail="table does not support offset")
    path = require_index_file(output_dir, relative_path)
    if path.suffix == ".parquet":
        return _parquet_preview_payload(
            path,
            combination_id=combination_id,
            limit=effective_limit,
            offset=offset,
        )
    return _csv_preview_payload(
        path,
        combination_id=combination_id,
        limit=effective_limit,
        offset=offset,
    )


def _csv_preview_payload(
    path: Path,
    *,
    combination_id: str | None,
    limit: int,
    offset: int,
) -> dict[str, object]:
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be nonnegative")
    total_rows = 0
    rows: list[pd.DataFrame] = []
    columns: list[str] | None = None
    target_count = offset + limit + 1
    observed_more = False
    for chunk in pd.read_csv(path, chunksize=65536):
        if columns is None:
            columns = list(chunk.columns)
        frame = chunk
        if combination_id and "combination_id" not in frame.columns:
            raise HTTPException(
                status_code=422,
                detail="table is missing combination_id column",
            )
        if combination_id:
            frame = frame[frame["combination_id"].astype(str) == combination_id]
        previous_total = total_rows
        total_rows += len(frame)
        if frame.empty:
            continue
        start = max(0, offset - previous_total)
        stop = min(len(frame), offset + limit - previous_total)
        if stop > start:
            rows.append(frame.iloc[start:stop])
        if total_rows >= target_count:
            observed_more = True
            break
    page = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=columns or [])
    )
    return _table_payload_page(
        page,
        total_rows=total_rows,
        limit=limit,
        offset=offset,
        has_more=observed_more,
        total_rows_exact=not observed_more,
    )


def _parquet_preview_payload(
    path: Path,
    *,
    combination_id: str | None,
    limit: int,
    offset: int,
) -> dict[str, object]:
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be nonnegative")
    parquet_file = pq.ParquetFile(path)
    if combination_id is None:
        total_rows = parquet_file.metadata.num_rows
        frame = _read_parquet_page(parquet_file, offset=offset, limit=limit)
        if frame.empty:
            frame = pd.DataFrame(columns=parquet_file.schema_arrow.names)
        return _table_payload_page(
            frame,
            total_rows=total_rows,
            limit=limit,
            offset=offset,
            has_more=(offset + limit) < total_rows,
            total_rows_exact=True,
        )

    total_rows = 0
    rows: list[pd.DataFrame] = []
    target_count = offset + limit + 1
    observed_more = False
    if combination_id and "combination_id" not in parquet_file.schema_arrow.names:
        raise HTTPException(
            status_code=422,
            detail="table is missing combination_id column",
        )
    for batch in parquet_file.iter_batches(batch_size=65536):
        frame = batch.to_pandas()
        if combination_id:
            frame = frame[frame["combination_id"].astype(str) == combination_id]
        previous_total = total_rows
        total_rows += len(frame)
        if frame.empty:
            continue
        start = max(0, offset - previous_total)
        stop = min(len(frame), offset + limit - previous_total)
        if stop > start:
            rows.append(frame.iloc[start:stop])
        if total_rows >= target_count:
            observed_more = True
            break
    page = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=parquet_file.schema_arrow.names)
    )
    return _table_payload_page(
        page,
        total_rows=total_rows,
        limit=limit,
        offset=offset,
        has_more=observed_more,
        total_rows_exact=not observed_more,
    )


def _read_parquet_page(
    parquet_file: pq.ParquetFile,
    *,
    offset: int,
    limit: int,
) -> pd.DataFrame:
    target_end = offset + limit
    total_seen = 0
    rows: list[pd.DataFrame] = []
    for batch in parquet_file.iter_batches(batch_size=65536):
        frame = batch.to_pandas()
        previous_total = total_seen
        total_seen += len(frame)
        if total_seen <= offset:
            continue
        start = max(0, offset - previous_total)
        stop = min(len(frame), target_end - previous_total)
        if stop > start:
            rows.append(frame.iloc[start:stop])
        if total_seen >= target_end:
            break
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _table_payload_page(
    frame: pd.DataFrame,
    *,
    total_rows: int,
    limit: int | None,
    offset: int,
    has_more: bool | None = None,
    total_rows_exact: bool | None = None,
) -> dict[str, object]:
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    payload: dict[str, object] = {
        "columns": list(normalized.columns),
        "rows": normalized.to_dict(orient="records"),
        "total_rows": total_rows,
        "offset": offset,
        "limit": limit,
    }
    if has_more is not None:
        payload["has_more"] = has_more
    if total_rows_exact is not None:
        payload["total_rows_exact"] = total_rows_exact
    return payload


def _validate_preview_limit(limit: int, *, max_limit: int) -> int:
    if limit < 1 or limit > max_limit:
        raise HTTPException(
            status_code=422,
            detail=f"limit must be between 1 and {max_limit}",
        )
    return limit


__all__ = ["table_preview_payload"]
