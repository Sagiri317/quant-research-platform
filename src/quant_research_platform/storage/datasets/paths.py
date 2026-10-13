"""Dataset artifact path, partition, and timestamp helpers."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataSpec,
)
from quant_research_platform.contracts.validation import strict_timestamp_series

_YEAR_PARTITION_PATTERN = re.compile("^year=([0-9]{4})$")


def partition_timestamp_series(
    frame: pd.DataFrame, *, timestamp_column: str
) -> pd.Series:
    if timestamp_column not in frame:
        raise ValueError("missing_partition_timestamp_column")
    timestamp = strict_timestamp_series(
        frame[timestamp_column], error="invalid_partition_timestamp"
    )
    if timestamp.isna().any():
        raise ValueError("missing_partition_timestamp_column")
    return timestamp


def partition_frames(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None,
    partition_columns: tuple[str, ...],
) -> Iterator[tuple[str, pd.DataFrame]]:
    if frame.empty:
        return
    if partition_columns == ():
        yield ("", frame)
        return
    if partition_columns != ("year",):
        raise ValueError("unsupported_partitioning")
    if timestamp_column is None:
        raise ValueError("missing_partition_timestamp_column")
    timestamp = partition_timestamp_series(frame, timestamp_column=timestamp_column)
    min_timestamp = timestamp.min()
    max_timestamp = timestamp.max()
    timezone = timestamp.dt.tz
    for year in range(min_timestamp.year, max_timestamp.year + 1):
        start = pd.Timestamp(year=year, month=1, day=1, tz=timezone)
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz=timezone)
        mask = (timestamp >= start) & (timestamp < end)
        if bool(mask.any()):
            yield (format_year_partition_label(int(year)), frame.loc[mask])


def parse_year_partition_label(label: object) -> int:
    if type(label) is not str:
        raise ValueError("invalid_dataset_part_files")
    match = _YEAR_PARTITION_PATTERN.fullmatch(label)
    if match is None:
        raise ValueError("invalid_dataset_part_files")
    year = int(match.group(1))
    if year < date.min.year or year > date.max.year:
        raise ValueError("invalid_dataset_part_files")
    return year


def format_year_partition_label(year: object) -> str:
    if type(year) is not int or year < date.min.year or year > date.max.year:
        raise ValueError("invalid_dataset_partition_year")
    return f"year={year:04d}"


def partition_part_path(root: Path, partition: str, part_index: int) -> Path:
    if partition:
        return root / partition / f"part-{part_index:03d}.parquet"
    return root / f"part-{part_index:03d}.parquet"


def timestamp_primary_key_column_from_spec(spec: DataSpec) -> str | None:
    timestamp_columns = [
        column
        for column in spec.primary_key
        if spec.column_roles.get(column) is ColumnRole.TIMESTAMP
    ]
    return timestamp_columns[0] if timestamp_columns else None


def timestamp_column_from_spec_payload(spec: Mapping[str, Any]) -> str | None:
    roles = spec.get("column_roles", {})
    if not isinstance(roles, Mapping):
        raise ValueError("invalid_column_roles")
    primary_key = tuple(str(column) for column in spec.get("primary_key", ()))
    timestamp_columns = [
        column
        for column in primary_key
        if str(roles.get(column)) == ColumnRole.TIMESTAMP.value
    ]
    if len(timestamp_columns) > 1:
        raise ValueError("invalid_timestamp_primary_key")
    return timestamp_columns[0] if timestamp_columns else None


def relative_partition_label(data_root: Path, path: Path) -> str:
    relative_parent = path.parent.relative_to(data_root)
    label = relative_parent.as_posix()
    return "unpartitioned" if label == "." else label
