"""Internal dataset helpers shared by A-share processed builders."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    _dataset_planning_metadata,
    read_dataset,
)


def year_partition_filter(
    years: tuple[int, ...] | None,
) -> dict[str, tuple[int, ...]] | None:
    if years is None:
        return None
    return {"year": years}


def available_dataset_years(
    data_root: Path,
    *,
    start_year: int,
    end_year: int,
    error_code: str = "missing_requested_processed_input",
) -> tuple[int, ...]:
    years = tuple(
        year
        for year in range(start_year, end_year + 1)
        if dataset_partition_exists(data_root, year)
    )
    if not years:
        raise ValueError(error_code)
    return years


def dataset_partition_exists(data_root: Path, year: int) -> bool:
    try:
        _dataset_planning_metadata(
            data_root,
            partition_filter={"year": (year,)},
        )
        return True
    except ValueError as exc:
        if str(exc) == "partition_value_not_found":
            return False
        raise


def dataset_row_count(data_root: Path, *, years: tuple[int, ...]) -> int:
    return _dataset_planning_metadata(
        data_root,
        partition_filter=year_partition_filter(years),
    ).row_count


def dataset_asset_ids(
    data_root: Path,
    *,
    years: tuple[int, ...],
    error_code: str = "missing_requested_processed_input",
) -> tuple[str, ...]:
    asset_ids: set[str] = set()
    chunks = read_dataset(
        data_root,
        partition_filter=year_partition_filter(years),
        columns=["asset_id"],
        mode=DatasetReadMode.CHUNKED,
    )
    for chunk in chunks:
        asset_ids.update(str(value) for value in chunk["asset_id"].dropna().unique())
    if not asset_ids:
        raise ValueError(error_code)
    return tuple(sorted(asset_ids))


def read_dataset_asset_chunk(
    data_root: Path,
    *,
    years: tuple[int, ...],
    columns: tuple[str, ...],
    asset_ids: tuple[str, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    asset_id_set = set(asset_ids)
    chunks = read_dataset(
        data_root,
        partition_filter=year_partition_filter(years),
        columns=list(columns),
        mode=DatasetReadMode.CHUNKED,
    )
    for chunk in chunks:
        filtered = chunk.loc[chunk["asset_id"].isin(asset_id_set)]
        if not filtered.empty:
            frames.append(filtered)
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["market_id", "asset_id", "timestamp"],
        kind="mergesort",
    )


def filter_years_sorted(
    frame: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    sort_columns: tuple[str, ...] = ("market_id", "asset_id", "timestamp"),
) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result = result.loc[result["timestamp"].dt.year.between(start_year, end_year)]
    return result.sort_values(list(sort_columns)).reset_index(drop=True)


def chunked_values(values: tuple[str, ...], size: int) -> Iterator[tuple[str, ...]]:
    for index in range(0, len(values), max(1, size)):
        yield values[index : index + max(1, size)]
