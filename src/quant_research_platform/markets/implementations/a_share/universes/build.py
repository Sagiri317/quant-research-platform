"""Build A-share universe processed data from panels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.processed.io import (
    layout_from_processed_root,
)
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    read_dataset,
    write_dataset,
)

UNIVERSE_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
    "eligible",
)


def select_universe_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in UNIVERSE_COLUMNS if column in frame.columns]
    return frame[columns].copy()


def write_universe(
    frame: pd.DataFrame,
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> tuple[Path, int]:
    universe = _filter_years(select_universe_columns(frame), start_year, end_year)
    catalog = a_share_processed_catalog()
    spec = catalog.get("a_share_default")
    universe = universe[list(spec.required_columns)].copy()
    artifact_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        spec.category,
        spec.processed_data_id,
    )
    write_dataset(artifact_root, universe, spec=spec)
    return artifact_root, len(universe)


def write_universe_from_panel_dataset(
    processed_root: Path,
    *,
    start_year: int,
    end_year: int,
) -> tuple[Path, int]:
    catalog = a_share_processed_catalog()
    spec = catalog.get("a_share_default")
    panel_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        "panels",
        "daily_panel",
    )
    row_count = 0

    def _chunks():
        nonlocal row_count
        frames = read_dataset(
            panel_root,
            partition_filter={"year": tuple(range(start_year, end_year + 1))},
            columns=list(UNIVERSE_COLUMNS),
            mode=DatasetReadMode.CHUNKED,
        )
        for frame in frames:
            universe = _filter_years(
                select_universe_columns(frame),
                start_year,
                end_year,
            )
            universe = universe[list(spec.required_columns)].copy()
            if universe.empty:
                continue
            row_count += len(universe)
            yield universe

    artifact_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        spec.category,
        spec.processed_data_id,
    )
    write_dataset(
        artifact_root,
        _chunks(),
        spec=spec,
    )
    return artifact_root, row_count


def _filter_years(frame: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result = result.loc[result["timestamp"].dt.year.between(start_year, end_year)]
    sort_columns = ["market_id", "asset_id", "timestamp"]
    return result.sort_values(sort_columns).reset_index(drop=True)


__all__ = [
    "UNIVERSE_COLUMNS",
    "select_universe_columns",
    "write_universe",
    "write_universe_from_panel_dataset",
]
