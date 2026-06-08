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
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage.datasets import (
    DatasetAccessMode,
    read_dataset,
    write_dataset_artifact,
)
from quant_research_platform.storage.layout import processed_data_parquet_root
from quant_research_platform.storage.parquet import write_processed_data

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
    data_root = write_processed_data(
        universe,
        layout=layout_from_processed_root(processed_root),
        catalog=catalog,
        processed_data_id="a_share_default",
    )
    return data_root, len(universe)


def write_universe_from_panel_dataset(
    processed_root: Path,
    *,
    start_year: int,
    end_year: int,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> tuple[Path, int]:
    catalog = a_share_processed_catalog()
    spec = catalog.get("a_share_default")
    panel_root = processed_data_parquet_root(processed_root, "panels", "daily_panel")
    row_count = 0

    def _chunks():
        nonlocal row_count
        frames = read_dataset(
            panel_root,
            years=tuple(range(start_year, end_year + 1)),
            columns=list(UNIVERSE_COLUMNS),
            access_mode=DatasetAccessMode.STREAMING,
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

    data_root = write_dataset_artifact(
        _chunks(),
        Path(processed_root) / "universes" / "a_share_default",
        catalog=catalog,
        data_id="a_share_default",
        access_mode=DatasetAccessMode.STREAMING,
        resource_policy=resource_policy,
    )
    return data_root, row_count


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
