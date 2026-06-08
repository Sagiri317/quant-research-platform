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

from ..processed._dataset_helpers import (
    filter_years_sorted,
)

UNIVERSE_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
    "eligible",
)


def _universe_frame_from_panel(
    frame: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    catalog = a_share_processed_catalog()
    spec = catalog.get("a_share_default")
    columns = [column for column in UNIVERSE_COLUMNS if column in frame.columns]
    universe = filter_years_sorted(
        frame[columns].copy(),
        start_year=start_year,
        end_year=end_year,
    )
    return universe[list(spec.required_columns)].copy()


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
            universe = _universe_frame_from_panel(
                frame,
                start_year=start_year,
                end_year=end_year,
            )
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


__all__ = [
    "UNIVERSE_COLUMNS",
    "write_universe_from_panel_dataset",
]
