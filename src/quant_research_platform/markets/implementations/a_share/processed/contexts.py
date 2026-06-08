"""Build A-share context processed data from panels."""

from __future__ import annotations

from pathlib import Path

from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.processed.io import (
    layout_from_processed_root,
)
from quant_research_platform.storage.datasets import (
    DatasetReadMode,
    inspect_dataset,
    read_dataset,
    write_dataset,
)

from ._dataset_helpers import (
    filter_years_sorted,
)

CONTEXT_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
    "market_type",
    "is_st",
    "is_pt",
    "limit_status",
    "listing_age_trading_days",
    "csrc_2012_industry_code",
    "csrc_2012_industry_name",
    "total_market_cap",
    "float_market_cap",
    "amount",
    "turnover",
)


def write_contexts_from_panel_dataset(
    processed_root: Path,
    *,
    start_year: int,
    end_year: int,
) -> tuple[Path, int]:
    catalog = a_share_processed_catalog()
    spec = catalog.get("daily_context")
    panel_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        "panels",
        "daily_panel",
    )
    available = set(inspect_dataset(panel_root).columns)
    columns = [column for column in CONTEXT_COLUMNS if column in available]
    row_count = 0

    def _chunks():
        nonlocal row_count
        frames = read_dataset(
            panel_root,
            partition_filter={"year": tuple(range(start_year, end_year + 1))},
            columns=columns,
            mode=DatasetReadMode.CHUNKED,
        )
        for frame in frames:
            context = filter_years_sorted(
                frame,
                start_year=start_year,
                end_year=end_year,
            )
            context = context[list(spec.required_columns)].copy()
            if context.empty:
                continue
            row_count += len(context)
            yield context

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
    "CONTEXT_COLUMNS",
    "write_contexts_from_panel_dataset",
]
