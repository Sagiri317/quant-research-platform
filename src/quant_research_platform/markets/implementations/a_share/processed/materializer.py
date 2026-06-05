"""A-share processed layer materialization orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_research_platform.storage.layout import processed_data_parquet_root
from quant_research_platform.storage.parquet import read_partitioned_parquet_dataset

from ..factors.build_factors import calculate_price_volume_factor_full
from ..ids import PRICE_VOLUME_PROCESSED_DATA_ID
from ..labels.build_labels import calculate_panel_labels, write_forward_labels
from ..masks.build import write_masks
from ..universes.build import write_universe
from .contexts import write_contexts
from .panels import build_panel_frame_from_interim, build_panel_from_interim


@dataclass(frozen=True)
class AShareProcessedBuildResult:
    """Aggregate metadata for a processed-layer build run."""

    row_counts: dict[str, int]
    data_roots: dict[str, Path]


def build_a_share_processed_layers(
    *,
    interim_root: Path,
    processed_root: Path,
    start_year: int,
    end_year: int,
    end_date: str,
    label_calculation_end_year: int | None = None,
    min_listing_trading_days: int = 120,
) -> AShareProcessedBuildResult:
    """Build panel, contexts, universes, masks, factors, and labels.

    This orchestrator owns sequencing only. Persisted outputs remain owned by
    spec-driven storage writers called inside each layer builder.
    """

    panel_root, panel_rows = build_panel_from_interim(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
        end_date=end_date,
        calculation_end_year=label_calculation_end_year,
        min_listing_trading_days=min_listing_trading_days,
    )
    panel = _read_processed_panel(
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
    )
    context_root, context_rows = write_contexts(
        panel,
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
    )
    universe_root, universe_rows = write_universe(
        panel,
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
    )
    mask_outputs = write_masks(
        panel,
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
    )
    factor_root, factor_rows = calculate_price_volume_factor_full(
        interim_root=interim_root,
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
    )
    label_panel = build_panel_frame_from_interim(
        interim_root=interim_root,
        start_year=start_year,
        end_year=label_calculation_end_year or end_year,
        min_listing_trading_days=min_listing_trading_days,
    )
    label_root, label_rows = write_forward_labels(
        calculate_panel_labels(label_panel),
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
    )
    data_roots = {
        "daily_panel": panel_root,
        "daily_context": context_root,
        "a_share_default": universe_root,
        PRICE_VOLUME_PROCESSED_DATA_ID: factor_root,
        "forward_returns": label_root,
    }
    row_counts = {
        "daily_panel": panel_rows,
        "daily_context": context_rows,
        "a_share_default": universe_rows,
        PRICE_VOLUME_PROCESSED_DATA_ID: factor_rows,
        "forward_returns": label_rows,
    }
    for mask_kind, (path, rows) in mask_outputs.items():
        data_roots[f"mask_{mask_kind}"] = path
        row_counts[f"mask_{mask_kind}"] = rows
    return AShareProcessedBuildResult(row_counts=row_counts, data_roots=data_roots)


def _read_processed_panel(
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    data_root = processed_data_parquet_root(processed_root, "panels", "daily_panel")
    return read_partitioned_parquet_dataset(
        data_root,
        years=tuple(range(start_year, end_year + 1)),
    )


__all__ = ["AShareProcessedBuildResult", "build_a_share_processed_layers"]
