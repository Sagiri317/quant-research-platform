"""Build split A-share price-volume factor processed data from panels."""

# ruff: noqa: E402

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.markets.implementations.a_share.calendar import (
    warmup_start_date_for_window,
)
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_price_volume_panel_input_columns,
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    DAILY_PANEL_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.a_share.processed.io import (
    layout_from_processed_root,
)
from quant_research_platform.research.factors.price_volume import (
    calculate_price_volume_factors,
    largest_price_volume_lookback_days,
)
from quant_research_platform.runtime.resources import (
    ChunkAxis,
    ChunkPlanner,
    RuntimeResourcePolicy,
)
from quant_research_platform.storage.datasets import (
    inspect_dataset,
    read_dataset,
    write_dataset,
)

from ..processed._dataset_helpers import (
    available_dataset_years,
    chunked_values,
    dataset_asset_ids,
    dataset_partition_exists,
    dataset_row_count,
    read_dataset_asset_chunk,
)


def calculate_price_volume_factor_asset_batches(
    *,
    interim_root: Path,
    processed_root: Path,
    start_year: int,
    end_year: int,
    asset_chunk_size: int | None = None,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> tuple[Path, int]:
    """Calculate price-volume factors in asset batches with warmup context."""

    catalog = a_share_processed_catalog()
    spec = catalog.get(PRICE_VOLUME_PROCESSED_DATA_ID)
    factor_columns = _factor_output_columns()
    max_window = _infer_factor_warmup_window(factor_columns)
    trading_calendar = _read_trading_calendar_interim(interim_root)
    requested_start_date = pd.Timestamp(year=start_year, month=1, day=1)
    requested_end_date = pd.Timestamp(year=end_year, month=12, day=31)
    warmup_start_date = warmup_start_date_for_window(
        trading_calendar,
        start_date=requested_start_date,
        max_window=max_window,
    )
    _assert_requested_panel_partitions_present(
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
    )
    data_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        "panels",
        DAILY_PANEL_PROCESSED_DATA_ID,
    )
    panel_input_columns = a_share_price_volume_panel_input_columns()
    input_years = available_dataset_years(
        data_root,
        start_year=warmup_start_date.year,
        end_year=end_year,
    )
    available_columns = set(inspect_dataset(data_root).columns)
    missing = [
        column
        for column in panel_input_columns
        if column not in available_columns
    ]
    if missing:
        raise ValueError("missing_factor_panel_input_columns: " + ", ".join(missing))
    asset_ids = dataset_asset_ids(data_root, years=input_years)
    chunk_size = asset_chunk_size or _planned_asset_chunk_size(
        asset_ids=asset_ids,
        data_root=data_root,
        years=input_years,
        read_columns=panel_input_columns,
        resource_policy=resource_policy,
    )
    row_count = 0

    def _output_chunks() -> Iterator[pd.DataFrame]:
        nonlocal row_count
        for asset_chunk in chunked_values(asset_ids, chunk_size):
            panel = read_dataset_asset_chunk(
                data_root,
                years=input_years,
                columns=panel_input_columns,
                asset_ids=asset_chunk,
            )
            if panel.empty:
                continue
            panel["timestamp"] = pd.to_datetime(panel["timestamp"])
            panel = panel.loc[
                panel["timestamp"].between(
                    warmup_start_date,
                    requested_end_date,
                    inclusive="both",
                )
            ].copy()
            factored = calculate_price_volume_factors(
                panel,
                factor_columns=factor_columns,
            )
            factored["timestamp"] = pd.to_datetime(factored["timestamp"])
            mask = factored["timestamp"].between(
                requested_start_date,
                requested_end_date,
                inclusive="both",
            )
            output_columns = list(spec.required_columns)
            missing = [
                column for column in output_columns if column not in factored.columns
            ]
            if missing:
                raise ValueError(
                    "missing_factor_output_columns: " + ", ".join(missing)
                )
            output = factored.loc[mask, output_columns].copy()
            output = output.sort_values(list(spec.primary_key)).reset_index(drop=True)
            if output.empty:
                continue
            row_count += len(output)
            yield output

    artifact_root = layout_from_processed_root(processed_root).processed_data_root(
        spec.market_id,
        spec.category,
        spec.processed_data_id,
    )
    write_dataset(
        artifact_root,
        _output_chunks(),
        spec=spec,
    )
    return artifact_root, row_count


def _read_trading_calendar_interim(interim_root: Path) -> pd.DataFrame:
    try:
        return read_dataset(Path(interim_root) / "trading_calendar")
    except (FileNotFoundError, ValueError) as exc:
        if _is_missing_dataset_artifact_error(exc):
            raise ValueError("missing_trading_calendar_interim") from exc
        raise


def _is_missing_dataset_artifact_error(exc: Exception) -> bool:
    message = str(exc)
    return isinstance(exc, FileNotFoundError) or message in {
        "dataset_spec_missing",
        "dataset_data_dir_missing",
    }


def _factor_output_columns() -> tuple[str, ...]:
    spec = a_share_processed_catalog().get(PRICE_VOLUME_PROCESSED_DATA_ID)
    return tuple(
        column
        for column in spec.required_columns
        if spec.column_roles[column] is ColumnRole.FACTOR
    )


def _infer_factor_warmup_window(factor_columns: tuple[str, ...]) -> int:
    if not factor_columns:
        raise ValueError("unable_to_infer_factor_warmup_window")
    max_window = largest_price_volume_lookback_days(factor_columns)
    if max_window < 0:
        raise ValueError("invalid_factor_warmup_window")
    return max_window


def _assert_requested_panel_partitions_present(
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> None:
    data_root = layout_from_processed_root(processed_root).processed_data_root(
        a_share_processed_catalog().get(DAILY_PANEL_PROCESSED_DATA_ID).market_id,
        "panels",
        DAILY_PANEL_PROCESSED_DATA_ID,
    )
    missing = [
        year
        for year in range(start_year, end_year + 1)
        if not dataset_partition_exists(data_root, year)
    ]
    if missing:
        raise ValueError(
            "missing_requested_processed_input: "
            + ", ".join(str(year) for year in missing)
        )


def _planned_asset_chunk_size(
    *,
    asset_ids: tuple[str, ...],
    data_root: Path,
    years: tuple[int, ...],
    read_columns: tuple[str, ...],
    resource_policy: RuntimeResourcePolicy | None,
) -> int:
    total_rows = dataset_row_count(data_root, years=years)
    rows_per_asset = max(1, total_rows // max(1, len(asset_ids)))
    output_width = len(
        a_share_processed_catalog().get(PRICE_VOLUME_PROCESSED_DATA_ID).required_columns
    )
    working_width = len(read_columns) + (output_width * 2)
    estimated_bytes_per_asset = max(1, rows_per_asset * working_width * 16)
    plan = ChunkPlanner(resource_policy).plan(
        axis=ChunkAxis.ASSET,
        item_count=len(asset_ids),
        estimated_bytes_per_item=estimated_bytes_per_asset,
        lookback_window=largest_price_volume_lookback_days(_factor_output_columns()),
        output_width=output_width,
        disjoint=True,
    )
    return plan.chunk_size
