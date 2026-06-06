"""Build split A-share price-volume factor processed data from panels."""

# ruff: noqa: E402

from __future__ import annotations

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
    price_volume_factor_columns_for_windows,
)
from quant_research_platform.storage.layout import processed_data_parquet_root
from quant_research_platform.storage.parquet import (
    parquet_dataset_columns,
    read_partitioned_parquet_dataset,
    write_processed_data,
)

DEFAULT_PRICE_VOLUME_WINDOWS: tuple[int, ...] = (20, 60, 120)


def read_panel(
    path: Path,
    columns: tuple[str, ...] | list[str] | None = None,
    years: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Read one panel dataset for factor construction."""

    return read_partitioned_parquet_dataset(path, years=years, columns=columns)


def select_slim_factor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep canonical primary key and factor columns only."""

    spec = a_share_processed_catalog().get(PRICE_VOLUME_PROCESSED_DATA_ID)
    missing = [
        column for column in spec.required_columns if column not in frame.columns
    ]
    if missing:
        raise ValueError("missing_factor_output_columns: " + ", ".join(missing))
    return frame[list(spec.required_columns)].copy()


def calculate_panel_factors(
    panel: pd.DataFrame,
    *,
    windows: tuple[int, ...] = DEFAULT_PRICE_VOLUME_WINDOWS,
) -> pd.DataFrame:
    """Add price-volume factors without forward labels."""

    return calculate_price_volume_factors(
        panel,
        factor_columns=price_volume_factor_columns_for_windows(windows),
    )


def read_full_panel_for_factor_build(
    *,
    processed_root: Path,
    years: tuple[int, ...],
) -> pd.DataFrame:
    """读取 factor construction 所需的全样本 panel。"""

    data_root = processed_data_parquet_root(
        processed_root, "panels", DAILY_PANEL_PROCESSED_DATA_ID
    )
    read_columns = _factor_panel_input_columns()
    available_columns = set(parquet_dataset_columns(data_root))
    missing = [column for column in read_columns if column not in available_columns]
    if missing:
        raise ValueError(
            "missing_factor_panel_input_columns: " + ", ".join(missing)
        )
    return read_panel(
        data_root,
        columns=read_columns,
        years=years,
    )


def calculate_price_volume_factor_full(
    *,
    interim_root: Path,
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> tuple[Path, int]:
    """Calculate and write split price-volume factor processed data."""

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
    input_years = _available_panel_years(
        processed_root=processed_root,
        start_year=warmup_start_date.year,
        end_year=end_year,
    )
    panel = read_full_panel_for_factor_build(
        processed_root=processed_root,
        years=input_years,
    )
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.loc[
        panel["timestamp"].between(
            warmup_start_date,
            requested_end_date,
            inclusive="both",
        )
    ].copy()
    missing_key = [
        column for column in spec.primary_key if column not in panel.columns
    ]
    if missing_key:
        raise ValueError(
            "factor_input_missing_canonical_key: " + ", ".join(missing_key)
        )
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
    output = select_slim_factor_columns(factored.loc[mask].copy())
    output = output.sort_values(list(spec.primary_key)).reset_index(drop=True)
    data_root = write_processed_data(
        output,
        layout=layout_from_processed_root(processed_root),
        catalog=catalog,
        processed_data_id=PRICE_VOLUME_PROCESSED_DATA_ID,
    )
    return data_root, len(output)


def _read_trading_calendar_interim(interim_root: Path) -> pd.DataFrame:
    try:
        return read_partitioned_parquet_dataset(
            Path(interim_root) / "trading_calendar" / "data"
        )
    except FileNotFoundError as exc:
        raise ValueError("missing_trading_calendar_interim") from exc


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
    data_root = processed_data_parquet_root(
        processed_root, "panels", DAILY_PANEL_PROCESSED_DATA_ID
    )
    missing = [
        year
        for year in range(start_year, end_year + 1)
        if not list((data_root / f"year={year}").glob("*.parquet"))
    ]
    if missing:
        raise ValueError(
            "missing_requested_processed_input: "
            + ", ".join(str(year) for year in missing)
        )


def _available_panel_years(
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> tuple[int, ...]:
    data_root = processed_data_parquet_root(
        processed_root, "panels", DAILY_PANEL_PROCESSED_DATA_ID
    )
    years = tuple(
        year
        for year in range(start_year, end_year + 1)
        if list((data_root / f"year={year}").glob("*.parquet"))
    )
    if not years:
        raise ValueError("missing_requested_processed_input")
    return years


def _panel_part_paths(
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> list[Path]:
    data_root = processed_data_parquet_root(
        processed_root, "panels", DAILY_PANEL_PROCESSED_DATA_ID
    )
    paths = [
        data_root / f"year={year}" / "part-000.parquet"
        for year in range(start_year, end_year + 1)
        if (data_root / f"year={year}" / "part-000.parquet").exists()
    ]
    if paths:
        return paths
    raise FileNotFoundError(data_root)


def _factor_panel_input_columns() -> tuple[str, ...]:
    return a_share_price_volume_panel_input_columns()
