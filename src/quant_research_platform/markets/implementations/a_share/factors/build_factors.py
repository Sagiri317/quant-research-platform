"""Build split A-share price-volume factor processed data from panels."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.factors.literature_price_volume import (  # noqa: E501
    DEFAULT_LITERATURE_WINDOWS,
    calculate_literature_price_volume_factors,
)
from quant_research_platform.markets.implementations.a_share.processed.io import (
    layout_from_processed_root,
)
from quant_research_platform.storage.layout import processed_data_parquet_root
from quant_research_platform.storage.parquet import (
    parquet_dataset_columns,
    read_partitioned_parquet_dataset,
    write_processed_data,
)

from .price_volume import (  # noqa: E402
    DEFAULT_PRICE_VOLUME_WINDOWS,
    calculate_price_volume_factors,
)

FACTOR_INPUT_COLUMNS = (
    "market_id",
    "asset_id",
    "timestamp",
    "session_date",
    "available_at",
    "stock_code",
    "trade_date",
    "adj_open",
    "adj_close",
    "return_1d",
    "amount",
    "turnover",
    "high",
    "low",
    "prev_close",
)


def read_panel(
    path: Path,
    columns: tuple[str, ...] | list[str] | None = None,
    years: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """读取一个 panel 文件并保留六位 stock codes。"""

    result = read_partitioned_parquet_dataset(path, years=years, columns=columns)
    if "stock_code" in result.columns:
        result["stock_code"] = result["stock_code"].astype(str).str.zfill(6)
    return result


def select_existing_columns(path: Path, wanted: tuple[str, ...]) -> list[str]:
    """返回 panel 文件中实际存在的 requested columns。"""

    wanted_unique = list(dict.fromkeys(wanted))
    columns = parquet_dataset_columns(path)
    return [column for column in wanted_unique if column in columns]


def select_slim_factor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep keys, filters, common controls, and factor columns only."""

    spec = a_share_processed_catalog().get("price_volume_56")
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

    return calculate_price_volume_factors(panel, windows=windows)


def read_full_panel_for_factor_build(
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """读取 factor construction 所需的全样本 panel。"""

    data_root = processed_data_parquet_root(processed_root, "panels", "daily_panel")
    read_columns = select_existing_columns(
        data_root,
        _factor_panel_input_columns(),
    )
    return read_panel(
        data_root,
        columns=read_columns,
        years=tuple(range(start_year, end_year + 1)),
    )


def calculate_price_volume_factor_full(
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
    windows: tuple[int, ...] = DEFAULT_LITERATURE_WINDOWS,
) -> tuple[Path, int]:
    """Calculate and write split price-volume factor processed data."""

    panel = read_full_panel_for_factor_build(
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
    )
    catalog = a_share_processed_catalog()
    spec = catalog.get("price_volume_56")
    missing_key = [
        column for column in spec.primary_key if column not in panel.columns
    ]
    if missing_key:
        raise ValueError(
            "factor_input_missing_canonical_key: " + ", ".join(missing_key)
        )
    factored = calculate_literature_price_volume_factors(
        panel,
        windows=windows,
    )
    factored["trade_date"] = pd.to_datetime(factored["trade_date"])
    mask = factored["trade_date"].dt.year.between(start_year, end_year)
    output = select_slim_factor_columns(factored.loc[mask].copy())
    output = output.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    data_root = write_processed_data(
        output,
        layout=layout_from_processed_root(processed_root),
        catalog=catalog,
        processed_data_id="price_volume_56",
    )
    return data_root, len(output)


def _panel_part_paths(
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> list[Path]:
    data_root = processed_data_parquet_root(processed_root, "panels", "daily_panel")
    paths = [
        data_root / f"year={year}" / "part-000.parquet"
        for year in range(start_year, end_year + 1)
        if (data_root / f"year={year}" / "part-000.parquet").exists()
    ]
    if paths:
        return paths
    legacy_path = (
        data_root / f"year={start_year}_{end_year}" / "part-000.parquet"
    )
    if legacy_path.exists():
        return [legacy_path]
    raise FileNotFoundError(data_root)


def _factor_panel_input_columns() -> tuple[str, ...]:
    spec = a_share_processed_catalog().get("price_volume_56")
    control_columns = tuple(
        column
        for column in spec.required_columns
        if spec.column_roles[column] is not ColumnRole.FACTOR
    )
    return tuple(dict.fromkeys((*FACTOR_INPUT_COLUMNS, *control_columns)))


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calculate one full-sample factor table."
    )
    parser.add_argument("--processed-root", default="data/processed/a_share")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--factor-set", default="price_volume")
    parser.add_argument("--windows", default="1,5,10,20,60,120,252")
    args = parser.parse_args(argv)

    if args.factor_set != "price_volume":
        print(f"ERROR: unsupported factor set: {args.factor_set}")
        return 2
    output_path, rows = calculate_price_volume_factor_full(
        processed_root=Path(args.processed_root),
        start_year=args.start_year,
        end_year=args.end_year,
        windows=_parse_int_tuple(args.windows),
    )
    print(f"wrote price_volume_56 rows={rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
