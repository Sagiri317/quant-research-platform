"""Build slim full-sample factor tables from processed research panels.

This script materializes the child-project handoff artifact for evaluation:
one compact Parquet with keys, filters, controls, factor columns, and forward
labels. Heavy panel-only market fields stay out of the slim factor table.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from csmar.columns import FACTOR_COLUMN_PREFIXES, SLIM_FACTOR_BASE_COLUMNS  # noqa: E402
from runtime.paths import factor_full_exact_path, panel_full_path  # noqa: E402

from factors.labels import add_forward_return_labels  # noqa: E402
from factors.price_volume import (  # noqa: E402
    DEFAULT_PRICE_VOLUME_WINDOWS,
    calculate_price_volume_factors,
)

FACTOR_INPUT_COLUMNS = (
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


def read_panel(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read one panel file and preserve six-character stock codes."""

    result = pd.read_parquet(path, columns=columns, engine="pyarrow")
    if "stock_code" in result.columns:
        result["stock_code"] = result["stock_code"].astype(str).str.zfill(6)
    return result


def select_existing_columns(path: Path, wanted: tuple[str, ...]) -> list[str]:
    """Return requested columns that are present in a panel file."""

    wanted_unique = list(dict.fromkeys(wanted))
    columns = pq.ParquetFile(path).schema.names
    return [column for column in wanted_unique if column in columns]


def select_slim_factor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only keys, filters, common controls, factors, and labels."""

    base_columns = [
        column for column in SLIM_FACTOR_BASE_COLUMNS if column in frame.columns
    ]
    factor_columns = [
        column
        for column in frame.columns
        if column not in base_columns and column.startswith(FACTOR_COLUMN_PREFIXES)
    ]
    return frame[base_columns + factor_columns].copy()


def calculate_panel_factors(
    panel: pd.DataFrame,
    *,
    windows: tuple[int, ...] = DEFAULT_PRICE_VOLUME_WINDOWS,
    horizons: tuple[int, ...] = (1, 5, 10, 20, 60, 120),
) -> pd.DataFrame:
    """Add first-batch price-volume factors and forward labels to a panel.

    Labels are added before rolling factor calculation because both operations
    are stock-local and date-sorted. The subsequent slim-column projection keeps
    only columns needed by validation and core evaluation.
    """

    labeled = add_forward_return_labels(panel, horizons=horizons)
    return calculate_price_volume_factors(labeled, windows=windows)


def write_parquet(path: Path, frame: pd.DataFrame) -> Path:
    """Write compressed Parquet using the project processed-data default."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


def read_full_panel_for_factor_build(
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Read the full-sample panel needed for factor construction."""

    path = panel_full_path(processed_root, start_year, end_year)
    if not path.exists():
        raise FileNotFoundError(path)
    read_columns = select_existing_columns(
        path,
        FACTOR_INPUT_COLUMNS + SLIM_FACTOR_BASE_COLUMNS,
    )
    return read_panel(path, columns=read_columns)


def calculate_price_volume_factor_full(
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
    windows: tuple[int, ...],
    horizons: tuple[int, ...],
) -> tuple[Path, int]:
    """Calculate and write one full-sample slim factor Parquet.

    Factor calculation runs on the full requested panel before the final year
    mask, so rolling windows and forward labels are internally consistent at
    range boundaries when a wider input is used in the future.
    """

    panel = read_full_panel_for_factor_build(
        processed_root=processed_root,
        start_year=start_year,
        end_year=end_year,
    )
    factored = calculate_panel_factors(
        panel,
        windows=windows,
        horizons=horizons,
    )
    factored["trade_date"] = pd.to_datetime(factored["trade_date"])
    mask = factored["trade_date"].dt.year.between(start_year, end_year)
    output = select_slim_factor_columns(factored.loc[mask].copy())
    output = output.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    output_path = factor_full_exact_path(processed_root, start_year, end_year)
    write_parquet(output_path, output)
    return output_path, len(output)


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calculate one full-sample factor table."
    )
    parser.add_argument("--processed-root", default="data/processed_private/csmar")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--factor-set", default="price_volume")
    parser.add_argument("--windows", default="20,60,120")
    parser.add_argument("--horizons", default="1,5,10,20,60,120")
    args = parser.parse_args(argv)

    if args.factor_set != "price_volume":
        print(f"ERROR: unsupported factor set: {args.factor_set}")
        return 2
    output_path, rows = calculate_price_volume_factor_full(
        processed_root=Path(args.processed_root),
        start_year=args.start_year,
        end_year=args.end_year,
        windows=_parse_int_tuple(args.windows),
        horizons=_parse_int_tuple(args.horizons),
    )
    print(f"wrote {output_path} rows={rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
