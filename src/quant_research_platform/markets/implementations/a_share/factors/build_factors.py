"""从 processed research panels 构建全样本 slim factor tables。

本脚本物化 evaluation 所需的 handoff artifact：
一个包含 keys、filters、controls、factor columns 和 forward labels 的紧凑 Parquet。
较重的 panel-only market fields 不进入 slim factor table。
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from quant_research_platform.runtime.paths import (  # noqa: E402
    factor_full_exact_path,
    panel_full_path,
)

from ..csmar.columns import (  # noqa: E402
    FACTOR_COLUMN_PREFIXES,
    SLIM_FACTOR_BASE_COLUMNS,
)
from .labels import (
    add_forward_return_labels,  # noqa: E402
)
from .price_volume import (  # noqa: E402
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
    """读取一个 panel 文件并保留六位 stock codes。"""

    result = pd.read_parquet(path, columns=columns, engine="pyarrow")
    if "stock_code" in result.columns:
        result["stock_code"] = result["stock_code"].astype(str).str.zfill(6)
    return result


def select_existing_columns(path: Path, wanted: tuple[str, ...]) -> list[str]:
    """返回 panel 文件中实际存在的 requested columns。"""

    wanted_unique = list(dict.fromkeys(wanted))
    columns = pq.ParquetFile(path).schema.names
    return [column for column in wanted_unique if column in columns]


def select_slim_factor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """只保留 keys、filters、common controls、factors 和 labels。"""

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
    """向 panel 添加第一批 price-volume factors 和 forward labels。

    先添加 labels，再计算 rolling factors；两类操作都按 stock 本地顺序和日期排序。
    后续 slim-column projection 只保留 validation 和 evaluation layer 需要的 columns。

    """

    labeled = add_forward_return_labels(panel, horizons=horizons)
    return calculate_price_volume_factors(labeled, windows=windows)


def write_parquet(path: Path, frame: pd.DataFrame) -> Path:
    """使用项目 processed-data 默认参数写出压缩 Parquet。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


def read_full_panel_for_factor_build(
    *,
    processed_root: Path,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """读取 factor construction 所需的全样本 panel。"""

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
    """计算并写出一个全样本 slim factor Parquet。

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
