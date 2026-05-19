"""Private CSMAR raw-to-interim helpers.

This module is for local private files only. It never downloads data and does
not contain credentials or CSMAR account information.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from csmar.columns import (
    apply_interim_column_rules,
    load_or_create_column_overrides,
    write_column_pruning_report,
)
from csmar.contracts import (
    build_raw_export_metadata,
    default_metadata_path,
    map_csmar_columns,
    normalize_stock_code,
)

DAILY_A_SHARE_MARKET_TYPES = {1, 4, 16, 32}
DAILY_ST_STATUS_CODES = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15}
STRING_CODE_DTYPES = {"Stkcd": str, "Symbol": str, "Indexcd": str}


@dataclass(frozen=True)
class BenchmarkIndex:
    """Selected benchmark index metadata."""

    code: str
    name: str
    tier: str


BENCHMARK_INDEX_POOL: dict[str, BenchmarkIndex] = {
    "000985": BenchmarkIndex("000985", "中证全指指数", "core"),
    "000300": BenchmarkIndex("000300", "沪深300指数", "core"),
    "000905": BenchmarkIndex("000905", "中证小盘500指数", "core"),
    "000852": BenchmarkIndex("000852", "中证1000指数", "core"),
    "000001": BenchmarkIndex("000001", "上证综合指数", "core"),
    "399001": BenchmarkIndex("399001", "深证成份指数", "core"),
    "399006": BenchmarkIndex("399006", "创业板指数(价格)", "core"),
    "000688": BenchmarkIndex("000688", "上证科创板50成份指数", "core"),
    "932000": BenchmarkIndex("932000", "中证2000指数", "extended"),
    "000906": BenchmarkIndex("000906", "中证800指数", "extended"),
    "000510": BenchmarkIndex("000510", "中证A500指数", "extended"),
    "000016": BenchmarkIndex("000016", "上证50指数", "extended"),
    "399106": BenchmarkIndex("399106", "深证综合指数", "extended"),
    "399102": BenchmarkIndex("399102", "创业板综合指数", "extended"),
    "000698": BenchmarkIndex("000698", "上证科创板100指数", "extended"),
    "000984": BenchmarkIndex("000984", "沪深300等权重指数", "equal_weight"),
    "000982": BenchmarkIndex("000982", "中证500等权重指数", "equal_weight"),
    "000842": BenchmarkIndex("000842", "中证800等权重指数", "equal_weight"),
    "932382": BenchmarkIndex("932382", "中证1000等权重指数", "equal_weight"),
    "932548": BenchmarkIndex("932548", "中证A500等权重指数", "equal_weight"),
}


def write_raw_export_metadata(path: str | Path) -> Path:
    """Write one sidecar ``.metadata.json`` next to a raw CSMAR export."""

    export_path = Path(path)
    metadata_path = default_metadata_path(export_path)
    payload = build_raw_export_metadata(export_path)
    metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def normalize_private_export_frame(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Normalize one raw CSMAR frame into the first interim schema."""

    mapped = map_csmar_columns(frame, dataset)
    if "stock_code" in mapped.columns:
        mapped["stock_code"] = normalize_stock_code(mapped["stock_code"])
    if "index_code" in mapped.columns:
        mapped["index_code"] = mapped["index_code"].astype(str).str.strip()

    if dataset == "daily_market":
        result = mapped.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result["return_1d"] = result["return_with_dividend"]
        result["adj_close"] = result["comparable_close_with_dividend"]
        result["adj_open"] = result["open"] * result["adj_close"] / result["close"]
        result["float_market_cap"] = result["float_market_cap_thousand"] * 1000.0
        result["total_market_cap"] = result["total_market_cap_thousand"] * 1000.0
        result["is_limit_up"] = result["limit_status"] == 1
        result["is_limit_down"] = result["limit_status"] == -1
        result["is_trading"] = result["filling_flag"].astype(str).eq("0")
        result["is_st"] = result["trading_status_code"].isin(DAILY_ST_STATUS_CODES)
        result["is_pt"] = result["trading_status_code"].eq(16)
        result["is_a_share_scope"] = result["market_type"].isin(
            DAILY_A_SHARE_MARKET_TYPES
        )
        result = result[result["is_a_share_scope"]].copy()
        result["eligible_base"] = (
            result["is_a_share_scope"]
            & result["is_trading"]
            & ~result["is_st"]
            & ~result["is_pt"]
        )
        result["turnover"] = pd.NA
        return result.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)

    if dataset == "index_market":
        result = mapped.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result["return_1d"] = result["return_1d_pct"]
        return result.sort_values(["index_code", "trade_date"]).reset_index(drop=True)

    if dataset == "adjustment":
        result = mapped.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        return result.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)

    if dataset == "market_cap":
        result = mapped.copy()
        result["share_change_date"] = pd.to_datetime(result["share_change_date"])
        return result.sort_values(
            ["stock_code", "share_change_date"]
        ).reset_index(drop=True)

    if dataset == "stock_master":
        result = mapped.copy()
        for column in ("listing_date", "delisting_date", "status_change_date"):
            if column in result.columns:
                result[column] = pd.to_datetime(result[column], errors="coerce")
        return result.sort_values(["stock_code"]).reset_index(drop=True)

    if dataset == "index_info":
        return mapped.sort_values(["index_code"]).reset_index(drop=True)

    raise ValueError(f"Unsupported private CSMAR dataset: {dataset}")


def read_private_export_file(path: str | Path, dataset: str) -> pd.DataFrame:
    """Read one private CSMAR CSV, repairing known vendor CSV edge cases."""

    export_path = Path(path)
    if dataset != "index_info":
        return pd.read_csv(export_path, dtype=STRING_CODE_DTYPES)

    with export_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = []
        for row in reader:
            if len(row) > len(header):
                row = row[: len(header) - 1] + [",".join(row[len(header) - 1 :])]
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            rows.append(row)
    return pd.DataFrame(rows, columns=header)


def _date_column(dataset: str) -> str:
    if dataset in {"daily_market", "index_market", "adjustment"}:
        return "trade_date"
    if dataset == "market_cap":
        return "share_change_date"
    raise ValueError(f"Dataset has no filtering date column: {dataset}")


def _format_dates_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = result[column].dt.strftime("%Y-%m-%d")
    return result


def _write_frame(path: Path, frame: pd.DataFrame, *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = _format_dates_for_csv(frame)
    output.to_csv(
        path,
        mode="a" if append else "w",
        header=not append,
        index=False,
        encoding="utf-8-sig",
    )


def write_interim_yearly_files(
    *,
    dataset: str,
    input_files: list[Path],
    output_dir: Path,
    start_date: str,
    end_date: str,
    overrides: dict[str, dict[str, object]] | None = None,
    pruning_reports: list[dict[str, object]] | None = None,
    chunksize: int = 500_000,
) -> list[Path]:
    """Write date-filtered yearly interim CSV files for a time-series dataset."""

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    date_column = _date_column(dataset)
    written: set[Path] = set()
    initialized: set[Path] = set()

    for input_file in sorted(input_files):
        for chunk in pd.read_csv(
            input_file,
            chunksize=chunksize,
            dtype=STRING_CODE_DTYPES,
        ):
            normalized = normalize_private_export_frame(chunk, dataset)
            filtered = normalized[
                (normalized[date_column] >= start) & (normalized[date_column] <= end)
            ]
            if filtered.empty:
                continue
            filtered, report = apply_interim_column_rules(
                filtered, dataset, overrides=overrides
            )
            if pruning_reports is not None:
                pruning_reports.append(report)
            for year, year_frame in filtered.groupby(filtered[date_column].dt.year):
                output_path = output_dir / f"{dataset}_{year}.csv"
                _write_frame(output_path, year_frame, append=output_path in initialized)
                initialized.add(output_path)
                written.add(output_path)

    return sorted(written)


def _with_ohlc_valid(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    prices = result[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    result["ohlc_valid"] = (
        (prices["high"] >= prices[["open", "close"]].max(axis=1))
        & (prices["low"] <= prices[["open", "close"]].min(axis=1))
        & (prices["high"] >= prices["low"])
    )
    return result


def write_benchmark_index_yearly_files(
    *,
    input_files: list[Path],
    output_dir: Path,
    start_date: str,
    end_date: str,
    overrides: dict[str, dict[str, object]] | None = None,
    pruning_reports: list[dict[str, object]] | None = None,
    chunksize: int = 500_000,
) -> list[Path]:
    """Write selected benchmark index yearly files from raw index market data."""

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    written: set[Path] = set()
    initialized: set[Path] = set()
    selected_codes = set(BENCHMARK_INDEX_POOL)

    for input_file in sorted(input_files):
        for chunk in pd.read_csv(
            input_file,
            chunksize=chunksize,
            dtype=STRING_CODE_DTYPES,
        ):
            normalized = normalize_private_export_frame(chunk, "index_market")
            filtered = normalized[
                (normalized["trade_date"] >= start)
                & (normalized["trade_date"] <= end)
                & normalized["index_code"].isin(selected_codes)
            ].copy()
            if filtered.empty:
                continue
            filtered["benchmark_name"] = filtered["index_code"].map(
                lambda code: BENCHMARK_INDEX_POOL[code].name
            )
            filtered["benchmark_tier"] = filtered["index_code"].map(
                lambda code: BENCHMARK_INDEX_POOL[code].tier
            )
            filtered = _with_ohlc_valid(filtered)
            filtered = filtered.sort_values(
                ["benchmark_tier", "index_code", "trade_date"]
            )
            filtered, report = apply_interim_column_rules(
                filtered, "benchmark_index_market", overrides=overrides
            )
            if pruning_reports is not None:
                pruning_reports.append(report)
            for year, year_frame in filtered.groupby(filtered["trade_date"].dt.year):
                output_path = output_dir / f"benchmark_index_market_{year}.csv"
                _write_frame(output_path, year_frame, append=output_path in initialized)
                initialized.add(output_path)
                written.add(output_path)

    return sorted(written)


def write_interim_single_file(
    *,
    dataset: str,
    input_files: list[Path],
    output_path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    overrides: dict[str, dict[str, object]] | None = None,
    pruning_reports: list[dict[str, object]] | None = None,
) -> Path:
    """Write one normalized interim CSV for a static, event, or factor table."""

    frames = [read_private_export_file(path, dataset) for path in sorted(input_files)]
    if not frames:
        raise ValueError(f"No input files provided for {dataset}")

    combined = pd.concat(frames, ignore_index=True)
    normalized = normalize_private_export_frame(combined, dataset)

    if dataset in {"adjustment", "market_cap"} and end_date is not None:
        date_column = _date_column(dataset)
        end = pd.Timestamp(end_date)
        normalized = normalized[normalized[date_column] <= end]
        if dataset == "adjustment" and start_date is not None:
            start = pd.Timestamp(start_date)
            normalized = normalized[normalized[date_column] >= start]

    normalized, report = apply_interim_column_rules(
        normalized, dataset, overrides=overrides
    )
    if pruning_reports is not None:
        pruning_reports.append(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _format_dates_for_csv(normalized).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
    return output_path


def find_dataset_files(raw_root: Path, dataset: str) -> list[Path]:
    """Find raw CSV files for one normalized CSMAR dataset."""

    if dataset == "index_info":
        return sorted(
            path
            for path in raw_root.rglob("csmar__index_info__*.csv")
            if path.is_file() and "__description__" not in path.name
        )

    return sorted(
        path
        for path in (raw_root / dataset).rglob(f"csmar__{dataset}__*.csv")
        if path.is_file() and "__description__" not in path.name
    )


def _output_single_path(output_root: Path, dataset: str, start: str, end: str) -> Path:
    compact_start = start.replace("-", "")
    compact_end = end.replace("-", "")
    if dataset == "stock_master":
        return output_root / "stock_master" / "stock_master.csv"
    if dataset == "index_info":
        return output_root / "index_info" / "index_info.csv"
    if dataset == "market_cap":
        return output_root / "market_cap" / f"market_cap_to_{compact_end}.csv"
    if dataset == "adjustment":
        return output_root / "adjustment" / (
            f"adjustment_{compact_start}_{compact_end}.csv"
        )
    raise ValueError(f"Unsupported single-file dataset: {dataset}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build local private interim CSMAR files without changing raw data."
    )
    parser.add_argument("--raw-root", default="data/raw_private/csmar")
    parser.add_argument("--output-root", default="data/interim_private/csmar")
    parser.add_argument(
        "--override-path",
        default="config/private_column_overrides.json",
    )
    parser.add_argument("--start-date", default="2014-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args(argv)

    raw_root = Path(args.raw_root)
    output_root = Path(args.output_root)
    if not raw_root.exists():
        print(f"ERROR: raw root does not exist: {raw_root}")
        return 2

    overrides = load_or_create_column_overrides(Path(args.override_path))
    pruning_reports: list[dict[str, object]] = []

    daily_files = find_dataset_files(raw_root, "daily_market")
    if daily_files:
        outputs = write_interim_yearly_files(
            dataset="daily_market",
            input_files=daily_files,
            output_dir=output_root / "daily_market_by_year",
            start_date=args.start_date,
            end_date=args.end_date,
            overrides=overrides,
            pruning_reports=pruning_reports,
            chunksize=args.chunksize,
        )
        print(f"daily_market: wrote {len(outputs)} yearly files")
        for path in outputs:
            print(f"  {path}")
    else:
        print("skip daily_market: no raw CSV files found")

    index_files = find_dataset_files(raw_root, "index_market")
    if index_files:
        benchmark_outputs = write_benchmark_index_yearly_files(
            input_files=index_files,
            output_dir=output_root / "benchmark_index_market_by_year",
            start_date=args.start_date,
            end_date=args.end_date,
            overrides=overrides,
            pruning_reports=pruning_reports,
            chunksize=args.chunksize,
        )
        print(f"benchmark_index_market: wrote {len(benchmark_outputs)} yearly files")
        for path in benchmark_outputs:
            print(f"  {path}")
    else:
        print("skip benchmark_index_market: no raw index_market CSV files found")

    for dataset in ("stock_master", "adjustment", "market_cap", "index_info"):
        files = find_dataset_files(raw_root, dataset)
        if not files:
            print(f"skip {dataset}: no raw CSV files found")
            continue
        output_path = _output_single_path(
            output_root, dataset, args.start_date, args.end_date
        )
        written = write_interim_single_file(
            dataset=dataset,
            input_files=files,
            output_path=output_path,
            start_date=args.start_date,
            end_date=args.end_date,
            overrides=overrides,
            pruning_reports=pruning_reports,
        )
        print(f"{dataset}: wrote {written}")

    report_path = write_column_pruning_report(
        output_root / "_manifests" / "column_pruning_report.md",
        pruning_reports,
    )
    print(f"column pruning report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
