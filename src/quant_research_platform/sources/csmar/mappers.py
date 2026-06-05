"""CSMAR raw-to-interim mapping helpers.

This module is intentionally read/transform only. Persisted interim output is
owned by ``sources.csmar.importer`` through the spec-driven storage writer.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.fields import canonical_field_names
from quant_research_platform.sources.csmar.field_mappings import (
    map_csmar_columns,
    normalize_stock_code,
)
from quant_research_platform.sources.csmar.interim import (
    normalize_asset_time_frame,
    normalize_daily_market_frame,
    normalize_index_metadata_frame,
    normalize_index_time_frame,
)

DAILY_A_SHARE_MARKET_TYPES = {1, 4, 16, 32}
DAILY_ST_STATUS_CODES = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15}
STRING_CODE_DTYPES = {"Stkcd": str, "Symbol": str, "Indexcd": str}


@dataclass(frozen=True)
class BenchmarkIndex:
    """Candidate benchmark index metadata."""

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


def normalize_private_export_frame(
    frame: pd.DataFrame,
    dataset: str,
    *,
    fallback_timestamp: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalize one raw CSMAR frame to canonical interim columns."""

    mapped = map_csmar_columns(frame, dataset)
    if "stock_code" in mapped.columns:
        mapped["stock_code"] = normalize_stock_code(mapped["stock_code"])
    if "index_code" in mapped.columns:
        mapped["index_code"] = mapped["index_code"].astype(str).str.strip()

    if dataset == "trading_calendar":
        return _normalize_trading_calendar_frame(mapped)

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
        result = normalize_daily_market_frame(result)
        return result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    if dataset == "index_market":
        result = mapped.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result["return_1d"] = result["return_1d_pct"]
        result = normalize_index_time_frame(result)
        return result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    if dataset == "adjustment":
        result = mapped.copy()
        result["trade_date"] = pd.to_datetime(result["trade_date"])
        result = normalize_asset_time_frame(result, time_column="trade_date")
        return result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    if dataset == "market_cap":
        result = mapped.copy()
        result["share_change_date"] = pd.to_datetime(result["share_change_date"])
        result = normalize_asset_time_frame(result, time_column="share_change_date")
        return result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    if dataset == "stock_master":
        result = mapped.copy()
        for column in ("listing_date", "delisting_date", "status_change_date"):
            if column in result.columns:
                result[column] = pd.to_datetime(result[column], errors="coerce")
        status_change = (
            result["status_change_date"]
            if "status_change_date" in result.columns
            else pd.Series(pd.NaT, index=result.index)
        )
        result["asset_record_date"] = status_change.combine_first(
            result["listing_date"]
        )
        result = normalize_asset_time_frame(result, time_column="asset_record_date")
        return result.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    if dataset == "index_info":
        result = mapped.copy()
        if "index_class" not in result.columns and "index_category" in result.columns:
            result["index_class"] = result["index_category"]
        if "release_date" not in result.columns and "start_date" in result.columns:
            result["release_date"] = result["start_date"]
        for column in ("release_date", "base_date", "start_date"):
            if column in result.columns:
                result[column] = pd.to_datetime(result[column], errors="coerce")
        result = normalize_index_metadata_frame(result)
        if fallback_timestamp is not None:
            result = _fill_missing_canonical_time(result, fallback_timestamp)
        return result.sort_values(["asset_id"]).reset_index(drop=True)

    raise ValueError(f"unsupported_csmar_dataset: {dataset}")


def _normalize_trading_calendar_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["calendar_date"] = pd.to_datetime(result["calendar_date"], errors="raise")
    result["source_market_type"] = pd.to_numeric(
        result["source_market_type"],
        errors="raise",
    ).astype("int64")
    result["source_day_of_week"] = pd.to_numeric(
        result["source_day_of_week"],
        errors="raise",
    ).astype("int64")
    result["source_state"] = result["source_state"].astype("string").str.strip()
    result["source_state"] = result["source_state"].str.upper()
    invalid = sorted(
        state
        for state in result["source_state"].dropna().unique()
        if state not in {"O", "C"}
    )
    if invalid or result["source_state"].isna().any():
        raise ValueError(f"invalid_trading_calendar_state: {invalid}")
    duplicate = result.duplicated(["source_market_type", "calendar_date"], keep=False)
    if duplicate.any():
        raise ValueError("duplicate_trading_calendar_key")
    result["is_trading_day"] = result["source_state"].eq("O")
    result["market_id"] = "a_share"
    result["source_id"] = "csmar"
    result["timestamp"] = result["calendar_date"]
    columns = [
        "market_id",
        "source_id",
        "timestamp",
        "calendar_date",
        "source_market_type",
        "is_trading_day",
        "source_day_of_week",
        "source_state",
    ]
    return (
        result[columns]
        .sort_values(["source_market_type", "calendar_date"])
        .reset_index(drop=True)
    )


def read_private_export_file(path: str | Path, dataset: str) -> pd.DataFrame:
    """Read one private CSMAR CSV export and handle known CSV edge cases."""

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
    return pd.DataFrame(rows, columns=header).replace("", pd.NA)


def iter_private_export_chunks(
    path: str | Path,
    *,
    dataset: str,
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    """Yield raw export chunks for large time-series files."""

    export_path = Path(path)
    if dataset == "index_info":
        yield read_private_export_file(export_path, dataset)
        return
    yield from pd.read_csv(export_path, chunksize=chunksize, dtype=STRING_CODE_DTYPES)


def date_column(dataset: str) -> str:
    """Return the canonical time/filter column for one normalized CSMAR dataset."""

    if dataset in {"daily_market", "index_market", "adjustment"}:
        return "timestamp"
    if dataset in {"market_cap", "stock_master", "index_info"}:
        return "timestamp"
    raise ValueError(f"dataset_has_no_filtering_date_column: {dataset}")


def filter_by_date_range(
    frame: pd.DataFrame,
    *,
    dataset: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """Filter a normalized frame by timestamp when bounds are provided."""

    if start_date is None and end_date is None:
        return frame
    column = date_column(dataset)
    result = frame.copy()
    timestamps = pd.to_datetime(result[column], errors="coerce")
    mask = pd.Series(True, index=result.index)
    if start_date is not None:
        mask &= timestamps >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= timestamps <= pd.Timestamp(end_date)
    return result[mask].copy()


def with_benchmark_index_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter index market rows to the approved benchmark pool and add metadata."""

    selected_codes = set(BENCHMARK_INDEX_POOL)
    result = frame[frame["index_code"].isin(selected_codes)].copy()
    if result.empty:
        return result
    result["benchmark_name"] = result["index_code"].map(
        lambda code: BENCHMARK_INDEX_POOL[code].name
    )
    result["benchmark_tier"] = result["index_code"].map(
        lambda code: BENCHMARK_INDEX_POOL[code].tier
    )
    result = with_ohlc_valid(result)
    return result.sort_values(["benchmark_tier", "asset_id", "timestamp"])


def with_ohlc_valid(frame: pd.DataFrame) -> pd.DataFrame:
    """Add an OHLC validity flag."""

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


def find_dataset_files(raw_root: Path, dataset: str) -> list[Path]:
    """Find raw CSMAR CSV files for one normalized dataset id."""

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


def _fill_missing_canonical_time(
    frame: pd.DataFrame,
    fallback_timestamp: str | pd.Timestamp,
) -> pd.DataFrame:
    names = canonical_field_names()
    result = frame.copy()
    fallback = pd.Timestamp(fallback_timestamp).normalize()
    for column in (names.timestamp, names.session_date, names.available_at):
        parsed = pd.to_datetime(result[column], errors="coerce")
        result[column] = parsed.fillna(fallback)
    return result


__all__ = [
    "BENCHMARK_INDEX_POOL",
    "BenchmarkIndex",
    "DAILY_A_SHARE_MARKET_TYPES",
    "DAILY_ST_STATUS_CODES",
    "STRING_CODE_DTYPES",
    "filter_by_date_range",
    "find_dataset_files",
    "iter_private_export_chunks",
    "normalize_private_export_frame",
    "read_private_export_file",
    "with_benchmark_index_fields",
    "with_ohlc_valid",
]
