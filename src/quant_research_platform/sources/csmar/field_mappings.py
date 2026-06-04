"""Internal provider field mapping and normalization helpers."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.sources.csmar.raw_schemas import (
    RAW_TABLE_CONTRACTS,
    missing_raw_columns,
)


def map_csmar_columns(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Rename known raw provider columns to internal canonical names."""

    contract = RAW_TABLE_CONTRACTS[dataset]
    missing = missing_raw_columns(frame, dataset)
    if missing:
        raise ValueError(f"{dataset} raw export is missing CSMAR columns: {missing}")
    aliases = contract.column_aliases or {}
    alias_mapping = {
        alias: contract.column_mapping[column]
        for column, alias_columns in aliases.items()
        for alias in alias_columns
    }
    return frame.rename(columns={**alias_mapping, **contract.column_mapping}).copy()


def normalize_stock_code(series: pd.Series) -> pd.Series:
    """Normalize provider stock codes to six-character strings."""

    return (
        series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )


def normalize_trdnew_capchg(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw share-change rows."""

    result = map_csmar_columns(frame, "market_cap")
    result["stock_code"] = normalize_stock_code(result["stock_code"])
    result["share_change_date"] = pd.to_datetime(result["share_change_date"])
    return result.sort_values(["stock_code", "share_change_date"]).reset_index(
        drop=True
    )


def normalize_trdnew_daily_market(
    frame: pd.DataFrame, *, raw_capchg: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Normalize raw daily-market rows to the internal daily schema."""

    result = map_csmar_columns(frame, "daily_market")
    result["stock_code"] = normalize_stock_code(result["stock_code"])
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    result["return_1d"] = result["return_with_dividend"]
    result["adj_close"] = result["comparable_close_with_dividend"]
    result["adj_open"] = result["open"] * result["adj_close"] / result["close"]
    result["float_market_cap"] = result["float_market_cap_thousand"] * 1000.0
    result["total_market_cap"] = result["total_market_cap_thousand"] * 1000.0
    result["is_limit_up"] = result["limit_status"] == 1
    result["is_limit_down"] = result["limit_status"] == -1
    result["is_trading"] = result["filling_flag"].astype(str).eq("0")
    result["is_st"] = result["trading_status_code"].isin(
        [2, 3, 5, 6, 8, 9, 11, 12, 14, 15]
    )

    if raw_capchg is not None:
        capchg = normalize_trdnew_capchg(raw_capchg)
        merged_groups = []
        for stock_code, stock_frame in result.groupby("stock_code", sort=False):
            capchg_frame = capchg[capchg["stock_code"] == stock_code]
            merged_groups.append(
                pd.merge_asof(
                    stock_frame.sort_values("trade_date"),
                    capchg_frame.sort_values("share_change_date"),
                    left_on="trade_date",
                    right_on="share_change_date",
                    direction="backward",
                    suffixes=("", "_capchg"),
                )
            )
        result = pd.concat(merged_groups, ignore_index=True)
        result["float_shares"] = result["a_float_shares"]
        result["turnover"] = result["volume"] / result["float_shares"]
    else:
        result["turnover"] = pd.NA

    return result.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)


__all__ = [
    "map_csmar_columns",
    "normalize_stock_code",
    "normalize_trdnew_capchg",
    "normalize_trdnew_daily_market",
]
