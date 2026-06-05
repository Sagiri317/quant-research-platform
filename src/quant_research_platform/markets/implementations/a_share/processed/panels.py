"""从 interim CSMAR tables 构建私有 A-share research panel 文件。"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.fields import canonical_field_names
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.processed.io import (
    layout_from_processed_root,
)
from quant_research_platform.storage.parquet import (
    read_partitioned_parquet_dataset,
    write_processed_data,
)

_CANONICAL_NAMES = canonical_field_names()
CANONICAL_DAILY_COLUMNS: tuple[str, ...] = (
    _CANONICAL_NAMES.market_id,
    _CANONICAL_NAMES.asset_id,
    _CANONICAL_NAMES.timestamp,
    _CANONICAL_NAMES.session_date,
    _CANONICAL_NAMES.available_at,
)


def _ensure_datetime(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def build_research_panel(
    daily_market: pd.DataFrame,
    *,
    stock_master: pd.DataFrame,
    market_cap: pd.DataFrame,
    min_listing_trading_days: int = 120,
) -> pd.DataFrame:
    """从 interim tables 构建第一版 A-share research panel。

    The panel stays at the stock-date grain. Static stock attributes are joined
    directly, while share counts use an as-of join so each trading date sees the
    latest known capital structure, not a future share-change record.
    """

    daily = _ensure_datetime(daily_market, ["trade_date", "timestamp"]).copy()
    missing_canonical = [
        column for column in CANONICAL_DAILY_COLUMNS if column not in daily.columns
    ]
    if missing_canonical:
        raise ValueError(
            "daily_market_interim_missing_canonical_columns: "
            + ", ".join(missing_canonical)
        )
    stocks = _ensure_datetime(
        stock_master, ["listing_date", "delisting_date", "status_change_date"]
    )
    cap = _ensure_datetime(market_cap, ["share_change_date"])

    keep_stock_columns = [
        column
        for column in (
            "stock_code",
            "stock_name",
            "listing_date",
            "delisting_date",
            "company_status",
            "status_change_date",
            "csrc_2012_industry_code",
            "csrc_2012_industry_name",
        )
        if column in stocks.columns
    ]
    panel = _asof_join_stock_master(daily, stocks[keep_stock_columns])
    panel["listing_age_days"] = (panel["trade_date"] - panel["listing_date"]).dt.days

    cap_columns = [
        column
        for column in (
            "stock_code",
            "share_change_date",
            "total_shares",
            "a_float_shares",
        )
        if column in cap.columns
    ]
    cap = (
        cap[cap_columns]
        .sort_values(["stock_code", "share_change_date"])
        .drop_duplicates(["stock_code", "share_change_date"], keep="last")
    )
    merged_groups = []
    for stock_code, stock_frame in panel.groupby("stock_code", sort=False):
        cap_frame = cap[cap["stock_code"] == stock_code]
        if cap_frame.empty:
            stock_result = stock_frame.copy()
            stock_result["share_change_date"] = pd.NaT
            stock_result["total_shares"] = pd.NA
            stock_result["a_float_shares"] = pd.NA
        else:
            stock_result = pd.merge_asof(
                stock_frame.sort_values("trade_date"),
                cap_frame.sort_values("share_change_date"),
                left_on="trade_date",
                right_on="share_change_date",
                direction="backward",
                suffixes=("", "_capchg"),
            )
            if "stock_code_capchg" in stock_result.columns:
                stock_result = stock_result.drop(columns=["stock_code_capchg"])
        merged_groups.append(stock_result)
    panel = pd.concat(merged_groups, ignore_index=True)
    panel["turnover"] = pd.to_numeric(panel["volume"], errors="coerce") / pd.to_numeric(
        panel["a_float_shares"], errors="coerce"
    )
    panel = panel.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    panel["listing_age_trading_days"] = panel.groupby("stock_code").cumcount() + 1
    panel["eligible_after_listing_age"] = (
        panel["listing_age_trading_days"] >= min_listing_trading_days
    )
    panel["eligible_liquidity"] = (
        pd.to_numeric(panel["amount"], errors="coerce").gt(0)
        & pd.to_numeric(panel["volume"], errors="coerce").gt(0)
        & pd.to_numeric(panel["turnover"], errors="coerce").ge(0)
    )
    base = panel["eligible_base"].astype(bool) if "eligible_base" in panel else True
    panel["eligible"] = (
        base & panel["eligible_after_listing_age"] & panel["eligible_liquidity"]
    )
    grouped = panel.groupby("stock_code", sort=False)
    # 入场和出场标记基于下一交易日，因为 label 使用下一次可执行开盘。
    # 涨停阻断新开仓，跌停阻断退出。
    panel["next_day_is_trading"] = grouped["is_trading"].shift(-1).fillna(False)
    panel["next_day_limit_status"] = grouped["limit_status"].shift(-1)
    panel["can_enter"] = (
        panel["eligible"]
        & panel["next_day_is_trading"].astype(bool)
        & panel["next_day_limit_status"].ne(1)
    )
    panel["can_exit"] = panel["next_day_is_trading"].astype(bool) & panel[
        "next_day_limit_status"
    ].ne(-1)
    leading = [
        "market_id",
        "asset_id",
        "timestamp",
        "session_date",
        "available_at",
        "stock_code",
        "trade_date",
    ]
    ordered = [column for column in leading if column in panel.columns]
    ordered.extend(column for column in panel.columns if column not in ordered)
    panel = panel[ordered]
    return panel


def _asof_join_stock_master(daily: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    stock_rows = stocks.copy()
    stock_rows["_stock_master_row_order"] = range(len(stock_rows))
    stock_rows["stock_effective_date"] = _stock_master_effective_date(stock_rows)
    static_listing_date = _first_static_stock_value(stock_rows, "listing_date")
    stock_rows = (
        stock_rows.dropna(subset=["stock_effective_date"])
        .sort_values(
            ["stock_code", "stock_effective_date", "_stock_master_row_order"],
            kind="mergesort",
        )
        .drop_duplicates(["stock_code", "stock_effective_date"], keep="last")
    )
    merged_groups = []
    stock_output_columns = [
        column
        for column in stocks.columns
        if column not in {"stock_code", "stock_effective_date"}
    ]
    for stock_code, daily_stock_frame in daily.groupby("stock_code", sort=False):
        stock_frame = stock_rows[stock_rows["stock_code"] == stock_code]
        if stock_frame.empty:
            stock_result = daily_stock_frame.copy()
            for column in stock_output_columns:
                if column not in stock_result.columns:
                    stock_result[column] = pd.NA
        else:
            stock_result = pd.merge_asof(
                daily_stock_frame.sort_values("trade_date"),
                stock_frame.sort_values("stock_effective_date"),
                left_on="trade_date",
                right_on="stock_effective_date",
                direction="backward",
                suffixes=("", "_stock_master"),
            )
            if "stock_code_stock_master" in stock_result.columns:
                stock_result = stock_result.drop(columns=["stock_code_stock_master"])
            stock_result = stock_result.drop(
                columns=[
                    column
                    for column in ("stock_effective_date", "_stock_master_row_order")
                    if column in stock_result.columns
                ]
            )
        if "listing_date" in stock_result.columns and stock_code in static_listing_date:
            stock_result["listing_date"] = stock_result["listing_date"].fillna(
                static_listing_date[stock_code]
            )
        merged_groups.append(stock_result)
    return pd.concat(merged_groups, ignore_index=True)


def _stock_master_effective_date(stocks: pd.DataFrame) -> pd.Series:
    effective = (
        pd.to_datetime(stocks["status_change_date"], errors="coerce")
        if "status_change_date" in stocks
        else pd.Series(pd.NaT, index=stocks.index)
    )
    if "timestamp" in stocks:
        effective = effective.combine_first(
            pd.to_datetime(stocks["timestamp"], errors="coerce")
        )
    if "listing_date" in stocks:
        effective = effective.combine_first(
            pd.to_datetime(stocks["listing_date"], errors="coerce")
        )
    return effective


def _first_static_stock_value(
    stocks: pd.DataFrame,
    column: str,
) -> dict[object, object]:
    if column not in stocks:
        return {}
    values = (
        stocks.dropna(subset=[column])
        .sort_values(["stock_code", "_stock_master_row_order"], kind="mergesort")
        .drop_duplicates(["stock_code"], keep="first")
        .set_index("stock_code")[column]
    )
    return values.to_dict()


def write_panel_processed_data(
    frame: pd.DataFrame,
    *,
    processed_root: Path,
) -> Path:
    """Persist the A-share daily panel through the processed-data catalog."""

    catalog = a_share_processed_catalog()
    spec = catalog.get("daily_panel")
    output = frame[list(spec.required_columns)].copy()
    return write_processed_data(
        output,
        layout=layout_from_processed_root(processed_root),
        catalog=catalog,
        processed_data_id="daily_panel",
    )


def build_panel_from_interim(
    *,
    interim_root: Path,
    processed_root: Path,
    start_year: int,
    end_year: int,
    end_date: str,
    calculation_end_year: int | None = None,
    min_listing_trading_days: int = 120,
) -> tuple[Path, int]:
    """Build a partitioned 0.1 panel from canonical interim parquet data."""

    panel = build_panel_frame_from_interim(
        interim_root=interim_root,
        start_year=start_year,
        end_year=calculation_end_year or end_year,
        min_listing_trading_days=min_listing_trading_days,
    )
    panel = _filter_panel_years(panel, start_year=start_year, end_year=end_year)
    panel = _filter_panel_end_date(panel, end_date=end_date)
    return write_panel_processed_data(panel, processed_root=processed_root), len(panel)


def build_panel_frame_from_interim(
    *,
    interim_root: Path,
    start_year: int,
    end_year: int,
    min_listing_trading_days: int = 120,
) -> pd.DataFrame:
    """Build an in-memory A-share panel from canonical interim parquet data."""

    stock_master = _read_interim_data(interim_root, "stock_master")
    market_cap = _read_interim_data(interim_root, "market_cap")
    daily = _read_interim_data(
        interim_root,
        "daily_market",
        years=tuple(range(start_year, end_year + 1)),
    )
    panel = build_research_panel(
        daily,
        stock_master=stock_master,
        market_cap=market_cap,
        min_listing_trading_days=min_listing_trading_days,
    )
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    return panel


def _read_interim_data(
    interim_root: Path,
    interim_data_id: str,
    *,
    years: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    data_root = interim_root / interim_data_id / "data"
    return read_partitioned_parquet_dataset(data_root, years=years)


def _filter_panel_years(
    frame: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    return result.loc[
        result["trade_date"].dt.year.between(start_year, end_year)
    ].reset_index(drop=True)


def _filter_panel_end_date(frame: pd.DataFrame, *, end_date: str) -> pd.DataFrame:
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"])
    return result.loc[result["trade_date"] <= pd.Timestamp(end_date)].reset_index(
        drop=True
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build full-sample private CSMAR research panel."
    )
    parser.add_argument("--interim-root", default="data/interim/csmar")
    parser.add_argument("--processed-root", default="data/processed/a_share")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--min-listing-trading-days", type=int, default=120)
    args = parser.parse_args(argv)

    output_path, rows = build_panel_from_interim(
        interim_root=Path(args.interim_root),
        processed_root=Path(args.processed_root),
        start_year=args.start_year,
        end_year=args.end_year,
        end_date=args.end_date,
        min_listing_trading_days=args.min_listing_trading_days,
    )
    print(f"wrote daily_panel rows={rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
