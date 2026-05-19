"""Build private A-share research panel files from interim CSMAR tables."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from runtime.paths import panel_full_path  # noqa: E402


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
    """Build the first-version A-share research panel from interim tables.

    The panel stays at the stock-date grain. Static stock attributes are joined
    directly, while share counts use an as-of join so each trading date sees the
    latest known capital structure, not a future share-change record.
    """

    daily = _ensure_datetime(daily_market, ["trade_date"]).copy()
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
            "csrc_2012_industry_code",
            "csrc_2012_industry_name",
        )
        if column in stocks.columns
    ]
    panel = daily.merge(stocks[keep_stock_columns], on="stock_code", how="left")
    panel["listing_age_days"] = (
        panel["trade_date"] - panel["listing_date"]
    ).dt.days

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
    panel["turnover"] = (
        pd.to_numeric(panel["volume"], errors="coerce")
        / pd.to_numeric(panel["a_float_shares"], errors="coerce")
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
    # Entry/exit flags are based on the next trading day because labels use the
    # next execution open. Limit-up blocks new entries; limit-down blocks exits.
    panel["next_day_is_trading"] = grouped["is_trading"].shift(-1).fillna(False)
    panel["next_day_limit_status"] = grouped["limit_status"].shift(-1)
    panel["can_enter"] = (
        panel["eligible"]
        & panel["next_day_is_trading"].astype(bool)
        & panel["next_day_limit_status"].ne(1)
    )
    panel["can_exit"] = (
        panel["next_day_is_trading"].astype(bool)
        & panel["next_day_limit_status"].ne(-1)
    )
    return panel


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"stock_code": str},
        encoding="utf-8-sig",
        low_memory=False,
    )


def write_panel_parquet(path: Path, frame: pd.DataFrame) -> Path:
    """Write compressed Parquet using the processed-data default."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


def _market_cap_path(interim_root: Path, end_date: str) -> Path:
    compact_end = end_date.replace("-", "")
    return interim_root / "market_cap" / f"market_cap_to_{compact_end}.csv"


def build_panel_full_from_interim(
    *,
    interim_root: Path,
    processed_root: Path,
    start_year: int,
    end_year: int,
    end_date: str,
    min_listing_trading_days: int = 120,
) -> tuple[Path, int]:
    """Build one full-sample panel Parquet directly from interim yearly tables."""

    stock_master = _read_csv(interim_root / "stock_master" / "stock_master.csv")
    market_cap = _read_csv(_market_cap_path(interim_root, end_date))
    frames = []
    for year in range(start_year, end_year + 1):
        daily_path = (
            interim_root / "daily_market_by_year" / f"daily_market_{year}.csv"
        )
        if not daily_path.exists():
            raise FileNotFoundError(daily_path)
        frames.append(_read_csv(daily_path))
    daily = pd.concat(frames, ignore_index=True)
    panel = build_research_panel(
        daily,
        stock_master=stock_master,
        market_cap=market_cap,
        min_listing_trading_days=min_listing_trading_days,
    )
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel = panel.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    output_path = panel_full_path(processed_root, start_year, end_year)
    write_panel_parquet(output_path, panel)
    return output_path, len(panel)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build full-sample private CSMAR research panel."
    )
    parser.add_argument("--interim-root", default="data/interim_private/csmar")
    parser.add_argument("--processed-root", default="data/processed_private/csmar")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--min-listing-trading-days", type=int, default=120)
    args = parser.parse_args(argv)

    output_path, rows = build_panel_full_from_interim(
        interim_root=Path(args.interim_root),
        processed_root=Path(args.processed_root),
        start_year=args.start_year,
        end_year=args.end_year,
        end_date=args.end_date,
        min_listing_trading_days=args.min_listing_trading_days,
    )
    print(f"wrote {output_path} rows={rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
