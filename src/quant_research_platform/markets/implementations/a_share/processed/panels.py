"""A-share market-session facts and dense panel construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_integer_series,
    strict_string_series,
    strict_timestamp_series,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    CURRENT_MARKET_TYPES,
    compute_listing_age_trading_days,
    unified_trading_sessions,
)
from quant_research_platform.markets.implementations.a_share.ids import MARKET_ID

ST_STATUS_CODES = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15}
MARKET_PANEL_KEY_COLUMNS = ("market_id", "asset_id", "timestamp")


def compute_market_panel(
    market_bars: pd.DataFrame,
    *,
    stock_master: pd.DataFrame,
    market_cap: pd.DataFrame,
    trading_calendar: pd.DataFrame,
    industry_history: pd.DataFrame | None = None,
    special_treatment_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build lifecycle × open-session facts with suspension rows retained."""

    bars = _canonical_bars(market_bars)
    if bars.empty:
        return bars
    sessions = unified_trading_sessions(trading_calendar)
    if sessions.empty:
        raise ValueError("a_share_trading_calendar_empty")
    session_window = sessions[
        (sessions >= bars["timestamp"].min()) & (sessions <= bars["timestamp"].max())
    ]
    if session_window.empty:
        return bars.iloc[0:0].copy()
    stocks = _canonical_stock_master(stock_master)
    caps = _canonical_market_cap(market_cap)
    dense = _dense_lifecycle_panel(
        bars,
        stocks,
        session_window,
    )
    dense = _asof_market_cap(dense, caps)
    dense = _asof_industry(dense, stocks, industry_history)
    dense = _apply_special_treatment(dense, special_treatment_history)
    dense = _apply_suspension_semantics(dense)
    delisting_sentinel = pd.Timestamp.max.normalize()
    dense["delisting_date"] = dense["delisting_date"].where(
        dense["delisting_date"] != delisting_sentinel
    )
    dense["listing_age_trading_days"] = compute_listing_age_trading_days(
        dense,
        trading_calendar,
        observation_date_column="timestamp",
        listing_date_column="listing_date",
        market_type_column="market_type",
    )
    dense["scenario_universe"] = (
        dense["is_a_share_scope"]
        & dense["is_trading"]
        & ~dense["is_st"]
        & ~dense["is_pt"]
        & dense["listing_age_trading_days"].ge(252)
        & dense["amount"].gt(0.0)
        & dense["volume"].gt(0.0)
        & dense["turnover"].notna()
        & dense["turnover"].ge(0.0)
    )
    return dense.sort_values(
        ["market_id", "asset_id", "timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def _canonical_bars(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "source_asset_id",
        "timestamp",
        "market_type",
        "trading_status_code",
        "filling_flag",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "period_return",
        "adj_close",
        "adj_open",
        "float_market_cap",
        "total_market_cap",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("market_bars_missing_columns: " + ", ".join(missing))
    result = frame.copy()
    result["asset_id"] = strict_string_series(
        result["source_asset_id"], error="invalid_source_asset_id"
    )
    result["timestamp"] = strict_timestamp_series(
        result["timestamp"], error="invalid_source_timestamp"
    ).dt.normalize()
    for column in (
        "market_type",
        "trading_status_code",
        "filling_flag",
    ):
        result[column] = strict_integer_series(
            result[column], error=f"invalid_{column}", allow_null=True
        )
    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "period_return",
        "adj_close",
        "adj_open",
        "float_market_cap",
        "total_market_cap",
        "limit_status",
    ):
        if column in result:
            result[column] = strict_finite_numeric_series(
                result[column],
                error=f"invalid_market_bar_value: {column}",
                allow_null=True,
            )
        else:
            result[column] = np.nan
    result = result.loc[result["market_type"].isin(CURRENT_MARKET_TYPES)]
    result = result.loc[result["filling_flag"].ne(2)]
    if result.duplicated(["asset_id", "timestamp"]).any():
        raise ValueError("duplicate_market_panel_key")
    result["market_id"] = str(MARKET_ID)
    result["session_date"] = result["timestamp"]
    result["available_at"] = result["timestamp"]
    return result


def _canonical_stock_master(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"source_asset_id", "listing_date"}
    if not required.issubset(frame.columns):
        raise ValueError("stock_master_missing_columns")
    result = frame.copy()
    result["asset_id"] = strict_string_series(
        result["source_asset_id"], error="invalid_stock_master_asset_id"
    )
    for column in ("listing_date", "delisting_date", "timestamp"):
        if column in result:
            result[column] = strict_timestamp_series(
                result[column], error="invalid_stock_master_date", allow_null=True
            ).dt.normalize()
    result["listing_date"] = strict_timestamp_series(
        result["listing_date"], error="invalid_stock_master_date"
    ).dt.normalize()
    if "delisting_date" not in result:
        result["delisting_date"] = pd.NaT
    result["delisting_date"] = result["delisting_date"].fillna(
        pd.Timestamp.max.normalize()
    )
    for column in (
        "stock_name",
        "market_type",
        "company_status",
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    ):
        if column not in result:
            result[column] = pd.NA
    for column in (
        "stock_name",
        "company_status",
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    ):
        result[column] = result[column].astype("string")
    result["stock_effective_date"] = result.get(
        "timestamp", result["listing_date"]
    ).fillna(result["listing_date"])
    return result.sort_values(
        ["asset_id", "stock_effective_date"], kind="mergesort"
    ).drop_duplicates(["asset_id", "stock_effective_date"], keep="last")


def _canonical_market_cap(frame: pd.DataFrame) -> pd.DataFrame:
    if "source_asset_id" not in frame or "timestamp" not in frame:
        raise ValueError("market_cap_missing_columns")
    result = frame.copy()
    result["asset_id"] = strict_string_series(
        result["source_asset_id"], error="invalid_market_cap_asset_id"
    )
    result["timestamp"] = strict_timestamp_series(
        result["timestamp"], error="invalid_market_cap_date"
    ).dt.normalize()
    for column in ("share_change_date", "total_shares", "a_float_shares"):
        if column not in result:
            result[column] = pd.NaT if column == "share_change_date" else np.nan
    result["share_change_date"] = strict_timestamp_series(
        result["share_change_date"], error="invalid_market_cap_date", allow_null=True
    ).dt.normalize()
    for column in ("total_shares", "a_float_shares"):
        result[column] = strict_finite_numeric_series(
            result[column], error=f"invalid_market_cap_value: {column}", allow_null=True
        )
    if result.duplicated(["asset_id", "timestamp"]).any():
        raise ValueError("duplicate_market_cap_key")
    return result.sort_values(["asset_id", "timestamp"], kind="mergesort")


def _dense_lifecycle_panel(
    bars: pd.DataFrame,
    stocks: pd.DataFrame,
    sessions: pd.DatetimeIndex,
) -> pd.DataFrame:
    assets = tuple(sorted(set(bars["asset_id"])))
    rows: list[pd.DataFrame] = []
    for asset_id in assets:
        stock = stocks.loc[stocks["asset_id"].eq(asset_id)]
        listing = stock["listing_date"].min() if not stock.empty else sessions[0]
        delisting = stock["delisting_date"].max() if not stock.empty else sessions[-1]
        dates = sessions[(sessions >= listing) & (sessions <= delisting)]
        if dates.empty:
            continue
        base = pd.DataFrame({"asset_id": asset_id, "timestamp": dates})
        raw = bars.loc[bars["asset_id"].eq(asset_id)]
        base["market_type"] = raw["market_type"].iloc[0]
        base = base.merge(
            raw.drop(
                columns=[
                    "market_id",
                    "session_date",
                    "available_at",
                    "market_type",
                ],
                errors="ignore",
            ),
            on=["asset_id", "timestamp"],
            how="left",
            validate="one_to_one",
        )
        rows.append(base)
    if not rows:
        return bars.iloc[0:0].copy()
    result = pd.concat(rows, ignore_index=True)
    result["market_id"] = str(MARKET_ID)
    result["session_date"] = result["timestamp"]
    result["available_at"] = result["timestamp"]
    return _asof_join_stock_master(result, stocks)


def _asof_join_stock_master(panel: pd.DataFrame, stocks: pd.DataFrame) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    columns = (
        "stock_name",
        "market_type",
        "listing_date",
        "delisting_date",
        "company_status",
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    )
    for asset_id, group in panel.groupby("asset_id", sort=False):
        right = stocks.loc[stocks["asset_id"].eq(asset_id)]
        if right.empty:
            current = group.copy()
            for column in columns:
                current[column] = (
                    pd.NA
                    if column
                    in {
                        "stock_name",
                        "company_status",
                        "industry_standard_id",
                        "industry_level1_code",
                        "industry_level1_name",
                    }
                    else pd.NaT
                )
        else:
            current = pd.merge_asof(
                group.sort_values("timestamp"),
                right.loc[
                    :, ["asset_id", "stock_effective_date", *columns]
                ].sort_values("stock_effective_date"),
                left_on="timestamp",
                right_on="stock_effective_date",
                by="asset_id",
                direction="backward",
                allow_exact_matches=True,
                suffixes=("", "_stock_master"),
            ).drop(columns=["stock_effective_date"])
            for column in columns:
                joined = f"{column}_stock_master"
                if joined in current:
                    current[column] = current[joined].combine_first(
                        current.get(column, pd.Series(pd.NA, index=current.index))
                    )
                    current.drop(columns=joined, inplace=True)
        output.append(current)
    result = pd.concat(output, ignore_index=True)
    for column in (
        "stock_name",
        "company_status",
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    ):
        result[column] = result[column].astype("string")
    return result


def _asof_market_cap(panel: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for asset_id, group in panel.groupby("asset_id", sort=False):
        right = caps.loc[caps["asset_id"].eq(asset_id)]
        if right.empty:
            current = group.copy()
            current["share_change_date"] = pd.NaT
            current["total_shares"] = np.nan
            current["a_float_shares"] = np.nan
        else:
            current = pd.merge_asof(
                group.sort_values("timestamp"),
                right.loc[
                    :,
                    [
                        "asset_id",
                        "timestamp",
                        "share_change_date",
                        "total_shares",
                        "a_float_shares",
                    ],
                ].sort_values("timestamp"),
                on="timestamp",
                by="asset_id",
                direction="backward",
                allow_exact_matches=True,
                suffixes=("", "_cap"),
            )
        current["float_market_cap"] = current["float_market_cap"].fillna(
            current["a_float_shares"] * current["close"]
        )
        current["total_market_cap"] = current["total_market_cap"].fillna(
            current["total_shares"] * current["close"]
        )
        outputs.append(current)
    return pd.concat(outputs, ignore_index=True)


def _asof_industry(
    panel: pd.DataFrame,
    stocks: pd.DataFrame,
    history: pd.DataFrame | None,
) -> pd.DataFrame:
    if history is None or history.empty:
        return panel
    history = history.rename(columns={"source_asset_id": "asset_id"})
    required = {
        "asset_id",
        "timestamp",
        "industry_standard_id",
        "industry_level1_code",
        "industry_level1_name",
    }
    if not required.issubset(history.columns):
        raise ValueError("industry_history_columns_invalid")
    right = history.loc[:, list(required)].sort_values(
        ["asset_id", "timestamp"], kind="mergesort"
    )
    output: list[pd.DataFrame] = []
    for asset_id, group in panel.groupby("asset_id", sort=False):
        current = pd.merge_asof(
            group.sort_values("timestamp"),
            right.loc[right["asset_id"].eq(asset_id)].sort_values("timestamp"),
            on="timestamp",
            by="asset_id",
            direction="backward",
            allow_exact_matches=True,
            suffixes=("", "_industry"),
        )
        for column in (
            "industry_standard_id",
            "industry_level1_code",
            "industry_level1_name",
        ):
            current[column] = current[f"{column}_industry"].combine_first(
                current[column]
            )
            current.drop(columns=f"{column}_industry", inplace=True)
        output.append(current)
    return pd.concat(output, ignore_index=True)


def _apply_special_treatment(
    panel: pd.DataFrame,
    history: pd.DataFrame | None,
) -> pd.DataFrame:
    result = panel.copy()
    status = result["trading_status_code"].fillna(-1)
    result["is_st"] = status.isin(ST_STATUS_CODES)
    result["is_pt"] = status.eq(16)
    if history is None or history.empty:
        return result
    history = history.rename(columns={"source_asset_id": "asset_id"})
    if {"asset_id", "timestamp", "change_type"}.issubset(history.columns):
        right = history.loc[:, ["asset_id", "timestamp", "change_type"]].copy()
        change_type = right["change_type"].astype("string").str.upper()
        right["is_pt"] = change_type.eq("C")
        right["is_st"] = change_type.isin({"B", "D"})
    elif {"asset_id", "timestamp", "is_st", "is_pt"}.issubset(history.columns):
        right = history.loc[:, ["asset_id", "timestamp", "is_st", "is_pt"]].copy()
    else:
        raise ValueError("special_treatment_history_columns_invalid")
    right = right.loc[:, ["asset_id", "timestamp", "is_st", "is_pt"]].sort_values(
        ["asset_id", "timestamp"], kind="mergesort"
    )
    merged = pd.merge_asof(
        result.sort_values("timestamp"),
        right,
        on="timestamp",
        by="asset_id",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_event"),
    )
    merged["is_st"] = merged["is_st_event"].combine_first(merged["is_st"])
    merged["is_pt"] = merged["is_pt_event"].combine_first(merged["is_pt"])
    return merged.drop(columns=["is_st_event", "is_pt_event"])


def _apply_suspension_semantics(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.sort_values(["asset_id", "timestamp"], kind="mergesort").copy()
    result["is_trading"] = result["filling_flag"].fillna(1).eq(0)
    for asset_id, positions in result.groupby("asset_id", sort=False).groups.items():
        _ = asset_id
        values = result.loc[positions]
        last_close = (
            values["adj_close"].where(result.loc[positions, "is_trading"]).ffill()
        )
        missing_price = ~result.loc[positions, "is_trading"]
        result.loc[positions, "adj_close"] = result.loc[positions, "adj_close"].where(
            result.loc[positions, "adj_close"].notna(), last_close
        )
        result.loc[positions, "close"] = result.loc[positions, "close"].where(
            result.loc[positions, "close"].notna(), result.loc[positions, "adj_close"]
        )
        for column in ("open", "high", "low", "adj_open"):
            result.loc[positions, column] = result.loc[positions, column].where(
                result.loc[positions, column].notna(),
                result.loc[positions, "adj_close"],
            )
        result.loc[positions, "period_return"] = result.loc[
            positions, "period_return"
        ].where(result.loc[positions, "is_trading"], 0.0)
        result.loc[positions, "volume"] = result.loc[positions, "volume"].where(
            result.loc[positions, "is_trading"], 0.0
        )
        result.loc[positions, "amount"] = result.loc[positions, "amount"].where(
            result.loc[positions, "is_trading"], 0.0
        )
        for column in ("after_hours_volume", "after_hours_amount"):
            if column not in result:
                result[column] = 0.0
            result.loc[positions, column] = result.loc[positions, column].where(
                result.loc[positions, "is_trading"], 0.0
            )
        free_shares = result.loc[positions, "float_market_cap"].div(
            result.loc[positions, "close"].where(result.loc[positions, "close"].gt(0.0))
        )
        result.loc[positions, "turnover"] = result.loc[positions, "volume"].div(
            free_shares.where(free_shares.gt(0.0))
        )
        previous_trading = result.loc[positions, "is_trading"].shift(1).fillna(False)
        reopening = result.loc[positions, "is_trading"] & ~previous_trading
        prior_close = result.loc[positions, "adj_close"].shift(1).ffill()
        reopening_return = result.loc[positions, "adj_close"].div(prior_close) - 1.0
        result.loc[positions, "period_return"] = result.loc[
            positions, "period_return"
        ].where(~reopening | reopening_return.isna(), reopening_return)
        _ = missing_price
    result["is_trading"] = result["is_trading"].astype(bool)
    result["is_a_share_scope"] = result["market_type"].isin(CURRENT_MARKET_TYPES)
    return result
