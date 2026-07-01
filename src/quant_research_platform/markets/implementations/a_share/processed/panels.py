"""Pure A-share market-panel calculations from source-normalized interim frames."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from quant_research_platform.catalog.fields import canonical_field_names
from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.markets.implementations.a_share.calendar import (
    CURRENT_MARKET_TYPES,
    compute_listing_age_trading_days,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    MARKET_ID,
    require_supported_panel_frequency,
)

_CANONICAL_NAMES = canonical_field_names()
MARKET_PANEL_KEY_COLUMNS: tuple[str, ...] = (
    _CANONICAL_NAMES.market_id,
    _CANONICAL_NAMES.asset_id,
    _CANONICAL_NAMES.timestamp,
    _CANONICAL_NAMES.session_date,
    _CANONICAL_NAMES.available_at,
)
ST_STATUS_CODES = {2, 3, 5, 6, 8, 9, 11, 12, 14, 15}


def compute_market_panel(
    market_bars: pd.DataFrame,
    *,
    frequency: DataFrequency | str = DataFrequency.DAILY,
    stock_master: pd.DataFrame,
    market_cap: pd.DataFrame,
    trading_calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the A-share processed market panel from source interim frames."""

    require_supported_panel_frequency(frequency, context="market panel")
    market_bars = _ensure_datetime(
        _with_time_fields(_with_asset_keys(market_bars)),
        ["timestamp", "session_date", "available_at"],
    ).copy()
    market_bars = _with_market_semantics(market_bars)
    market_bars = market_bars.loc[market_bars["is_a_share_scope"].astype(bool)].copy()
    missing_canonical = [
        column
        for column in MARKET_PANEL_KEY_COLUMNS
        if column not in market_bars.columns
    ]
    if missing_canonical:
        raise ValueError(
            "market_bars_interim_missing_canonical_columns: "
            + ", ".join(missing_canonical)
        )
    stocks = _ensure_datetime(
        _with_asset_keys(stock_master),
        ["listing_date", "delisting_date", "status_change_date"],
    )
    cap = _ensure_datetime(
        _with_asset_keys(market_cap),
        ["timestamp", "share_change_date"],
    )
    if "timestamp" not in cap or cap["timestamp"].isna().any():
        raise ValueError("missing_market_cap_panel_timestamp")

    keep_stock_columns = [
        column
        for column in (
            "asset_id",
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
    panel = _asof_join_stock_master(market_bars, stocks[keep_stock_columns])
    panel["listing_age_days"] = (panel["timestamp"] - panel["listing_date"]).dt.days

    cap_columns = [
        column
        for column in (
            "asset_id",
            "timestamp",
            "share_change_date",
            "total_shares",
            "a_float_shares",
        )
        if column in cap.columns
    ]
    cap = (
        _dedupe_exact_key_duplicates_or_raise_conflict(
            cap[cap_columns],
            key_columns=("asset_id", "timestamp"),
            dataset="market_cap",
        )
        .sort_values(["asset_id", "timestamp"], kind="mergesort")
        .reset_index(drop=True)
    )
    merged_groups = []
    for asset_id, stock_frame in panel.groupby("asset_id", sort=False):
        cap_frame = cap[cap["asset_id"] == asset_id]
        if cap_frame.empty:
            stock_result = stock_frame.copy()
            stock_result["share_change_date"] = pd.NaT
            stock_result["total_shares"] = pd.NA
            stock_result["a_float_shares"] = pd.NA
        else:
            stock_result = pd.merge_asof(
                stock_frame.sort_values("timestamp"),
                cap_frame.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                suffixes=("", "_capchg"),
            )
            if "asset_id_capchg" in stock_result.columns:
                stock_result = stock_result.drop(columns=["asset_id_capchg"])
        merged_groups.append(stock_result)
    if merged_groups:
        panel = pd.concat(merged_groups, ignore_index=True)
    else:
        panel = panel.copy()
        panel["share_change_date"] = pd.Series(dtype="datetime64[ns]")
        panel["total_shares"] = pd.Series(dtype="float64")
        panel["a_float_shares"] = pd.Series(dtype="float64")
    volume = pd.to_numeric(panel["volume"], errors="coerce")
    a_float_shares = pd.to_numeric(panel["a_float_shares"], errors="coerce")
    panel["turnover"] = volume.div(a_float_shares.where(a_float_shares.gt(0)))
    panel = panel.sort_values(["market_id", "asset_id", "timestamp"]).reset_index(
        drop=True
    )
    market_type_column = "market_type" if "market_type" in panel.columns else None
    panel["listing_age_trading_days"] = compute_listing_age_trading_days(
        panel,
        trading_calendar,
        observation_date_column="timestamp",
        listing_date_column="listing_date",
        market_type_column=market_type_column,
    )
    leading = [
        "market_id",
        "asset_id",
        "timestamp",
        "session_date",
        "available_at",
    ]
    panel = panel.drop(
        columns=[
            column
            for column in ("source_asset_id", "stock_code", "index_code")
            if column in panel.columns
        ]
    )
    ordered = [column for column in leading if column in panel.columns]
    ordered.extend(column for column in panel.columns if column not in ordered)
    return panel[ordered]


def _with_asset_keys(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "asset_id" not in result.columns:
        if "source_asset_id" not in result.columns:
            raise ValueError("missing_source_asset_id")
        result["asset_id"] = result["source_asset_id"].astype(str)
    result["market_id"] = MARKET_ID
    return result


def _with_time_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "timestamp" not in result.columns:
        raise ValueError("missing_source_timestamp")
    timestamp = pd.to_datetime(result["timestamp"], errors="coerce")
    result["timestamp"] = timestamp
    result["session_date"] = timestamp
    result["available_at"] = timestamp
    return result


def _with_market_semantics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["trading_status_code"] = pd.to_numeric(
        result["trading_status_code"],
        errors="coerce",
    )
    result["filling_flag"] = pd.to_numeric(result["filling_flag"], errors="coerce")
    result["market_type"] = pd.to_numeric(result["market_type"], errors="coerce")
    raw_limit_status = result["limit_status"]
    result["limit_status"] = pd.to_numeric(raw_limit_status, errors="coerce")
    invalid_non_null = result["limit_status"].isna() & raw_limit_status.notna()
    invalid_value = result["limit_status"].notna() & ~result["limit_status"].isin(
        [-1, 0, 1]
    )
    if invalid_non_null.any() or invalid_value.any():
        raise ValueError("invalid_limit_status")
    filled_limit_status = result["limit_status"].fillna(0)
    result["is_limit_up"] = filled_limit_status.eq(1)
    result["is_limit_down"] = filled_limit_status.eq(-1)
    result["is_trading"] = result["filling_flag"].eq(0)
    result["is_st"] = result["trading_status_code"].isin(ST_STATUS_CODES)
    result["is_pt"] = result["trading_status_code"].eq(16)
    result["is_a_share_scope"] = result["market_type"].isin(CURRENT_MARKET_TYPES)
    return result


def _ensure_datetime(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def _asof_join_stock_master(
    market_frame: pd.DataFrame,
    stocks: pd.DataFrame,
) -> pd.DataFrame:
    stock_rows = stocks.copy()
    stock_rows["stock_effective_date"] = _stock_master_effective_date(stock_rows)
    if stock_rows["stock_effective_date"].isna().any():
        raise ValueError("missing_stock_master_panel_effective_timestamp")
    stock_rows = _dedupe_exact_key_duplicates_or_raise_conflict(
        stock_rows,
        key_columns=("asset_id", "stock_effective_date"),
        dataset="stock_master",
    )
    stock_rows["_stock_master_row_order"] = range(len(stock_rows))
    static_listing_date = _first_static_stock_value(stock_rows, "listing_date")
    stock_rows = stock_rows.sort_values(
        ["asset_id", "stock_effective_date", "_stock_master_row_order"],
        kind="mergesort",
    )
    merged_groups = []
    stock_output_columns = [
        column
        for column in stocks.columns
        if column not in {"asset_id", "stock_effective_date"}
    ]
    for asset_id, market_bar_frame in market_frame.groupby("asset_id", sort=False):
        stock_frame = stock_rows[stock_rows["asset_id"] == asset_id]
        if stock_frame.empty:
            stock_result = market_bar_frame.copy()
            for column in stock_output_columns:
                if column not in stock_result.columns:
                    stock_result[column] = pd.NA
        else:
            stock_result = pd.merge_asof(
                market_bar_frame.sort_values("timestamp"),
                stock_frame.sort_values("stock_effective_date"),
                left_on="timestamp",
                right_on="stock_effective_date",
                direction="backward",
                suffixes=("", "_stock_master"),
            )
            if "asset_id_stock_master" in stock_result.columns:
                stock_result = stock_result.drop(columns=["asset_id_stock_master"])
            stock_result = stock_result.drop(
                columns=[
                    column
                    for column in ("stock_effective_date", "_stock_master_row_order")
                    if column in stock_result.columns
                ]
            )
        if "listing_date" in stock_result.columns and asset_id in static_listing_date:
            stock_result["listing_date"] = stock_result["listing_date"].fillna(
                static_listing_date[asset_id]
            )
        merged_groups.append(stock_result)
    if not merged_groups:
        stock_result = market_frame.copy()
        for column in stock_output_columns:
            if column not in stock_result.columns:
                stock_result[column] = _empty_stock_master_column(column)
        return stock_result
    return pd.concat(merged_groups, ignore_index=True)


def _empty_stock_master_column(column: str) -> pd.Series:
    if column in {"listing_date", "delisting_date", "status_change_date"}:
        return pd.Series(dtype="datetime64[ns]")
    return pd.Series(dtype="object")


def _stock_master_effective_date(stocks: pd.DataFrame) -> pd.Series:
    effective = (
        pd.to_datetime(stocks["status_change_date"], errors="coerce")
        if "status_change_date" in stocks
        else pd.Series(pd.NaT, index=stocks.index)
    )
    if "listing_date" in stocks:
        effective = effective.combine_first(
            pd.to_datetime(stocks["listing_date"], errors="coerce")
        )
    return effective


def _dedupe_exact_key_duplicates_or_raise_conflict(
    frame: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    dataset: str,
) -> pd.DataFrame:
    missing = [column for column in key_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"missing_panel_join_key_columns: {dataset}: {', '.join(missing)}"
        )
    exact_deduped = frame.drop_duplicates(keep="first").copy()
    if exact_deduped.duplicated(list(key_columns)).any():
        raise ValueError(f"conflicting_panel_join_duplicate_keys: {dataset}")
    return exact_deduped


def _first_static_stock_value(
    stocks: pd.DataFrame,
    column: str,
) -> dict[object, object]:
    if column not in stocks:
        return {}
    values = (
        stocks.dropna(subset=[column])
        .sort_values(["asset_id", "_stock_master_row_order"], kind="mergesort")
        .drop_duplicates(["asset_id"], keep="first")
        .set_index("asset_id")[column]
    )
    return values.to_dict()


__all__ = ["MARKET_PANEL_KEY_COLUMNS", "compute_market_panel"]
