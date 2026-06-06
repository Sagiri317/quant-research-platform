from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def make_synthetic_trading_calendar() -> pd.DataFrame:
    rows = [
        ("2023-12-29", True),
        ("2024-01-01", False),
        ("2024-01-02", True),
        ("2024-01-03", True),
        ("2024-01-04", False),
        ("2024-01-05", True),
        ("2024-01-08", True),
        ("2024-01-09", True),
    ]
    return pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "source_id": "csmar",
                "timestamp": pd.Timestamp(date),
                "calendar_date": pd.Timestamp(date),
                "source_market_type": 1,
                "is_trading_day": is_trading_day,
                "source_day_of_week": pd.Timestamp(date).dayofweek + 1,
                "source_state": "O" if is_trading_day else "C",
            }
            for date, is_trading_day in rows
        ]
    )


def make_synthetic_daily_market() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"]
    )
    rows: list[dict[str, object]] = []
    for asset_id, base_price in (
        ("asset_a", 10.0),
        ("asset_b", 20.0),
        ("asset_c", 30.0),
    ):
        for offset, date in enumerate(dates):
            limit_status = 0
            if asset_id == "asset_a" and date == pd.Timestamp("2024-01-03"):
                limit_status = 1
            if asset_id == "asset_b" and date == pd.Timestamp("2024-01-03"):
                limit_status = -1
            is_st = asset_id == "asset_b" and date == pd.Timestamp("2024-01-02")
            is_pt = asset_id == "asset_c" and date == pd.Timestamp("2024-01-02")
            zero_liquidity = asset_id == "asset_c" and date == pd.Timestamp(
                "2024-01-05"
            )
            rows.append(
                {
                    "market_id": "a_share",
                    "asset_id": asset_id,
                    "timestamp": date,
                    "session_date": date,
                    "available_at": date,
                    "market_type": 1,
                    "trading_status_code": 2 if is_st else 16 if is_pt else 1,
                    "filling_flag": "0",
                    "open": base_price + offset,
                    "high": base_price + offset + 0.5,
                    "low": base_price + offset - 0.5,
                    "close": base_price + offset + 0.2,
                    "adj_open": base_price + offset,
                    "adj_close": base_price + offset + 0.2,
                    "prev_close": base_price + offset - 0.1,
                    "volume": 0.0 if zero_liquidity else 1000.0 + offset,
                    "amount": 0.0 if zero_liquidity else 10_000.0 + offset,
                    "return_1d": 0.01 * (offset + 1),
                    "limit_status": limit_status,
                    "is_limit_up": limit_status == 1,
                    "is_limit_down": limit_status == -1,
                    "is_trading": True,
                    "is_st": is_st,
                    "is_pt": is_pt,
                    "is_a_share_scope": True,
                    "eligible_base": not (is_st or is_pt),
                    "turnover": pd.NA,
                    "total_market_cap": (base_price + offset) * 1_000_000.0,
                    "float_market_cap": (base_price + offset) * 500_000.0,
                }
            )
    return pd.DataFrame(rows)


def make_synthetic_stock_master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "asset_id": "asset_a",
                "source_id": "csmar",
                "source_asset_id": "asset_a",
                "timestamp": pd.Timestamp("2023-12-29"),
                "stock_name": "Asset A",
                "listing_date": pd.Timestamp("2023-12-29"),
                "delisting_date": pd.NaT,
                "company_status": "normal",
                "status_change_date": pd.Timestamp("2023-12-29"),
                "csrc_2012_industry_code": "I1",
                "csrc_2012_industry_name": "Industry 1",
            },
            {
                "market_id": "a_share",
                "asset_id": "asset_a",
                "source_id": "csmar",
                "source_asset_id": "asset_a",
                "timestamp": pd.Timestamp("2024-01-05"),
                "stock_name": "Asset A",
                "listing_date": pd.Timestamp("2023-12-29"),
                "delisting_date": pd.NaT,
                "company_status": "watched",
                "status_change_date": pd.Timestamp("2024-01-05"),
                "csrc_2012_industry_code": "I2",
                "csrc_2012_industry_name": "Industry 2",
            },
            {
                "market_id": "a_share",
                "asset_id": "asset_b",
                "source_id": "csmar",
                "source_asset_id": "asset_b",
                "timestamp": pd.Timestamp("2023-12-29"),
                "stock_name": "Asset B",
                "listing_date": pd.Timestamp("2023-12-29"),
                "delisting_date": pd.NaT,
                "company_status": "normal",
                "status_change_date": pd.Timestamp("2023-12-29"),
                "csrc_2012_industry_code": "I3",
                "csrc_2012_industry_name": "Industry 3",
            },
            {
                "market_id": "a_share",
                "asset_id": "asset_c",
                "source_id": "csmar",
                "source_asset_id": "asset_c",
                "timestamp": pd.Timestamp("2024-01-02"),
                "stock_name": "Asset C",
                "listing_date": pd.Timestamp("2024-01-02"),
                "delisting_date": pd.NaT,
                "company_status": "normal",
                "status_change_date": pd.Timestamp("2024-01-02"),
                "csrc_2012_industry_code": "I4",
                "csrc_2012_industry_name": "Industry 4",
            },
        ]
    )


def make_synthetic_market_cap() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _market_cap_row("asset_a", "2023-12-29", 1000.0, 500.0),
            _market_cap_row("asset_a", "2024-01-05", 2000.0, 800.0),
            _market_cap_row("asset_b", "2023-12-29", 3000.0, 1000.0),
            _market_cap_row("asset_c", "2024-01-02", 4000.0, 1200.0),
        ]
    )


def make_synthetic_daily_panel(
    *,
    sessions: int = 130,
    assets: tuple[str, ...] = ("asset_a", "asset_b"),
) -> pd.DataFrame:
    dates = pd.bdate_range("2023-12-15", periods=sessions)
    rows: list[dict[str, object]] = []
    for asset_offset, asset_id in enumerate(assets):
        for offset, date in enumerate(dates):
            price = 10.0 + asset_offset * 100.0 + offset
            rows.append(
                {
                    "market_id": "a_share",
                    "asset_id": asset_id,
                    "timestamp": pd.Timestamp(date),
                    "adj_open": price,
                    "adj_close": price + 0.5,
                }
            )
    return pd.DataFrame(rows)


def make_synthetic_factor_panel(
    *,
    sessions: int = 260,
    assets: tuple[str, ...] = ("asset_a", "asset_b"),
) -> pd.DataFrame:
    dates = pd.bdate_range("2023-12-15", periods=sessions)
    rows: list[dict[str, object]] = []
    for asset_offset, asset_id in enumerate(assets):
        base = 100.0 + 1000.0 * asset_offset
        for offset, date in enumerate(dates):
            close = base + offset + 1.0
            previous_close = base + offset
            rows.append(
                {
                    "market_id": "a_share",
                    "asset_id": asset_id,
                    "timestamp": pd.Timestamp(date),
                    "session_date": pd.Timestamp(date),
                    "available_at": pd.Timestamp(date),
                    "adj_close": close,
                    "return_1d": close / previous_close - 1.0,
                    "amount": 10_000.0 + offset,
                    "turnover": 0.01 + offset / 100_000.0,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "prev_close": previous_close,
                    "total_market_cap": 1_000_000.0 + offset,
                    "float_market_cap": 500_000.0 + offset,
                }
            )
    return pd.DataFrame(rows)


def make_synthetic_index_info_static() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _index_info_row("index_a", "2020-01-01", "2019-12-31", "broad"),
            _index_info_row("index_b", "2021-01-01", "2020-12-31", "style"),
        ]
    )


def make_synthetic_index_info_dynamic() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _index_info_row("index_a", "2020-01-01", "2019-12-31", "broad"),
            _index_info_row("index_a", "2022-01-01", "2019-12-31", "broad_v2"),
            _index_info_row("index_b", "2021-01-01", "2020-12-31", "style"),
        ]
    )


def shuffled(frame: pd.DataFrame, order: Iterable[int]) -> pd.DataFrame:
    return frame.iloc[list(order)].reset_index(drop=True)


def _market_cap_row(
    asset_id: str,
    timestamp: str,
    total_shares: float,
    a_float_shares: float,
) -> dict[str, object]:
    parsed = pd.Timestamp(timestamp)
    return {
        "market_id": "a_share",
        "asset_id": asset_id,
        "timestamp": parsed,
        "share_change_date": parsed,
        "total_shares": total_shares,
        "a_float_shares": a_float_shares,
    }


def _index_info_row(
    asset_id: str,
    release_date: str,
    base_date: str,
    index_class: str,
) -> dict[str, object]:
    return {
        "market_id": "a_share",
        "asset_id": asset_id,
        "source_id": "csmar",
        "source_asset_id": asset_id,
        "timestamp": pd.Timestamp(release_date),
        "index_name": asset_id,
        "calculation_formula": "synthetic formula",
        "index_class": index_class,
        "release_date": pd.Timestamp(release_date),
        "base_date": pd.Timestamp(base_date),
    }
