from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
)
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers import frequency_ids


def processed_a_share_root(project_root: Path, kind: str, data_id: str) -> Path:
    return project_root / "data" / "processed" / "a_share" / kind / data_id


def make_synthetic_market_bars_daily(
    dates: Sequence[str] | None = None,
    *,
    asset_id: str = "000001",
    market_type: int = 4,
) -> pd.DataFrame:
    values = list(dates or ("2024-01-02", "2024-01-03"))
    rows: list[dict[str, object]] = []
    for offset, value in enumerate(values, start=1):
        rows.append(
            make_synthetic_market_bars_daily_row(
                value,
                asset_id=asset_id,
                market_type=market_type,
                open_price=10.0 + offset,
                high=10.5 + offset,
                low=9.8 + offset,
                close=10.2 + offset,
                volume=1000.0 + offset,
                amount=10200.0 + offset,
                prev_close=9.9 + offset,
                period_return=0.01 * offset,
                float_market_cap=100000.0 + offset,
                total_market_cap=200000.0 + offset,
            )
        )
    return pd.DataFrame(rows)


def make_synthetic_market_bars_daily_row(
    value: str,
    *,
    asset_id: str = "000001",
    market_type: int = 4,
    limit_status: object = 0,
    open_price: float = 10.0,
    high: float = 10.5,
    low: float = 9.8,
    close: float = 10.2,
    volume: float = 1000.0,
    amount: float = 10200.0,
    prev_close: float = 9.9,
    period_return: float = 0.01,
    float_market_cap: float = 100000.0,
    total_market_cap: float = 200000.0,
) -> dict[str, object]:
    timestamp = pd.Timestamp(value)
    return {
        "source_asset_id": asset_id,
        "timestamp": timestamp,
        "trade_date": timestamp,
        "market_type": market_type,
        "trading_status_code": 1,
        "filling_flag": "0",
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
        "prev_close": prev_close,
        "limit_status": limit_status,
        "period_return": period_return,
        "adj_close": close,
        "adj_open": open_price,
        "float_market_cap": float_market_cap,
        "total_market_cap": total_market_cap,
        "turnover": 0.01,
    }


def make_synthetic_market_panel(
    *,
    sessions: int = 130,
    assets: Sequence[str] = ("asset_a", "asset_b"),
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
    assets: Sequence[str] = ("asset_a", "asset_b"),
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
                    "period_return": close / previous_close - 1.0,
                    "amount": 10_000.0 + offset,
                    "volume": 1_000.0 + offset,
                    "turnover": 0.01 + offset / 100_000.0,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "prev_close": previous_close,
                    "total_market_cap": 1_000_000.0 + offset,
                    "float_market_cap": 500_000.0 + offset,
                }
            )
    return pd.DataFrame(rows)


def make_synthetic_a_share_stock_master(
    *,
    asset_id: str = "000001",
    listing_date: str = "2010-01-01",
    stock_name: str = "demo",
) -> pd.DataFrame:
    timestamp = pd.Timestamp(listing_date)
    return pd.DataFrame(
        [
            {
                "source_asset_id": asset_id,
                "timestamp": timestamp,
                "stock_name": stock_name,
                "listing_date": timestamp,
                "delisting_date": pd.NaT,
                "company_status": "normal",
                "csrc_2012_industry_code": "I1",
                "csrc_2012_industry_name": "Industry",
                "status_change_date": timestamp,
            }
        ]
    )


def make_synthetic_a_share_market_cap(
    *,
    asset_id: str = "000001",
    share_change_date: str = "2010-01-01",
    a_float_shares: float = 500.0,
) -> pd.DataFrame:
    timestamp = pd.Timestamp(share_change_date)
    return pd.DataFrame(
        [
            {
                "source_asset_id": asset_id,
                "timestamp": timestamp,
                "share_change_date": timestamp,
                "total_shares": 1000.0,
                "a_float_shares": a_float_shares,
            }
        ]
    )


def make_synthetic_a_share_trading_calendar(
    dates: Sequence[str | pd.Timestamp] | None = None,
    *,
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-03",
    market_type: int = 4,
    is_trading_day: bool | None = True,
    closed_dates: Sequence[str] = (),
) -> pd.DataFrame:
    values = (
        list(dates)
        if dates is not None
        else list(pd.date_range(start_date, end_date, freq="D"))
    )
    closed = {pd.Timestamp(value).date() for value in closed_dates}
    rows: list[dict[str, object]] = []
    for value in values:
        timestamp = pd.Timestamp(value)
        trading = (
            is_trading_day
            if is_trading_day is not None
            else timestamp.weekday() < 5 and timestamp.date() not in closed
        )
        rows.append(
            {
                "timestamp": timestamp,
                "calendar_date": timestamp,
                "source_market_type": market_type,
                "is_trading_day": bool(trading),
                "source_day_of_week": timestamp.dayofweek + 1,
                "source_state": "O" if trading else "C",
            }
        )
    return pd.DataFrame(rows)


def make_synthetic_interim_catalog(
    source_id: SourceId,
    *,
    market_bars_data_id: str = frequency_ids.MARKET_BARS_DATA_ID,
    include_extra_daily_column: bool = False,
    drop_daily_column: str | None = None,
) -> InterimDataCatalog:
    specs = {
        market_bars_data_id: (
            "source_asset_id",
            "timestamp",
            "trade_date",
            "market_type",
            "trading_status_code",
            "filling_flag",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "prev_close",
            "limit_status",
            "period_return",
            "adj_close",
            "adj_open",
            "float_market_cap",
            "total_market_cap",
            "turnover",
            *(("vendor_extra_metric",) if include_extra_daily_column else ()),
        ),
        "stock_master": (
            "source_asset_id",
            "timestamp",
            "stock_name",
            "listing_date",
            "delisting_date",
            "company_status",
            "csrc_2012_industry_code",
            "csrc_2012_industry_name",
            "status_change_date",
        ),
        "market_cap": (
            "source_asset_id",
            "timestamp",
            "share_change_date",
            "total_shares",
            "a_float_shares",
        ),
        "trading_calendar": (
            "timestamp",
            "calendar_date",
            "source_market_type",
            "is_trading_day",
            "source_day_of_week",
            "source_state",
        ),
    }
    if drop_daily_column is not None:
        specs[market_bars_data_id] = tuple(
            column
            for column in specs[market_bars_data_id]
            if column != drop_daily_column
        )
    catalog = InterimDataCatalog()
    for data_id, columns in specs.items():
        primary_key = (
            ("timestamp",)
            if data_id == "trading_calendar"
            else ("source_asset_id", "timestamp")
        )
        roles = {
            column: (
                ColumnRole.TIMESTAMP
                if column in {"timestamp", "calendar_date"}
                else ColumnRole.KEY
                if column in primary_key
                else ColumnRole.CONTEXT
            )
            for column in columns
        }
        catalog.register(
            InterimDataSpec(
                data_id=data_id,
                source_id=source_id,
                frequency=DataFrequency.DAILY,
                grain=DataGrain.ASSET_TIMESTAMP,
                column_roles=roles,
                primary_key=primary_key,
                partition_columns=("year",),
            )
        )
    return catalog


def write_synthetic_interim_inputs(
    layout: StorageLayout,
    source_id: SourceId,
    *,
    market_bars_data_id: str = frequency_ids.MARKET_BARS_DATA_ID,
    dates: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    asset_id: str = "000001",
    market_type: int = 4,
    calendar_start_date: str = "2010-01-01",
    calendar_end_date: str = "2024-01-03",
    closed_dates: Sequence[str] = (),
    include_extra_daily_column: bool = False,
    drop_daily_column: str | None = None,
) -> None:
    catalog = make_synthetic_interim_catalog(
        source_id,
        market_bars_data_id=market_bars_data_id,
        include_extra_daily_column=include_extra_daily_column,
        drop_daily_column=drop_daily_column,
    )
    frames = {
        market_bars_data_id: make_synthetic_market_bars_daily(
            dates
            or tuple(
                str(value.date())
                for value in pd.bdate_range(
                    start_date or "2014-01-02",
                    end_date or "2014-01-03",
                )
            ),
            asset_id=asset_id,
            market_type=market_type,
        ),
        "stock_master": make_synthetic_a_share_stock_master(asset_id=asset_id),
        "market_cap": make_synthetic_a_share_market_cap(asset_id=asset_id),
        "trading_calendar": make_synthetic_a_share_trading_calendar(
            start_date=calendar_start_date,
            end_date=calendar_end_date,
            market_type=market_type,
            is_trading_day=None,
            closed_dates=closed_dates,
        ),
    }
    if include_extra_daily_column:
        frames[market_bars_data_id]["vendor_extra_metric"] = range(
            1, len(frames[market_bars_data_id]) + 1
        )
    if drop_daily_column is not None:
        frames[market_bars_data_id] = frames[market_bars_data_id].drop(
            columns=[drop_daily_column]
        )
    for data_id, frame in frames.items():
        spec = catalog.get(data_id)
        write_dataset(
            layout.interim_data_root(source_id, data_id),
            frame.loc[:, list(spec.required_columns)],
            spec=spec,
        )
