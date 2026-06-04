"""Canonical dataset contracts for normalized CSMAR-derived datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetContract:
    """Schema contract for one normalized internal dataset."""

    name: str
    required_columns: tuple[str, ...]
    primary_key: tuple[str, ...]


DATASET_CONTRACTS: dict[str, DatasetContract] = {
    "stock_master": DatasetContract(
        name="stock_master",
        required_columns=(
            "stock_code",
            "stock_name",
            "exchange",
            "board",
            "share_class",
            "listing_date",
            "delisting_date",
            "company_status",
        ),
        primary_key=("stock_code",),
    ),
    "daily_market": DatasetContract(
        name="daily_market",
        required_columns=(
            "stock_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "prev_close",
            "volume",
            "amount",
            "turnover",
            "return_1d",
        ),
        primary_key=("stock_code", "trade_date"),
    ),
    "adjustment": DatasetContract(
        name="adjustment",
        required_columns=("stock_code", "trade_date", "adj_factor"),
        primary_key=("stock_code", "trade_date"),
    ),
    "market_cap": DatasetContract(
        name="market_cap",
        required_columns=(
            "stock_code",
            "trade_date",
            "total_market_cap",
            "float_market_cap",
            "total_shares",
            "float_shares",
        ),
        primary_key=("stock_code", "trade_date"),
    ),
    "trading_status": DatasetContract(
        name="trading_status",
        required_columns=(
            "stock_code",
            "trade_date",
            "is_trading",
            "is_suspended",
        ),
        primary_key=("stock_code", "trade_date"),
    ),
    "st_status": DatasetContract(
        name="st_status",
        required_columns=("stock_code", "st_type", "start_date", "end_date"),
        primary_key=("stock_code", "start_date"),
    ),
    "industry": DatasetContract(
        name="industry",
        required_columns=(
            "stock_code",
            "industry_system",
            "industry_level",
            "industry_code",
            "industry_name",
            "start_date",
            "end_date",
        ),
        primary_key=("stock_code", "industry_system", "industry_level", "start_date"),
    ),
    "index_market": DatasetContract(
        name="index_market",
        required_columns=("index_code", "trade_date", "open", "close", "return_1d"),
        primary_key=("index_code", "trade_date"),
    ),
    "index_info": DatasetContract(
        name="index_info",
        required_columns=("index_code",),
        primary_key=("index_code",),
    ),
}


def missing_required_columns(frame: pd.DataFrame, dataset: str) -> list[str]:
    """Return required canonical columns missing from a normalized dataset."""

    contract = DATASET_CONTRACTS[dataset]
    return [
        column for column in contract.required_columns if column not in frame.columns
    ]


def assert_required_columns(frame: pd.DataFrame, dataset: str) -> None:
    """Raise when a normalized dataset is missing canonical required columns."""

    missing = missing_required_columns(frame, dataset)
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {missing}")


def duplicate_primary_keys(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Return normalized dataset rows with duplicate primary keys."""

    contract = DATASET_CONTRACTS[dataset]
    assert_required_columns(frame, dataset)
    return frame[frame.duplicated(list(contract.primary_key), keep=False)]


def ohlc_violations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return daily-market rows with impossible OHLC relationships."""

    assert_required_columns(frame, "daily_market")
    invalid = (
        (frame["high"] < frame[["open", "close"]].max(axis=1))
        | (frame["low"] > frame[["open", "close"]].min(axis=1))
        | (frame["high"] < frame["low"])
    )
    return frame[invalid]


def negative_liquidity_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return daily-market rows with negative volume, amount, or turnover."""

    assert_required_columns(frame, "daily_market")
    invalid = (frame[["volume", "amount", "turnover"]] < 0).any(axis=1)
    return frame[invalid]


__all__ = [
    "DATASET_CONTRACTS",
    "DatasetContract",
    "assert_required_columns",
    "duplicate_primary_keys",
    "missing_required_columns",
    "negative_liquidity_rows",
    "ohlc_violations",
]
