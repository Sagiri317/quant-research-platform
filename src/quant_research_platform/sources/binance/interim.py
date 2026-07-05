"""Binance raw export to source-normalized interim datasets."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
)
from quant_research_platform.sources.binance import ids
from quant_research_platform.sources.binance.fields import field_role
from quant_research_platform.sources.binance.raw_exports import (
    dataset_files_by_id,
    iter_raw_rows,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout

MARKET_BARS_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "open_time",
    "close_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "source_market_type",
    "interval",
)
AGGREGATE_TRADES_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "aggregate_trade_id",
    "price",
    "quantity",
    "quote_quantity",
    "first_trade_id",
    "last_trade_id",
    "is_buyer_maker",
    "taker_side",
    "signed_quantity",
    "source_market_type",
)
DEPTH_LEVELS_COLUMNS = (
    "source_asset_id",
    "timestamp",
    "side",
    "level",
    "price",
    "quantity",
    "source_market_type",
)
INTERIM_COLUMNS_BY_DATA_ID = {
    ids.market_bars_data_id(): MARKET_BARS_COLUMNS,
    ids.aggregate_trades_data_id(): AGGREGATE_TRADES_COLUMNS,
    ids.depth_levels_data_id(): DEPTH_LEVELS_COLUMNS,
}


@dataclass(frozen=True)
class BinanceInterimImportResult:
    artifact_roots: dict[str, Path]
    skipped: tuple[str, ...]


def interim_specs() -> tuple[InterimDataSpec, ...]:
    return tuple(
        _interim_spec(data_id, columns)
        for data_id, columns in sorted(INTERIM_COLUMNS_BY_DATA_ID.items())
    )


def interim_catalog() -> InterimDataCatalog:
    catalog = InterimDataCatalog()
    for spec in interim_specs():
        catalog.register(spec)
    return catalog


def iter_interim_frames(
    *,
    data_id: str,
    input_files: tuple[Path, ...],
) -> Iterator[pd.DataFrame]:
    for input_file in sorted(input_files):
        frame = normalize_raw_rows(iter_raw_rows(input_file), data_id=data_id)
        if not frame.empty:
            yield frame


def normalize_raw_rows(rows: Iterator[dict[str, Any]], *, data_id: str) -> pd.DataFrame:
    if data_id == ids.market_bars_data_id():
        return _normalize_klines(tuple(rows))
    if data_id == ids.aggregate_trades_data_id():
        return _normalize_aggregate_trades(tuple(rows))
    if data_id == ids.depth_levels_data_id():
        return _normalize_depth(tuple(rows))
    raise ValueError(f"unsupported_binance_data_id: {data_id}")


def materialize_interim_data(
    layout: StorageLayout,
    raw_root: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    data_ids: tuple[str, ...] = tuple(sorted(INTERIM_COLUMNS_BY_DATA_ID)),
) -> BinanceInterimImportResult:
    del start_date, end_date
    catalog = interim_catalog()
    source_raw_root = raw_root or layout.raw_source_root(ids.SOURCE_ID)
    files_by_data_id = dataset_files_by_id(source_raw_root)
    written: dict[str, Path] = {}
    skipped: list[str] = []
    for data_id in data_ids:
        spec = catalog.get(data_id)
        input_files = files_by_data_id.get(data_id, ())
        if not input_files:
            skipped.append(data_id)
            continue
        artifact_root = layout.interim_data_root(spec.source_id, spec.data_id)
        write_dataset(
            artifact_root,
            iter_interim_frames(data_id=data_id, input_files=input_files),
            spec=spec,
        )
        written[data_id] = artifact_root
    return BinanceInterimImportResult(artifact_roots=written, skipped=tuple(skipped))


def _normalize_klines(rows: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    normalized = [
        {
            "source_asset_id": str(row["symbol"]),
            "timestamp": _timestamp_ms(_first_present(row, "open_time", "openTime")),
            "open_time": _timestamp_ms(_first_present(row, "open_time", "openTime")),
            "close_time": _timestamp_ms(
                _first_present(row, "close_time", "closeTime")
            ),
            "open": _float(row["open"]),
            "high": _float(row["high"]),
            "low": _float(row["low"]),
            "close": _float(row["close"]),
            "volume": _float(row["volume"]),
            "quote_volume": _float(
                _first_present(row, "quote_volume", "quoteVolume")
            ),
            "trade_count": int(_first_present(row, "trade_count", "tradeCount")),
            "taker_buy_volume": _float(
                _first_present(row, "taker_buy_volume", "takerBuyVolume")
            ),
            "taker_buy_quote_volume": _float(
                _first_present(row, "taker_buy_quote_volume", "takerBuyQuoteVolume")
            ),
            "source_market_type": "spot",
            "interval": str(row.get("interval", "1m")),
        }
        for row in rows
    ]
    return _ordered_frame(
        normalized,
        MARKET_BARS_COLUMNS,
        ["source_asset_id", "timestamp"],
    )


def _normalize_aggregate_trades(rows: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    normalized = []
    for row in rows:
        quantity = _float(_first_present(row, "quantity", "q"))
        price = _float(_first_present(row, "price", "p"))
        is_buyer_maker = bool(row.get("is_buyer_maker", row.get("m", False)))
        normalized.append(
            {
                "source_asset_id": str(row["symbol"]),
                "timestamp": _timestamp_ms(_first_present(row, "timestamp", "T")),
                "aggregate_trade_id": int(
                    _first_present(row, "aggregate_trade_id", "a")
                ),
                "price": price,
                "quantity": quantity,
                "quote_quantity": price * quantity,
                "first_trade_id": int(_first_present(row, "first_trade_id", "f")),
                "last_trade_id": int(_first_present(row, "last_trade_id", "l")),
                "is_buyer_maker": is_buyer_maker,
                "taker_side": "sell" if is_buyer_maker else "buy",
                "signed_quantity": -quantity if is_buyer_maker else quantity,
                "source_market_type": "spot",
            }
        )
    return _ordered_frame(
        normalized,
        AGGREGATE_TRADES_COLUMNS,
        ["source_asset_id", "timestamp", "aggregate_trade_id"],
    )


def _normalize_depth(rows: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        timestamp = _timestamp_ms(row["timestamp"])
        symbol = str(row["symbol"])
        for side, raw_levels in (
            ("bid", row.get("bids", ())),
            ("ask", row.get("asks", ())),
        ):
            for level, raw_level in enumerate(raw_levels, start=1):
                price, quantity = raw_level
                normalized.append(
                    {
                        "source_asset_id": symbol,
                        "timestamp": timestamp,
                        "side": side,
                        "level": level,
                        "price": _float(price),
                        "quantity": _float(quantity),
                        "source_market_type": "spot",
                    }
                )
    return _ordered_frame(
        normalized,
        DEPTH_LEVELS_COLUMNS,
        ["source_asset_id", "timestamp", "side", "level"],
    )


def _interim_spec(data_id: str, columns: tuple[str, ...]) -> InterimDataSpec:
    if data_id == ids.aggregate_trades_data_id():
        primary_key = ("source_asset_id", "timestamp", "aggregate_trade_id")
        frequency = DataFrequency.TICK
        grain = DataGrain.TRADE_EVENT
    elif data_id == ids.depth_levels_data_id():
        primary_key = ("source_asset_id", "timestamp", "side", "level")
        frequency = DataFrequency.TICK
        grain = DataGrain.ORDER_BOOK_LEVEL
    else:
        primary_key = ("source_asset_id", "timestamp")
        frequency = DataFrequency.MINUTE
        grain = DataGrain.ASSET_TIMESTAMP
    return InterimDataSpec(
        data_id=data_id,
        source_id=ids.SOURCE_ID,
        frequency=frequency,
        grain=grain,
        column_roles=_column_roles(columns, primary_key=primary_key),
        primary_key=primary_key,
        partition_columns=("year",),
    )


def _column_roles(
    columns: tuple[str, ...],
    *,
    primary_key: tuple[str, ...],
) -> dict[str, ColumnRole]:
    return {column: field_role(column, primary_key=primary_key) for column in columns}


def _ordered_frame(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    sort_columns: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(sort_columns).reset_index(drop=True)


def _timestamp_ms(value: object) -> pd.Timestamp:
    return pd.to_datetime(int(value), unit="ms", utc=True).tz_localize(None)


def _float(value: object) -> float:
    return float(value)


def _first_present(row: dict[str, Any], *keys: str) -> object:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError(keys[0])


__all__ = [
    "BinanceInterimImportResult",
    "interim_catalog",
    "interim_specs",
    "iter_interim_frames",
    "materialize_interim_data",
    "normalize_raw_rows",
]
