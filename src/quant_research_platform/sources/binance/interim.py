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
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.sources.binance import ids
from quant_research_platform.sources.binance.fields import field_role
from quant_research_platform.sources.binance.raw_exports import (
    dataset_files_by_id,
    iter_raw_envelopes,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout

RAW_ROW_BATCH_SIZE = 100_000

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
    row_batch_size: int = RAW_ROW_BATCH_SIZE,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Iterator[pd.DataFrame]:
    if row_batch_size <= 0:
        raise ValueError("invalid_binance_raw_row_batch_size")
    for input_file in sorted(input_files):
        for batch in _iter_batches(
            iter_raw_envelopes(input_file),
            batch_size=row_batch_size,
        ):
            frame = normalize_raw_envelopes(batch, data_id=data_id)
            frame = _filter_time_window(frame, start_date=start_date, end_date=end_date)
            if not frame.empty:
                yield frame


def normalize_raw_envelopes(
    envelopes: tuple[dict[str, Any], ...],
    *,
    data_id: str,
) -> pd.DataFrame:
    _validate_raw_kind_matches_data_id(envelopes, data_id=data_id)
    if data_id == ids.market_bars_data_id():
        return _normalize_klines(envelopes)
    if data_id == ids.aggregate_trades_data_id():
        return _normalize_aggregate_trades(envelopes)
    if data_id == ids.depth_levels_data_id():
        return _normalize_depth(envelopes)
    raise ValueError(f"unsupported_binance_data_id: {data_id}")


def materialize_interim_data(
    layout: StorageLayout,
    raw_root: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    data_ids: tuple[str, ...] = tuple(sorted(INTERIM_COLUMNS_BY_DATA_ID)),
    row_batch_size: int = RAW_ROW_BATCH_SIZE,
) -> BinanceInterimImportResult:
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
            iter_interim_frames(
                data_id=data_id,
                input_files=input_files,
                row_batch_size=row_batch_size,
                start_date=start_date,
                end_date=end_date,
            ),
            spec=spec,
        )
        written[data_id] = artifact_root
    return BinanceInterimImportResult(artifact_roots=written, skipped=tuple(skipped))


def _normalize_klines(rows: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    normalized = [
        {
            "source_asset_id": str(envelope["symbol"]),
            "timestamp": _timestamp_ms(envelope["payload"][0]),
            "open_time": _timestamp_ms(envelope["payload"][0]),
            "close_time": _timestamp_ms(envelope["payload"][6]),
            "open": _float(envelope["payload"][1]),
            "high": _float(envelope["payload"][2]),
            "low": _float(envelope["payload"][3]),
            "close": _float(envelope["payload"][4]),
            "volume": _float(envelope["payload"][5]),
            "quote_volume": _float(envelope["payload"][7]),
            "trade_count": int(envelope["payload"][8]),
            "taker_buy_volume": _float(envelope["payload"][9]),
            "taker_buy_quote_volume": _float(envelope["payload"][10]),
            "source_market_type": str(envelope["market_type"]),
            "interval": str(envelope["interval"]),
        }
        for envelope in rows
    ]
    return _ordered_frame(
        normalized,
        MARKET_BARS_COLUMNS,
        ["source_asset_id", "timestamp"],
    )


def _validate_raw_kind_matches_data_id(
    envelopes: tuple[dict[str, Any], ...],
    *,
    data_id: str,
) -> None:
    expected = {
        ids.market_bars_data_id(): "klines",
        ids.aggregate_trades_data_id(): "agg_trades",
        ids.depth_levels_data_id(): "depth",
    }.get(data_id)
    if expected is None:
        raise ValueError(f"unsupported_binance_data_id: {data_id}")
    for envelope in envelopes:
        if envelope.get("data_kind") != expected:
            raise ValueError("binance_raw_data_id_kind_mismatch")


def _filter_time_window(
    frame: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if frame.empty or (start_date is None and end_date is None):
        return frame
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    mask = pd.Series(True, index=frame.index)
    start, end, end_inclusive = _timestamp_filter_interval(
        start_date=start_date,
        end_date=end_date,
    )
    if start is not None:
        mask &= timestamps.ge(start)
    if end is not None:
        mask &= timestamps.le(end) if end_inclusive else timestamps.lt(end)
    return frame.loc[mask.fillna(False)].copy()


def _timestamp_filter_interval(
    *,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, bool]:
    if start_date is not None and end_date is not None:
        start, end, end_inclusive = time_window_timestamp_interval(
            TimeWindow(start_date, end_date)
        )
        return pd.Timestamp(start), pd.Timestamp(end), end_inclusive
    if start_date is not None:
        start = pd.Timestamp(start_date)
        return start, None, True
    assert end_date is not None
    end_window = TimeWindow(end_date, end_date)
    _start, end, end_inclusive = time_window_timestamp_interval(end_window)
    return None, pd.Timestamp(end), end_inclusive


def _normalize_aggregate_trades(rows: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    normalized = []
    for envelope in rows:
        payload = envelope["payload"]
        quantity = _float(_first_present(payload, "quantity", "q"))
        price = _float(_first_present(payload, "price", "p"))
        is_buyer_maker = bool(payload.get("is_buyer_maker", payload.get("m", False)))
        normalized.append(
            {
                "source_asset_id": str(envelope["symbol"]),
                "timestamp": _timestamp_ms(_first_present(payload, "timestamp", "T")),
                "aggregate_trade_id": int(
                    _first_present(payload, "aggregate_trade_id", "a")
                ),
                "price": price,
                "quantity": quantity,
                "quote_quantity": price * quantity,
                "first_trade_id": int(_first_present(payload, "first_trade_id", "f")),
                "last_trade_id": int(_first_present(payload, "last_trade_id", "l")),
                "is_buyer_maker": is_buyer_maker,
                "taker_side": "sell" if is_buyer_maker else "buy",
                "signed_quantity": -quantity if is_buyer_maker else quantity,
                "source_market_type": str(envelope["market_type"]),
            }
        )
    return _ordered_frame(
        normalized,
        AGGREGATE_TRADES_COLUMNS,
        ["source_asset_id", "timestamp", "aggregate_trade_id"],
    )


def _normalize_depth(rows: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for envelope in rows:
        payload = envelope["payload"]
        timestamp = _timestamp_iso(envelope["captured_at"])
        symbol = str(envelope["symbol"])
        for side, raw_levels in (
            ("bid", payload.get("bids", ())),
            ("ask", payload.get("asks", ())),
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
                        "source_market_type": str(envelope["market_type"]),
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


def _timestamp_iso(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True).tz_localize(None)


def _float(value: object) -> float:
    return float(value)


def _first_present(row: dict[str, Any], *keys: str) -> object:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError(keys[0])


def _iter_batches(
    rows: Iterator[dict[str, Any]],
    *,
    batch_size: int,
) -> Iterator[tuple[dict[str, Any], ...]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


__all__ = [
    "BinanceInterimImportResult",
    "interim_catalog",
    "interim_specs",
    "iter_interim_frames",
    "materialize_interim_data",
    "normalize_raw_envelopes",
]
