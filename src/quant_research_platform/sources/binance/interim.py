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
from quant_research_platform.sources.binance.validation import (
    depth_side_levels,
    first_present,
    strict_boolean,
    strict_data_kind,
    strict_iso_timestamp,
    strict_market_type,
    strict_millisecond_timestamp,
    strict_non_negative_float,
    strict_non_negative_integer,
    strict_positive_float,
    strict_positive_integer,
    strict_symbol,
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
    strict_positive_integer(
        row_batch_size,
        error="invalid_binance_raw_row_batch_size",
    )
    if isinstance(input_files, str) or not isinstance(input_files, tuple | list):
        raise ValueError("invalid_binance_raw_input_files")
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
    requested_data_ids = _strict_data_ids(data_ids)
    for data_id in requested_data_ids:
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
            "source_asset_id": strict_symbol(envelope["symbol"]),
            "timestamp": _timestamp_ms(envelope["payload"][0]),
            "open_time": _timestamp_ms(envelope["payload"][0]),
            "close_time": _timestamp_ms(envelope["payload"][6]),
            "open": strict_positive_float(
                envelope["payload"][1],
                error="invalid_binance_kline_ohlc",
            ),
            "high": strict_positive_float(
                envelope["payload"][2],
                error="invalid_binance_kline_ohlc",
            ),
            "low": strict_positive_float(
                envelope["payload"][3],
                error="invalid_binance_kline_ohlc",
            ),
            "close": strict_positive_float(
                envelope["payload"][4],
                error="invalid_binance_kline_ohlc",
            ),
            "volume": strict_non_negative_float(
                envelope["payload"][5],
                error="invalid_binance_kline_volume",
            ),
            "quote_volume": strict_non_negative_float(
                envelope["payload"][7],
                error="invalid_binance_kline_volume",
            ),
            "trade_count": strict_non_negative_integer(
                envelope["payload"][8],
                error="invalid_binance_kline_trade_count",
            ),
            "taker_buy_volume": strict_non_negative_float(
                envelope["payload"][9],
                error="invalid_binance_kline_volume",
            ),
            "taker_buy_quote_volume": strict_non_negative_float(
                envelope["payload"][10],
                error="invalid_binance_kline_volume",
            ),
            "source_market_type": strict_market_type(envelope["market_type"]),
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
        if strict_data_kind(envelope.get("data_kind")) != expected:
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
        start, _end, _end_inclusive = time_window_timestamp_interval(
            TimeWindow(start_date, start_date)
        )
        return start, None, True
    assert end_date is not None
    end_window = TimeWindow(end_date, end_date)
    _start, end, end_inclusive = time_window_timestamp_interval(end_window)
    return None, pd.Timestamp(end), end_inclusive


def _normalize_aggregate_trades(rows: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    normalized = []
    for envelope in rows:
        payload = envelope["payload"]
        quantity = strict_positive_float(
            first_present(payload, "quantity", "q"),
            error="invalid_binance_agg_trade_quantity",
        )
        price = strict_positive_float(
            first_present(payload, "price", "p"),
            error="invalid_binance_agg_trade_price",
        )
        is_buyer_maker = strict_boolean(
            payload.get("is_buyer_maker", payload.get("m")),
            error="invalid_binance_agg_trade_side",
        )
        normalized.append(
            {
                "source_asset_id": strict_symbol(envelope["symbol"]),
                "timestamp": _timestamp_ms(first_present(payload, "timestamp", "T")),
                "aggregate_trade_id": strict_non_negative_integer(
                    first_present(payload, "aggregate_trade_id", "a"),
                    error="invalid_binance_agg_trade_id_range",
                ),
                "price": price,
                "quantity": quantity,
                "quote_quantity": price * quantity,
                "first_trade_id": strict_non_negative_integer(
                    first_present(payload, "first_trade_id", "f"),
                    error="invalid_binance_agg_trade_id_range",
                ),
                "last_trade_id": strict_non_negative_integer(
                    first_present(payload, "last_trade_id", "l"),
                    error="invalid_binance_agg_trade_id_range",
                ),
                "is_buyer_maker": is_buyer_maker,
                "taker_side": "sell" if is_buyer_maker else "buy",
                "signed_quantity": -quantity if is_buyer_maker else quantity,
                "source_market_type": strict_market_type(envelope["market_type"]),
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
        symbol = strict_symbol(envelope["symbol"])
        for side, raw_levels in (
            ("bid", payload.get("bids", ())),
            ("ask", payload.get("asks", ())),
        ):
            for level, (price, quantity) in enumerate(
                depth_side_levels(
                    raw_levels,
                    side=side,
                    error=f"invalid_binance_depth_{side}",
                ),
                start=1,
            ):
                normalized.append(
                    {
                        "source_asset_id": symbol,
                        "timestamp": timestamp,
                        "side": side,
                        "level": level,
                        "price": price,
                        "quantity": quantity,
                        "source_market_type": strict_market_type(
                            envelope["market_type"]
                        ),
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
    return strict_millisecond_timestamp(
        value,
        error="invalid_binance_millisecond_timestamp",
    ).tz_localize(None)


def _timestamp_iso(value: object) -> pd.Timestamp:
    return strict_iso_timestamp(
        value,
        error="invalid_binance_raw_envelope_captured_at",
    ).tz_localize(None)


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


def _strict_data_ids(data_ids: object) -> tuple[str, ...]:
    if isinstance(data_ids, str) or not isinstance(data_ids, tuple | list):
        raise ValueError("invalid_binance_interim_data_ids")
    values = tuple(data_ids)
    if any(not isinstance(data_id, str) for data_id in values):
        raise ValueError("invalid_binance_interim_data_ids")
    unknown = sorted(set(values).difference(INTERIM_COLUMNS_BY_DATA_ID))
    if unknown:
        raise ValueError("unsupported_binance_data_id")
    return values


__all__ = [
    "BinanceInterimImportResult",
    "interim_catalog",
    "interim_specs",
    "iter_interim_frames",
    "materialize_interim_data",
    "normalize_raw_envelopes",
]
