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
from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    bind_runtime_resource_policy,
    pandas_frame_payload_bytes,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance import ids
from quant_research_platform.sources.binance.fields import field_role
from quant_research_platform.sources.binance.raw_exports import (
    _iter_raw_envelope_batches,
    dataset_files_by_id,
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
from quant_research_platform.storage._dataset_writing import write_dataset
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
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[pd.DataFrame]:
    _strict_data_ids((data_id,))
    strict_positive_integer(
        row_batch_size,
        error="invalid_binance_raw_row_batch_size",
    )
    if isinstance(input_files, str) or not isinstance(input_files, tuple | list):
        raise ValueError("invalid_binance_raw_input_files")
    bound_policy = bind_runtime_resource_policy(
        resource_policy,
        allow_implicit_active=False,
    )
    policy = bound_policy or active_runtime_resource_policy()

    def activate() -> Iterator[pd.DataFrame]:
        with runtime_resource_policy_scope(policy):
            owner_baseline = policy.reserved_memory_bytes
            yielded = False
            empty_schema_frame: pd.DataFrame | None = None
            last_key: tuple[object, ...] | None = None
            for input_file in sorted(input_files):
                for batch in _iter_raw_envelope_batches(
                    input_file,
                    batch_size=row_batch_size,
                    resource_policy=policy,
                    validate_depth_payload=False,
                ):
                    normalized = normalize_raw_envelopes(batch, data_id=data_id)
                    last_key = _validate_normalized_boundary(
                        normalized,
                        data_id=data_id,
                        previous_key=last_key,
                    )
                    empty_schema_frame = normalized.iloc[:0].copy()
                    filtered = _filter_time_window(
                        normalized,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if not filtered.empty:
                        yielded = True
                        retained_bytes = policy.reserved_memory_bytes - owner_baseline
                        if pandas_frame_payload_bytes(filtered) > retained_bytes:
                            raise RuntimeError("binance_record_envelope_bound_invalid")
                        yield filtered
            if input_files and not yielded:
                empty = (
                    empty_schema_frame
                    if empty_schema_frame is not None
                    else _empty_interim_frame(data_id)
                )
                empty_bytes = pandas_frame_payload_bytes(empty)
                with policy.reserve("source.binance.empty", empty_bytes):
                    yield empty

    return activate()


def normalize_raw_envelopes(
    envelopes: tuple[dict[str, Any], ...],
    *,
    data_id: str,
) -> pd.DataFrame:
    _validate_raw_kind_matches_data_id(envelopes, data_id=data_id)
    if data_id == ids.market_bars_data_id():
        frame = _normalize_klines(envelopes)
    elif data_id == ids.aggregate_trades_data_id():
        frame = _normalize_aggregate_trades(envelopes)
    elif data_id == ids.depth_levels_data_id():
        frame = _normalize_depth(envelopes)
    else:
        raise ValueError(f"unsupported_binance_data_id: {data_id}")
    _validate_normalized_order(frame, data_id=data_id, previous_key=None)
    return frame


def materialize_interim_data(
    layout: StorageLayout,
    raw_root: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    data_ids: tuple[str, ...] = tuple(sorted(INTERIM_COLUMNS_BY_DATA_ID)),
    row_batch_size: int = RAW_ROW_BATCH_SIZE,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> BinanceInterimImportResult:
    bound_policy = bind_runtime_resource_policy(
        resource_policy,
        allow_implicit_active=False,
    )
    policy = bound_policy or active_runtime_resource_policy()
    with runtime_resource_policy_scope(policy):
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
                    resource_policy=policy,
                ),
                spec=spec,
                resource_policy=policy,
            )
            written[data_id] = artifact_root
        return BinanceInterimImportResult(
            artifact_roots=written,
            skipped=tuple(skipped),
        )


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
    return _interim_frame(
        normalized,
        MARKET_BARS_COLUMNS,
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


def _empty_interim_frame(data_id: str) -> pd.DataFrame:
    columns = INTERIM_COLUMNS_BY_DATA_ID.get(data_id)
    if columns is None:
        raise ValueError(f"unsupported_binance_data_id: {data_id}")
    return pd.DataFrame(columns=columns)


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
    return _interim_frame(
        normalized,
        AGGREGATE_TRADES_COLUMNS,
    )


def _normalize_depth(rows: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for envelope in rows:
        payload = envelope["payload"]
        timestamp = strict_iso_timestamp(
            envelope["captured_at"],
            error="invalid_binance_raw_envelope_captured_at",
        ).tz_localize(None)
        symbol = strict_symbol(envelope["symbol"])
        best_prices: dict[str, float] = {}
        for side, raw_levels in (
            ("ask", payload.get("asks", ())),
            ("bid", payload.get("bids", ())),
        ):
            levels = depth_side_levels(
                raw_levels,
                side=side,
                error=f"invalid_binance_depth_{side}",
            )
            if levels:
                best_prices[side] = levels[0][0]
            for level, (price, quantity) in enumerate(
                levels,
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
        if (
            "bid" in best_prices
            and "ask" in best_prices
            and best_prices["bid"] >= best_prices["ask"]
        ):
            raise ValueError("crossed_binance_depth_book")
    return _interim_frame(
        normalized,
        DEPTH_LEVELS_COLUMNS,
    )


def _interim_spec(data_id: str, columns: tuple[str, ...]) -> InterimDataSpec:
    primary_key = _interim_primary_key(data_id)
    if data_id == ids.aggregate_trades_data_id():
        frequency = DataFrequency.TICK
        grain = DataGrain.TRADE_EVENT
    elif data_id == ids.depth_levels_data_id():
        frequency = DataFrequency.TICK
        grain = DataGrain.ORDER_BOOK_LEVEL
    else:
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


def _interim_frame(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns).reset_index(drop=True)


def _validate_normalized_order(
    frame: pd.DataFrame,
    *,
    data_id: str,
    previous_key: tuple[object, ...] | None,
) -> tuple[object, ...] | None:
    primary_key = _interim_primary_key(data_id)
    last_key = previous_key
    for row in frame.loc[:, list(primary_key)].itertuples(index=False, name=None):
        key = tuple(row)
        if last_key is not None and key <= last_key:
            raise ValueError("out_of_order_binance_interim_dependency")
        last_key = key
    return last_key


def _validate_normalized_boundary(
    frame: pd.DataFrame,
    *,
    data_id: str,
    previous_key: tuple[object, ...] | None,
) -> tuple[object, ...] | None:
    if frame.empty:
        return previous_key
    primary_key = _interim_primary_key(data_id)
    first = tuple(frame.iloc[0].loc[list(primary_key)])
    if previous_key is not None and first <= previous_key:
        raise ValueError("out_of_order_binance_interim_dependency")
    return tuple(frame.iloc[-1].loc[list(primary_key)])


def _interim_primary_key(data_id: str) -> tuple[str, ...]:
    if data_id == ids.aggregate_trades_data_id():
        return ("source_asset_id", "timestamp", "aggregate_trade_id")
    if data_id == ids.depth_levels_data_id():
        return ("source_asset_id", "timestamp", "side", "level")
    if data_id == ids.market_bars_data_id():
        return ("source_asset_id", "timestamp")
    raise ValueError(f"unsupported_binance_data_id: {data_id}")


def _timestamp_ms(value: object) -> pd.Timestamp:
    return strict_millisecond_timestamp(
        value,
        error="invalid_binance_millisecond_timestamp",
    ).tz_localize(None)


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
