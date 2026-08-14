"""Local Binance raw JSONL envelope readers."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import pandas as pd

from quant_research_platform.runtime.resources import (
    RuntimeResourcePolicy,
    active_runtime_resource_policy,
    bind_runtime_resource_policy,
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance.validation import (
    first_present,
    strict_boolean,
    strict_data_kind,
    strict_iso_timestamp,
    strict_market_type,
    strict_millisecond_timestamp,
    strict_non_negative_float,
    strict_non_negative_integer,
    strict_positive_float,
    strict_symbol,
)

DATA_ID_BY_RAW_KIND = {
    "klines": "market_bars_1m",
    "agg_trades": "aggregate_trades_tick",
    "depth": "depth_levels_tick",
}
_SCAN_BUFFER_BYTES = 64 * 1024


def dataset_files_by_id(raw_root: Path) -> dict[str, tuple[Path, ...]]:
    root = Path(raw_root)
    return {
        data_id: tuple(sorted(_raw_kind_files(root, raw_kind)))
        for raw_kind, data_id in DATA_ID_BY_RAW_KIND.items()
    }


def iter_raw_envelopes(
    path: Path,
    *,
    resource_policy: RuntimeResourcePolicy | None = None,
) -> Iterator[dict[str, Any]]:
    captured_policy = current_runtime_resource_policy()
    if resource_policy is None and captured_policy is not None:
        raise RuntimeError("runtime_resource_policy_competing_root")
    bound_policy = bind_runtime_resource_policy(resource_policy)

    def activate() -> Iterator[dict[str, Any]]:
        policy = bound_policy or active_runtime_resource_policy()
        with runtime_resource_policy_scope(policy):
            for batch in _iter_raw_envelope_batches(
                path,
                batch_size=1,
                resource_policy=policy,
                validate_depth_payload=True,
            ):
                yield batch[0]

    return activate()


def _iter_raw_envelope_batches(
    path: Path,
    *,
    batch_size: int,
    resource_policy: RuntimeResourcePolicy,
    validate_depth_payload: bool = True,
) -> Iterator[tuple[dict[str, Any], ...]]:
    raw_path = Path(path)
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if raw_path.stat().st_size == 0:
        raise ValueError("empty_binance_raw_file")
    if raw_path.suffix != ".jsonl":
        raise ValueError("unsupported_binance_raw_file_extension")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("invalid_binance_raw_row_batch_size")
    yielded = False
    line_number = 0
    scan_bytes = min(
        _SCAN_BUFFER_BYTES,
        max(1, resource_policy.remaining_memory_bytes),
    )
    with resource_policy.reserve("source.binance.scan", scan_bytes):
        buffer = bytearray(scan_bytes)
        with raw_path.open("rb") as handle:
            while True:
                batch: list[dict[str, Any]] = []
                with ExitStack() as retained:
                    records_in_batch = 0
                    while records_in_batch < batch_size:
                        record_length = _next_record_length(handle, buffer)
                        if record_length == 0:
                            break
                        records_in_batch += 1
                        record_bound = _json_record_working_set_bytes(record_length)
                        try:
                            retained.enter_context(
                                resource_policy.reserve(
                                    "source.binance.record",
                                    record_bound,
                                )
                            )
                        except MemoryError:
                            if batch:
                                yield tuple(batch)
                                batch = []
                                break
                            raise
                        raw = handle.read(record_length)
                        line_number += 1
                        if raw.isspace():
                            continue
                        yielded = True
                        text = raw.decode("utf-8", errors="strict")
                        batch.append(
                            _validate_envelope(
                                _loads_json(
                                    text,
                                    raw_path,
                                    line_number=line_number,
                                ),
                                validate_depth_payload=validate_depth_payload,
                            )
                        )
                    if batch:
                        yield tuple(batch)
                    if record_length == 0:
                        break
    if not yielded:
        raise ValueError("empty_binance_raw_file")


def _next_record_length(handle: Any, buffer: bytearray) -> int:
    start = handle.tell()
    length = 0
    while True:
        count = handle.readinto(buffer)
        if count == 0:
            handle.seek(start)
            return length
        newline = buffer.find(b"\n", 0, count)
        if newline >= 0:
            length += newline + 1
            handle.seek(start)
            return length
        length += count


def _json_record_working_set_bytes(record_bytes: int) -> int:
    unit = max(
        sys.getsizeof({}),
        sys.getsizeof([]),
        sys.getsizeof(0),
        sys.getsizeof(""),
    )
    raw_and_text = 5 * record_bytes + sys.getsizeof(b"") + sys.getsizeof("")
    decoded_and_normalized = (record_bytes + 1) * unit
    return raw_and_text + decoded_and_normalized


def _raw_kind_files(root: Path, raw_kind: str) -> tuple[Path, ...]:
    if raw_kind == "klines":
        base = root / "spot" / "klines"
        patterns = ("*" + "/1m/*.jsonl",)
    elif raw_kind == "agg_trades":
        base = root / "spot" / "agg_trades"
        patterns = ("*" + "/*.jsonl",)
    elif raw_kind == "depth":
        base = root / "spot" / "depth"
        patterns = ("*" + "/*.jsonl",)
    else:
        raise ValueError(f"unsupported_binance_raw_kind: {raw_kind}")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(base.glob(pattern))
    return tuple(path for path in files if path.is_file())


def _loads_json(text: str, path: Path, *, line_number: int | None = None) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        location = f":{line_number}" if line_number is not None else ""
        raise ValueError(f"malformed_binance_raw_row: {path.name}{location}") from exc


def _validate_envelope(
    envelope: Any,
    *,
    validate_depth_payload: bool = True,
) -> dict[str, Any]:
    if not isinstance(envelope, dict):
        raise ValueError("binance_raw_envelope_required")
    required = {
        "source",
        "market_type",
        "data_kind",
        "symbol",
        "captured_at",
        "payload",
    }
    if not required.issubset(envelope):
        raise ValueError("binance_raw_envelope_required")
    if envelope["source"] != "binance":
        raise ValueError("invalid_binance_raw_envelope_source")
    strict_market_type(envelope["market_type"])
    data_kind = strict_data_kind(envelope["data_kind"])
    strict_symbol(envelope["symbol"], error="invalid_binance_raw_envelope_symbol")
    strict_iso_timestamp(
        envelope["captured_at"],
        error="invalid_binance_raw_envelope_captured_at",
    )
    payload = envelope["payload"]
    if data_kind == "klines":
        if envelope.get("interval") != "1m":
            raise ValueError("unsupported_binance_kline_interval")
        if not isinstance(payload, list) or len(payload) < 11:
            raise ValueError("invalid_binance_kline_payload")
        _validate_kline_payload(payload)
    elif data_kind == "agg_trades":
        if "interval" in envelope:
            raise ValueError("unexpected_binance_interval")
        if not isinstance(payload, dict):
            raise ValueError("invalid_binance_agg_trade_payload")
        _validate_agg_trade_payload(payload)
    elif data_kind == "depth":
        if "interval" in envelope:
            raise ValueError("unexpected_binance_interval")
        if not isinstance(payload, dict):
            raise ValueError("invalid_binance_depth_payload")
        if validate_depth_payload:
            _validate_depth_payload(payload)
    return envelope


def _validate_kline_payload(payload: list[object]) -> None:
    open_time = strict_millisecond_timestamp(
        payload[0],
        error="invalid_binance_kline_open_time",
    )
    close_time = strict_millisecond_timestamp(
        payload[6],
        error="invalid_binance_kline_close_time",
    )
    if open_time != open_time.floor("min"):
        raise ValueError("invalid_binance_kline_open_time_alignment")
    if close_time < open_time:
        raise ValueError("invalid_binance_kline_time_order")
    if close_time != open_time + pd.Timedelta(milliseconds=59_999):
        raise ValueError("invalid_binance_kline_close_time")
    open_price = strict_positive_float(payload[1], error="invalid_binance_kline_ohlc")
    high = strict_positive_float(payload[2], error="invalid_binance_kline_ohlc")
    low = strict_positive_float(payload[3], error="invalid_binance_kline_ohlc")
    close = strict_positive_float(payload[4], error="invalid_binance_kline_ohlc")
    if high < max(open_price, close) or low > min(open_price, close):
        raise ValueError("invalid_binance_kline_ohlc")
    volume = strict_non_negative_float(payload[5], error="invalid_binance_kline_volume")
    quote_volume = strict_non_negative_float(
        payload[7],
        error="invalid_binance_kline_volume",
    )
    taker_buy_volume = strict_non_negative_float(
        payload[9],
        error="invalid_binance_kline_volume",
    )
    taker_buy_quote_volume = strict_non_negative_float(
        payload[10],
        error="invalid_binance_kline_volume",
    )
    if taker_buy_volume > volume or taker_buy_quote_volume > quote_volume:
        raise ValueError("invalid_binance_kline_volume")
    strict_non_negative_integer(payload[8], error="invalid_binance_kline_trade_count")


def _validate_agg_trade_payload(payload: dict[str, Any]) -> None:
    strict_millisecond_timestamp(
        payload.get("timestamp", payload.get("T")),
        error="invalid_binance_agg_trade_timestamp",
    )
    strict_positive_float(
        first_present(payload, "price", "p"),
        error="invalid_binance_agg_trade_price",
    )
    strict_positive_float(
        first_present(payload, "quantity", "q"),
        error="invalid_binance_agg_trade_quantity",
    )
    first = strict_non_negative_integer(
        first_present(payload, "first_trade_id", "f"),
        error="invalid_binance_agg_trade_id_range",
    )
    last = strict_non_negative_integer(
        first_present(payload, "last_trade_id", "l"),
        error="invalid_binance_agg_trade_id_range",
    )
    aggregate = strict_non_negative_integer(
        first_present(payload, "aggregate_trade_id", "a"),
        error="invalid_binance_agg_trade_id_range",
    )
    strict_boolean(
        payload.get("is_buyer_maker", payload.get("m")),
        error="invalid_binance_agg_trade_side",
    )
    if aggregate < 0 or first < 0 or last < first:
        raise ValueError("invalid_binance_agg_trade_id_range")


def _validate_depth_payload(payload: dict[str, Any]) -> None:
    bids = _validate_depth_side(
        payload.get("bids", ()),
        side="bid",
        error="invalid_binance_depth_bid",
    )
    asks = _validate_depth_side(
        payload.get("asks", ()),
        side="ask",
        error="invalid_binance_depth_ask",
    )
    if bids is not None and asks is not None and bids >= asks:
        raise ValueError("crossed_binance_depth_book")


def _validate_depth_side(
    raw_levels: object,
    *,
    side: str,
    error: str,
) -> float | None:
    if raw_levels is None:
        return None
    if side not in {"bid", "ask"} or not isinstance(raw_levels, list | tuple):
        raise ValueError(error)
    first_price: float | None = None
    previous_price: float | None = None
    for raw_level in raw_levels:
        if not isinstance(raw_level, list | tuple) or len(raw_level) != 2:
            raise ValueError(error)
        price = strict_positive_float(raw_level[0], error=error)
        strict_non_negative_float(raw_level[1], error=error)
        if previous_price is not None and (
            (side == "bid" and price >= previous_price)
            or (side == "ask" and price <= previous_price)
        ):
            raise ValueError(error)
        if first_price is None:
            first_price = price
        previous_price = price
    return first_price


__all__ = ["DATA_ID_BY_RAW_KIND", "dataset_files_by_id", "iter_raw_envelopes"]
