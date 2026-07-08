"""Local Binance raw JSONL envelope readers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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
    strict_symbol,
)

DATA_ID_BY_RAW_KIND = {
    "klines": "market_bars_1m",
    "agg_trades": "aggregate_trades_tick",
    "depth": "depth_levels_tick",
}


def dataset_files_by_id(raw_root: Path) -> dict[str, tuple[Path, ...]]:
    root = Path(raw_root)
    return {
        data_id: tuple(sorted(_raw_kind_files(root, raw_kind)))
        for raw_kind, data_id in DATA_ID_BY_RAW_KIND.items()
    }


def iter_raw_envelopes(path: Path) -> Iterator[dict[str, Any]]:
    raw_path = Path(path)
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if raw_path.stat().st_size == 0:
        raise ValueError("empty_binance_raw_file")
    if raw_path.suffix == ".jsonl":
        yielded = False
        with raw_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                yielded = True
                yield _validate_envelope(
                    _loads_object(line, raw_path, line_number=line_number)
                )
        if not yielded:
            raise ValueError("empty_binance_raw_file")
        return
    raise ValueError("unsupported_binance_raw_file_extension")


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


def _loads_object(text: str, path: Path, *, line_number: int) -> Any:
    return _loads_json(text, path, line_number=line_number)


def _loads_json(text: str, path: Path, *, line_number: int | None = None) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        location = f":{line_number}" if line_number is not None else ""
        raise ValueError(f"malformed_binance_raw_row: {path.name}{location}") from exc


def _validate_envelope(envelope: Any) -> dict[str, Any]:
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
    if close_time < open_time:
        raise ValueError("invalid_binance_kline_time_order")
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
    if bids and asks and bids[0][0] >= asks[0][0]:
        raise ValueError("crossed_binance_depth_book")


def _validate_depth_side(
    raw_levels: object,
    *,
    side: str,
    error: str,
) -> list[tuple[float, float]]:
    return depth_side_levels(raw_levels, side=side, error=error)


__all__ = ["DATA_ID_BY_RAW_KIND", "dataset_files_by_id", "iter_raw_envelopes"]
