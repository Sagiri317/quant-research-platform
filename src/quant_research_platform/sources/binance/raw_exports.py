"""Local Binance raw JSONL envelope readers."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

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
    if envelope["market_type"] != "spot":
        raise ValueError("unsupported_binance_market_type")
    data_kind = envelope["data_kind"]
    if data_kind not in DATA_ID_BY_RAW_KIND:
        raise ValueError("unsupported_binance_data_kind")
    if not isinstance(envelope["symbol"], str) or not envelope["symbol"]:
        raise ValueError("invalid_binance_raw_envelope_symbol")
    _parse_captured_at(envelope["captured_at"])
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
    open_price = _positive_number(payload[1], error="invalid_binance_kline_ohlc")
    high = _positive_number(payload[2], error="invalid_binance_kline_ohlc")
    low = _positive_number(payload[3], error="invalid_binance_kline_ohlc")
    close = _positive_number(payload[4], error="invalid_binance_kline_ohlc")
    if high < max(open_price, close) or low > min(open_price, close):
        raise ValueError("invalid_binance_kline_ohlc")
    volume = _non_negative_number(payload[5], error="invalid_binance_kline_volume")
    quote_volume = _non_negative_number(
        payload[7],
        error="invalid_binance_kline_volume",
    )
    taker_buy_volume = _non_negative_number(
        payload[9],
        error="invalid_binance_kline_volume",
    )
    taker_buy_quote_volume = _non_negative_number(
        payload[10],
        error="invalid_binance_kline_volume",
    )
    if taker_buy_volume > volume or taker_buy_quote_volume > quote_volume:
        raise ValueError("invalid_binance_kline_volume")
    _non_negative_integer(payload[8], error="invalid_binance_kline_trade_count")


def _validate_agg_trade_payload(payload: dict[str, Any]) -> None:
    _positive_number(
        _first_present(payload, "price", "p"),
        error="invalid_binance_agg_trade_price",
    )
    _positive_number(
        _first_present(payload, "quantity", "q"),
        error="invalid_binance_agg_trade_quantity",
    )
    first = _non_negative_integer(
        _first_present(payload, "first_trade_id", "f"),
        error="invalid_binance_agg_trade_id_range",
    )
    last = _non_negative_integer(
        _first_present(payload, "last_trade_id", "l"),
        error="invalid_binance_agg_trade_id_range",
    )
    aggregate = _non_negative_integer(
        _first_present(payload, "aggregate_trade_id", "a"),
        error="invalid_binance_agg_trade_id_range",
    )
    if aggregate < 0 or first < 0 or last < first:
        raise ValueError("invalid_binance_agg_trade_id_range")


def _validate_depth_payload(payload: dict[str, Any]) -> None:
    bids = _validate_depth_side(
        payload.get("bids", ()),
        error="invalid_binance_depth_bid",
    )
    asks = _validate_depth_side(
        payload.get("asks", ()),
        error="invalid_binance_depth_ask",
    )
    if bids and asks and bids[0][0] >= asks[0][0]:
        raise ValueError("crossed_binance_depth_book")


def _validate_depth_side(
    raw_levels: object,
    *,
    error: str,
) -> list[tuple[float, float]]:
    if raw_levels is None:
        return []
    if not isinstance(raw_levels, list | tuple):
        raise ValueError(error)
    levels: list[tuple[float, float]] = []
    for raw_level in raw_levels:
        if not isinstance(raw_level, list | tuple) or len(raw_level) < 2:
            raise ValueError(error)
        price = _positive_number(raw_level[0], error=error)
        quantity = _non_negative_number(raw_level[1], error=error)
        levels.append((price, quantity))
    return levels


def _parse_captured_at(value: object) -> pd.Timestamp:
    if not isinstance(value, str) or not _has_utc_offset(value):
        raise ValueError("invalid_binance_raw_envelope_captured_at")
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError("invalid_binance_raw_envelope_captured_at")
    return timestamp


def _has_utc_offset(value: str) -> bool:
    if value.endswith("Z"):
        return True
    if len(value) < 6:
        return False
    offset = value[-6:]
    return (
        offset[0] in {"+", "-"}
        and offset[1:3].isdigit()
        and offset[3] == ":"
        and offset[4:6].isdigit()
    )


def _positive_number(value: object, *, error: str) -> float:
    number = _finite_float(value, error=error)
    if number <= 0:
        raise ValueError(error)
    return number


def _non_negative_number(value: object, *, error: str) -> float:
    number = _finite_float(value, error=error)
    if number < 0:
        raise ValueError(error)
    return number


def _non_negative_integer(value: object, *, error: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if number < 0:
        raise ValueError(error)
    return number


def _finite_float(value: object, *, error: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if not math.isfinite(number):
        raise ValueError(error)
    return number


def _first_present(row: dict[str, Any], *keys: str) -> object:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    raise ValueError("missing_binance_raw_payload_field")


__all__ = ["DATA_ID_BY_RAW_KIND", "dataset_files_by_id", "iter_raw_envelopes"]
