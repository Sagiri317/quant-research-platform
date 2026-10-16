"""Strict Binance source-local validation helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant_research_platform.contracts.validation import strict_finite_float
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE


def validate_market_payload(
    market_type: str,
    data_kind: str,
    symbol: str,
    payload: object,
    *,
    source_schema: str = "qrp-binance-recorder",
    observed_time_ns: int | None = None,
    validate_values: bool = True,
) -> dict[str, Any]:
    """Validate one public stream payload without endpoint ownership."""

    if source_schema not in {"qrp-binance-recorder", "qrp-tardis-native-archive"}:
        raise ValueError("binance_native_source_schema_invalid")
    historical_time = None
    if source_schema == "qrp-tardis-native-archive":
        historical_time = (
            strict_non_negative_integer(
                observed_time_ns, error="binance_native_schema_time_invalid"
            )
            // 1_000_000
        )
    if not isinstance(payload, dict):
        raise ValueError("binance_recorder_payload_object_required")
    expected_event = "depthUpdate" if data_kind == "depth" else "aggTrade"
    if payload.get("e") != expected_event or payload.get("s") != symbol:
        raise ValueError("binance_recorder_market_payload_identity_invalid")
    strict_non_negative_integer(
        payload.get("E"), error="binance_recorder_event_time_invalid"
    )
    if market_type == FUTURES_MARKET_TYPE:
        # CM/UM integration added st/ps on 2026-06-30. Archive observation
        # time is bound by the native slice reader, never by download time.
        integrated = historical_time is None or historical_time >= 1782777600000
        if (
            (integrated or "st" in payload)
            and (type(payload.get("st")) is not int or payload["st"] != 1)
            or (data_kind == "depth" and (integrated or "ps" in payload))
            and payload.get("ps") != symbol
        ):
            raise ValueError("binance_recorder_market_payload_identity_invalid")
    if data_kind == "depth":
        for key in ("U", "u"):
            strict_non_negative_integer(
                payload.get(key), error="binance_recorder_depth_id_invalid"
            )
        if payload["U"] > payload["u"]:
            raise ValueError("binance_recorder_depth_id_range_invalid")
        if market_type == FUTURES_MARKET_TYPE:
            for key in ("T", "pu"):
                strict_non_negative_integer(
                    payload.get(key), error="binance_recorder_depth_id_invalid"
                )
        validate_depth_update_levels(
            payload.get("b"),
            error="binance_recorder_depth_bids_invalid",
            validate_values=validate_values,
        )
        validate_depth_update_levels(
            payload.get("a"),
            error="binance_recorder_depth_asks_invalid",
            validate_values=validate_values,
        )
        return payload
    for key in ("a", "f", "l", "T"):
        strict_non_negative_integer(
            payload.get(key), error="binance_recorder_trade_id_invalid"
        )
    if payload["l"] < payload["f"]:
        raise ValueError("binance_recorder_trade_id_range_invalid")
    if validate_values:
        strict_boolean(payload.get("m"), error="binance_recorder_trade_side_invalid")
    if (
        validate_values
        and market_type == FUTURES_MARKET_TYPE
        and (historical_time is None or historical_time >= 1767139200000)
        and "nq" not in payload
    ):
        raise ValueError("binance_recorder_trade_visible_quantity_invalid")
    if validate_values:
        strict_positive_float(
            payload.get("p"), error="binance_recorder_trade_price_invalid"
        )
        _visible_trade_quantity(market_type, payload, historical_time=historical_time)
    return payload


def _visible_trade_quantity(
    market_type: str, payload: dict, *, historical_time: int | None = None
) -> float | None:
    quantity = strict_positive_float(
        payload.get("q"), error="binance_recorder_trade_quantity_invalid"
    )
    if market_type != FUTURES_MARKET_TYPE:
        return quantity
    # nq first appeared on 2025-12-31; absence before that date is unknown,
    # not total quantity and not zero. A supplied nq always remains validated.
    if (
        "nq" not in payload
        and historical_time is not None
        and historical_time < 1767139200000
    ):
        return None
    visible = strict_non_negative_float(
        payload.get("nq"), error="binance_recorder_trade_visible_quantity_invalid"
    )
    if visible > quantity:
        raise ValueError("binance_recorder_trade_visible_quantity_invalid")
    return visible


def strict_positive_float(value: object, *, error: str) -> float:
    number = strict_finite_float(value, error=error)
    if number <= 0:
        raise ValueError(error)
    return number


def strict_non_negative_float(value: object, *, error: str) -> float:
    number = strict_finite_float(value, error=error)
    if number < 0:
        raise ValueError(error)
    return number


def strict_non_negative_integer(value: object, *, error: str) -> int:
    number = _strict_integer(value, error=error)
    if number < 0:
        raise ValueError(error)
    return number


def validate_depth_update_levels(
    value: object, *, error: str, validate_values: bool = True
) -> None:
    """Validate raw Binance ``[price, quantity]`` pairs without normalizing."""

    if not isinstance(value, list):
        raise ValueError(error)
    for level in value:
        if not isinstance(level, list) or len(level) != 2:
            raise ValueError(error)
        if validate_values:
            strict_positive_float(level[0], error=error)
            strict_non_negative_float(level[1], error=error)


def _strict_integer(value: object, *, error: str) -> int:
    if isinstance(value, bool):
        raise ValueError(error)
    if isinstance(value, int):
        return value
    raise ValueError(error)


def strict_boolean(value: object, *, error: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(error)
    return value


def strict_iso_timestamp(value: object, *, error: str) -> pd.Timestamp:
    if not isinstance(value, str) or not _has_utc_offset(value):
        raise ValueError(error)
    try:
        timestamp = pd.Timestamp(value).tz_convert("UTC")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if pd.isna(timestamp):
        raise ValueError(error)
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
