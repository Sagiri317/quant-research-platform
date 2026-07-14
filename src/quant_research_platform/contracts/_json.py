"""Strict internal JSON and typed canonical identity contracts."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Callable, Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any, NoReturn

import pandas as pd

_LENGTH_BYTES = 8
_MAX_INTEGER_DECIMAL_DIGITS = 4_096
_MAX_INTEGER_BITS = 13_607
_MAX_NESTING_DEPTH = 64
_RETIRED_JSON_IDENTITY_KEYS = frozenset({"contract_version", "schema_version"})


def parse_qrp_json_object(raw: bytes) -> dict[str, Any]:
    """Decode one strict UTF-8 QRP-owned JSON object."""

    if not isinstance(raw, bytes):
        raise TypeError("qrp_json_bytes_required")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("qrp_json_invalid_utf8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object_from_pairs,
            parse_constant=_reject_json_constant,
            parse_int=_parse_json_integer,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("qrp_json_invalid") from exc
    except RecursionError as exc:
        raise ValueError("qrp_json_recursion_limit") from exc
    if not isinstance(value, dict):
        raise ValueError("qrp_json_top_level_object_required")
    return _json_native_object(value)


def render_qrp_json_object(value: Mapping[str, Any]) -> str:
    """Render one deterministic strict QRP-owned JSON object."""

    payload = _json_native_object(value)
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def canonical_json_hash(value: object) -> str:
    """Hash one typed, collision-framed canonical identity value."""

    return hashlib.sha256(canonical_identity_bytes(value)).hexdigest()


def canonical_identity_bytes(value: object) -> bytes:
    """Return the typed canonical identity encoding for ``value``."""

    try:
        return _canonical_identity_bytes(value, active=set(), depth=0)
    except RecursionError as exc:
        raise ValueError("canonical_json_recursion_limit") from exc


def _canonical_identity_bytes(
    value: object,
    *,
    active: set[int],
    depth: int,
) -> bytes:
    if depth > _MAX_NESTING_DEPTH:
        raise ValueError("canonical_json_recursion_limit")

    if value is None:
        return _frame(b"n", b"")
    if value is False:
        return _frame(b"f", b"")
    if value is True:
        return _frame(b"t", b"")
    if type(value) is int:
        _validate_integer_size(value, error="canonical_json_integer_too_large")
        return _frame(b"i", str(value).encode("ascii"))
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical_json_nonfinite_number")
        normalized = 0.0 if value == 0.0 else value
        return _frame(b"r", struct.pack(">d", normalized))
    if type(value) is str:
        return _frame(b"s", _utf8_bytes(value))
    if type(value) is bytes:
        return _frame(b"b", value)
    if value is pd.NaT:
        raise ValueError("canonical_json_invalid_timestamp")
    if type(value) is Decimal:
        return _frame(b"d", _decimal_payload(value))
    if type(value) is pd.Timestamp:
        return _frame(b"p", _timestamp_payload(value))
    if type(value) is datetime:
        return _frame(b"z", _datetime_payload(value))
    if type(value) is date:
        return _frame(b"a", _date_payload(value))
    if type(value) is list:
        return _with_active_container(
            value,
            active=active,
            error="canonical_json_cycle",
            build=lambda: _frame(
                b"l",
                _sequence_payload(value, active=active, depth=depth + 1),
            ),
        )
    if type(value) is tuple:
        return _with_active_container(
            value,
            active=active,
            error="canonical_json_cycle",
            build=lambda: _frame(
                b"u",
                _sequence_payload(value, active=active, depth=depth + 1),
            ),
        )
    if type(value) is dict:
        return _with_active_container(
            value,
            active=active,
            error="canonical_json_cycle",
            build=lambda: _frame(
                b"m",
                _mapping_payload(value, active=active, depth=depth + 1),
            ),
        )
    raise ValueError("canonical_json_unsupported_value")


def _strict_object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("qrp_json_duplicate_key")
        result[key] = value
    return result


def _reject_json_constant(_: str) -> NoReturn:
    raise ValueError("qrp_json_nonfinite_number")


def _parse_json_integer(value: str) -> int:
    if len(value.lstrip("-")) > _MAX_INTEGER_DECIMAL_DIGITS:
        raise ValueError("qrp_json_integer_too_large")
    return int(value)


def _json_native_object(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("qrp_json_top_level_object_required")
    return _json_native_mapping(value, active=set(), depth=0)


def _json_native_mapping(
    value: Mapping[object, object],
    *,
    active: set[int],
    depth: int,
) -> dict[str, Any]:
    if depth > _MAX_NESTING_DEPTH:
        raise ValueError("qrp_json_recursion_limit")
    identity = id(value)
    if identity in active:
        raise ValueError("qrp_json_cycle")
    active.add(identity)
    result: dict[str, Any] = {}
    try:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("qrp_json_non_string_key")
            _utf8_bytes(key)
            if key in _RETIRED_JSON_IDENTITY_KEYS:
                raise ValueError("qrp_json_retired_identity_key")
            result[key] = _json_native_value(
                item,
                active=active,
                depth=depth + 1,
            )
        return result
    finally:
        active.remove(identity)


def _json_native_value(value: object, *, active: set[int], depth: int) -> Any:
    if depth > _MAX_NESTING_DEPTH:
        raise ValueError("qrp_json_recursion_limit")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        _validate_integer_size(value, error="qrp_json_integer_too_large")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("qrp_json_nonfinite_number")
        return value
    if type(value) is str:
        _utf8_bytes(value)
        return value
    if type(value) is list:
        identity = id(value)
        if identity in active:
            raise ValueError("qrp_json_cycle")
        active.add(identity)
        try:
            return [
                _json_native_value(item, active=active, depth=depth + 1)
                for item in value
            ]
        finally:
            active.remove(identity)
    if type(value) is dict:
        return _json_native_mapping(value, active=active, depth=depth)
    raise ValueError("qrp_json_unsupported_value")


def _frame(tag: bytes, payload: bytes) -> bytes:
    if len(tag) != 1:
        raise AssertionError("canonical_json_tag_must_be_one_byte")
    return tag + _uint64(len(payload)) + payload


def _uint64(value: int) -> bytes:
    if value < 0 or value >= 1 << (_LENGTH_BYTES * 8):
        raise ValueError("canonical_json_value_too_large")
    return value.to_bytes(_LENGTH_BYTES, byteorder="big", signed=False)


def _sequence_payload(
    value: list[object] | tuple[object, ...],
    *,
    active: set[int],
    depth: int,
) -> bytes:
    return _uint64(len(value)) + b"".join(
        _canonical_identity_bytes(item, active=active, depth=depth) for item in value
    )


def _mapping_payload(
    value: Mapping[object, object],
    *,
    active: set[int],
    depth: int,
) -> bytes:
    entries: list[tuple[bytes, object]] = []
    for key, item in value.items():
        if type(key) is not str:
            raise ValueError("canonical_json_non_string_key")
        entries.append((_utf8_bytes(key), item))
    entries.sort(key=lambda item: item[0])
    return _uint64(len(entries)) + b"".join(
        _frame(b"k", key)
        + _canonical_identity_bytes(item, active=active, depth=depth)
        for key, item in entries
    )


def _with_active_container(
    value: object,
    *,
    active: set[int],
    error: str,
    build: Callable[[], bytes],
) -> bytes:
    identity = id(value)
    if identity in active:
        raise ValueError(error)
    active.add(identity)
    try:
        return build()
    finally:
        active.remove(identity)


def _validate_integer_size(value: int, *, error: str) -> None:
    if value.bit_length() > _MAX_INTEGER_BITS:
        raise ValueError(error)
    if len(str(abs(value))) > _MAX_INTEGER_DECIMAL_DIGITS:
        raise ValueError(error)


def _decimal_payload(value: Decimal) -> bytes:
    if not value.is_finite():
        raise ValueError("canonical_json_nonfinite_number")
    decimal_tuple = value.as_tuple()
    is_zero = not any(decimal_tuple.digits)
    sign = 0 if is_zero else int(decimal_tuple.sign)
    digits = bytes(decimal_tuple.digits)
    return (
        bytes((sign,))
        + _uint64(len(digits))
        + digits
        + _frame(b"e", str(decimal_tuple.exponent).encode("ascii"))
    )


def _timestamp_payload(value: pd.Timestamp) -> bytes:
    if pd.isna(value):
        raise ValueError("canonical_json_invalid_timestamp")
    unit = value.unit
    integer_value = int(value.asm8.view("i8"))
    return b"".join(
        (
            _frame(b"q", unit.encode("ascii")),
            _frame(b"v", str(integer_value).encode("ascii")),
            _frame(b"o", _timezone_payload(value.tzinfo, value)),
        )
    )


def _datetime_payload(value: datetime) -> bytes:
    if pd.isna(value):
        raise ValueError("canonical_json_invalid_timestamp")
    components = (
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        value.fold,
    )
    return b"".join(
        (
            _frame(
                b"c",
                b",".join(str(item).encode("ascii") for item in components),
            ),
            _frame(b"o", _timezone_payload(value.tzinfo, value)),
        )
    )


def _date_payload(value: date) -> bytes:
    return b",".join(
        str(item).encode("ascii") for item in (value.year, value.month, value.day)
    )


def _timezone_payload(tzinfo: object | None, value: datetime) -> bytes:
    if tzinfo is None:
        return b""
    timezone_type = type(tzinfo)
    offset = value.utcoffset()
    offset_microseconds = (
        None if offset is None else int(offset.total_seconds() * 1_000_000)
    )
    name = value.tzname()
    return b"".join(
        (
            _frame(b"j", _utf8_bytes(timezone_type.__module__)),
            _frame(b"h", _utf8_bytes(timezone_type.__qualname__)),
            _frame(b"x", _utf8_bytes(str(tzinfo))),
            _frame(
                b"w",
                (
                    b""
                    if offset_microseconds is None
                    else str(offset_microseconds).encode("ascii")
                ),
            ),
            _frame(b"y", b"" if name is None else _utf8_bytes(name)),
        )
    )


def _utf8_bytes(value: str) -> bytes:
    try:
        return value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("qrp_json_invalid_unicode") from exc


__all__ = [
    "canonical_identity_bytes",
    "canonical_json_hash",
    "parse_qrp_json_object",
    "render_qrp_json_object",
]
