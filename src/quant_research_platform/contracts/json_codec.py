"""Strict internal JSON and typed canonical identity contracts."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Any, NoReturn

import pandas as pd

_LENGTH_BYTES = 8
_JSON_INDENT = 2


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

    return _render_native_json(_json_native_object(value))


def qrp_json_object_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    """Deeply snapshot native or frozen containers under the JSON contract."""
    return _json_native_object(value, allow_frozen=True)


def _qrp_json_snapshot_text(value: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    payload = qrp_json_object_snapshot(value)
    return payload, _render_native_json(payload)


def _render_native_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=_JSON_INDENT,
        sort_keys=True,
        allow_nan=False,
    )


def canonical_json_hash(value: object) -> str:
    """Hash one typed, collision-framed canonical identity value."""

    return hashlib.sha256(canonical_identity_bytes(value)).hexdigest()


def canonical_identity_bytes(value: object) -> bytes:
    """Encode one typed, collision-framed canonical identity."""
    active: set[int] = set()

    def encode(item: object) -> bytes:
        if item is None:
            return _frame(b"n", b"")
        if type(item) is bool:
            return _frame(b"t" if item else b"f", b"")
        if type(item) is int:
            return _frame(b"i", str(item).encode("ascii"))
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("canonical_json_nonfinite_number")
            return _frame(b"r", struct.pack(">d", 0.0 if item == 0.0 else item))
        if type(item) is str:
            return _frame(b"s", _utf8_bytes(item))
        if type(item) is bytes:
            return _frame(b"b", item)
        if item is pd.NaT:
            raise ValueError("canonical_json_invalid_timestamp")
        if type(item) is Decimal:
            if not item.is_finite():
                raise ValueError("canonical_json_nonfinite_number")
            parts = item.as_tuple()
            sign = parts.sign if any(parts.digits) else 0
            return _frame(
                b"d",
                bytes((sign,))
                + _uint64(len(parts.digits))
                + bytes(parts.digits)
                + _frame(b"e", str(parts.exponent).encode("ascii")),
            )
        if type(item) is pd.Timestamp:
            return _frame(b"p", _timestamp_payload(item))
        if type(item) is datetime:
            return _frame(b"z", _datetime_payload(item))
        if type(item) is date:
            return _frame(b"a", _date_payload(item))
        if type(item) not in {list, tuple, dict}:
            raise ValueError("canonical_json_unsupported_value")
        identity = id(item)
        if identity in active:
            raise ValueError("canonical_json_cycle")
        active.add(identity)
        try:
            if type(item) is dict:
                if any(type(key) is not str for key in item):
                    raise ValueError("canonical_json_non_string_key")
                keys = sorted(item, key=_utf8_bytes)
                payload = b"".join(
                    _frame(b"k", _utf8_bytes(key)) + encode(item[key]) for key in keys
                )
                tag = b"m"
            else:
                payload = b"".join(encode(child) for child in item)
                tag = b"l" if type(item) is list else b"u"
            return _frame(tag, _uint64(len(item)) + payload)
        finally:
            active.remove(identity)

    try:
        return encode(value)
    except RecursionError as exc:
        raise ValueError("canonical_json_recursion_limit") from exc


def _strict_object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("qrp_json_duplicate_key")
        result[key] = value
    return result


def _reject_json_constant(_: str) -> NoReturn:
    raise ValueError("qrp_json_nonfinite_number")


def _json_native_object(
    value: Mapping[str, Any], *, allow_frozen: bool = False
) -> dict[str, Any]:
    if type(value) is not dict and not (
        allow_frozen and type(value) is MappingProxyType
    ):
        raise ValueError("qrp_json_top_level_object_required")
    try:
        return _json_native_mapping(value, active=set(), allow_frozen=allow_frozen)
    except RecursionError as exc:
        raise ValueError("qrp_json_recursion_limit") from exc


def _json_native_mapping(
    value: Mapping[object, object],
    *,
    active: set[int],
    allow_frozen: bool = False,
) -> dict[str, Any]:
    identity = id(value)
    if identity in active:
        raise ValueError("qrp_json_cycle")
    active.add(identity)
    result: dict[str, Any] = {}
    try:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("qrp_json_non_string_key")
            if not key.isascii():
                _utf8_bytes(key)
            result[key] = _json_native_value(
                item,
                active=active,
                allow_frozen=allow_frozen,
            )
        return result
    finally:
        active.remove(identity)


def _json_native_value(
    value: object, *, active: set[int], allow_frozen: bool = False
) -> Any:
    kind = type(value)
    if kind is str:
        if not value.isascii():
            _utf8_bytes(value)
        return value
    if value is None or kind is bool:
        return value
    if kind is int:
        return value
    if kind is float:
        if not math.isfinite(value):
            raise ValueError("qrp_json_nonfinite_number")
        return value
    if kind is list or (allow_frozen and kind is tuple):
        identity = id(value)
        if identity in active:
            raise ValueError("qrp_json_cycle")
        active.add(identity)
        try:
            return [
                _json_native_value(item, active=active, allow_frozen=allow_frozen)
                for item in value
            ]
        finally:
            active.remove(identity)
    if kind is dict or (allow_frozen and kind is MappingProxyType):
        return _json_native_mapping(value, active=active, allow_frozen=allow_frozen)
    raise ValueError("qrp_json_unsupported_value")


def _frame(tag: bytes, payload: bytes) -> bytes:
    if len(tag) != 1:
        raise AssertionError("canonical_json_tag_must_be_one_byte")
    return tag + _uint64(len(payload)) + payload


def _uint64(value: int) -> bytes:
    if value < 0 or value >= 1 << (_LENGTH_BYTES * 8):
        raise ValueError("canonical_json_value_too_large")
    return value.to_bytes(_LENGTH_BYTES, byteorder="big", signed=False)


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
