"""Strict internal JSON and typed canonical identity contracts."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

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
            parse_constant=_finite_json_float,
            parse_float=_finite_json_float,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("qrp_json_invalid") from exc
    except RecursionError as exc:
        raise ValueError("qrp_json_recursion_limit") from exc
    if not isinstance(value, dict):
        raise ValueError("qrp_json_top_level_object_required")
    return value


def render_qrp_json_object(value: Mapping[str, Any]) -> str:
    """Render one deterministic strict QRP-owned JSON object."""

    return _render_native_json(_json_native_object(value))


def qrp_json_object_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    """Deeply snapshot native or frozen containers under the JSON contract."""
    return _json_native_object(value, allow_frozen=True)


def qrp_json_mapping_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt general mappings without relaxing the persisted JSON boundary."""
    if not isinstance(value, Mapping):
        raise ValueError("qrp_json_top_level_object_required")
    try:
        return _json_native_mapping(
            value, active=set(), allow_frozen=True, allow_mapping=True
        )
    except RecursionError as exc:
        raise ValueError("qrp_json_recursion_limit") from exc


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
    """Encode JSON values and frozen tuples without conflating scalar types."""
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
        if isinstance(item, str):
            return _frame(b"s", _utf8_bytes(str(item)))
        if type(item) not in {list, tuple, dict}:
            raise ValueError("canonical_json_unsupported_value")
        identity = id(item)
        if identity in active:
            raise ValueError("canonical_json_cycle")
        active.add(identity)
        try:
            if type(item) is dict:
                if any(not isinstance(key, str) for key in item):
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


def _finite_json_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("qrp_json_nonfinite_number")
    return value


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
    allow_mapping: bool = False,
) -> dict[str, Any]:
    identity = id(value)
    if identity in active:
        raise ValueError("qrp_json_cycle")
    active.add(identity)
    result: dict[str, Any] = {}
    try:
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("qrp_json_non_string_key")
            key = str(key)
            if not key.isascii():
                _utf8_bytes(key)
            result[key] = _json_native_value(
                item,
                active=active,
                allow_frozen=allow_frozen,
                allow_mapping=allow_mapping,
            )
        return result
    finally:
        active.remove(identity)


def _json_native_value(
    value: object,
    *,
    active: set[int],
    allow_frozen: bool = False,
    allow_mapping: bool = False,
) -> Any:
    kind = type(value)
    if isinstance(value, str):
        value = str(value)
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
                _json_native_value(
                    item,
                    active=active,
                    allow_frozen=allow_frozen,
                    allow_mapping=allow_mapping,
                )
                for item in value
            ]
        finally:
            active.remove(identity)
    if (
        kind is dict
        or (allow_frozen and kind is MappingProxyType)
        or (allow_mapping and isinstance(value, Mapping))
    ):
        return _json_native_mapping(
            value, active=active, allow_frozen=allow_frozen, allow_mapping=allow_mapping
        )
    raise ValueError("qrp_json_unsupported_value")


def _frame(tag: bytes, payload: bytes) -> bytes:
    return tag + _uint64(len(payload)) + payload


def _uint64(value: int) -> bytes:
    return value.to_bytes(_LENGTH_BYTES, byteorder="big", signed=False)


def _utf8_bytes(value: str) -> bytes:
    try:
        return value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("qrp_json_invalid_unicode") from exc


def _strict_config(config, required, optional=()):
    value = qrp_json_mapping_snapshot(config)
    if set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError("operation_config_fields_invalid")
    return value
