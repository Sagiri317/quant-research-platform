from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import IntEnum
from itertools import permutations
from types import MappingProxyType
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.json_codec import (
    canonical_identity_bytes,
    canonical_json_hash,
    parse_qrp_json_object,
    render_qrp_json_object,
)


@pytest.mark.parametrize(
    "raw",
    (
        b'{"schema_id":"x","value":1,"value":1}',
        b'{"schema_id":"x","value":1,"value":2}',
        b'{"schema_id":"x","nested":{"key":1,"key":2}}',
    ),
)
def test_strict_parser_rejects_duplicate_keys_before_dict_construction(
    raw: bytes,
) -> None:
    with pytest.raises(ValueError, match="qrp_json_duplicate_key"):
        parse_qrp_json_object(raw)


@pytest.mark.parametrize("constant", (b"NaN", b"Infinity", b"-Infinity"))
def test_strict_parser_rejects_nonfinite_constants(constant: bytes) -> None:
    with pytest.raises(ValueError, match="qrp_json_nonfinite_number"):
        parse_qrp_json_object(b'{"schema_id":"x","value":' + constant + b"}")


def test_strict_parser_rejects_invalid_utf8_and_non_object_top_level() -> None:
    with pytest.raises(ValueError, match="qrp_json_invalid_utf8"):
        parse_qrp_json_object(b'{"schema_id":"x","value":"\xff"}')
    with pytest.raises(ValueError, match="qrp_json_top_level_object_required"):
        parse_qrp_json_object(b"[]")
    with pytest.raises(ValueError, match="qrp_json_invalid"):
        parse_qrp_json_object(b'{"schema_id":')


def test_strict_writer_is_deterministic_and_rejects_non_json_values() -> None:
    expected = '{\n  "a": 1,\n  "schema_id": "x",\n  "text": "\u91cf\u5316"\n}'
    assert (
        render_qrp_json_object({"text": "量化", "schema_id": "x", "a": 1}) == expected
    )
    with pytest.raises(ValueError, match="qrp_json_nonfinite_number"):
        render_qrp_json_object({"schema_id": "x", "value": float("nan")})
    with pytest.raises(ValueError, match="qrp_json_non_string_key"):
        render_qrp_json_object({"schema_id": "x", 1: "value"})
    with pytest.raises(ValueError, match="qrp_json_unsupported_value"):
        render_qrp_json_object({"schema_id": "x", "value": (1, 2)})


def test_typed_canonical_identity_keeps_types_distinct() -> None:
    timestamp = pd.Timestamp("2026-01-01", tz="UTC")
    values = (
        None,
        False,
        0,
        0.0,
        "0",
        b"0",
        Decimal("0"),
        date(2026, 1, 1),
        datetime(2026, 1, 1, tzinfo=UTC),
        timestamp,
        ["x"],
        ("x",),
        {"x": 1},
    )
    assert len({canonical_identity_bytes(value) for value in values}) == len(values)
    assert canonical_json_hash(timestamp) != canonical_json_hash(str(timestamp))
    assert canonical_json_hash({"b": 2, "a": 1}) == canonical_json_hash(
        {"a": 1, "b": 2}
    )


def test_unbounded_decimal_identity_remains_available_above_bounded_ceiling() -> None:
    value = Decimal("7" * 4_097)
    exponent = b"0"
    payload = (
        b"\x00"
        + (4_097).to_bytes(8, "big")
        + b"\x07" * 4_097
        + b"e"
        + len(exponent).to_bytes(8, "big")
        + exponent
    )
    expected = b"d" + len(payload).to_bytes(8, "big") + payload

    assert canonical_identity_bytes(value) == expected


def test_decimal_identity_has_stable_literal_golden() -> None:
    value = Decimal("12.50")
    exponent = b"-2"
    payload = (
        b"\x00"
        + (4).to_bytes(8, "big")
        + bytes((1, 2, 5, 0))
        + b"e"
        + len(exponent).to_bytes(8, "big")
        + exponent
    )
    expected = b"d" + len(payload).to_bytes(8, "big") + payload

    assert canonical_identity_bytes(value) == expected


def test_typed_canonical_identity_cannot_be_mimicked_by_tag_shaped_content() -> None:
    encoded_integer = canonical_identity_bytes(17)
    tag_shaped_string = encoded_integer.decode("utf-8")

    assert canonical_identity_bytes(encoded_integer) != encoded_integer
    assert canonical_identity_bytes(tag_shaped_string) != encoded_integer
    assert canonical_json_hash(encoded_integer) != canonical_json_hash(17)
    assert canonical_json_hash(tag_shaped_string) != canonical_json_hash(17)


def test_typed_canonical_identity_normalizes_only_approved_zero_forms() -> None:
    assert canonical_json_hash(-0.0) == canonical_json_hash(0.0)
    assert canonical_json_hash(Decimal("-0.00")) == canonical_json_hash(Decimal("0.00"))
    assert canonical_json_hash(1) != canonical_json_hash(1.0)
    assert canonical_json_hash(True) != canonical_json_hash(1)


def test_typed_canonical_decimal_identity_preserves_exponent_and_zero_scale() -> None:
    assert canonical_json_hash(Decimal("1.0")) != canonical_json_hash(Decimal("1.00"))
    assert canonical_json_hash(Decimal("-0.00")) != canonical_json_hash(Decimal("0.0"))


def test_typed_canonical_identity_rejects_ambiguous_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="canonical_json_non_string_key"):
        canonical_json_hash({1: "value"})
    with pytest.raises(ValueError, match="canonical_json_nonfinite_number"):
        canonical_json_hash(float("inf"))
    with pytest.raises(ValueError, match="canonical_json_nonfinite_number"):
        canonical_json_hash(Decimal("NaN"))


@pytest.mark.parametrize(
    "value",
    (
        Decimal("Infinity"),
        Decimal("-Infinity"),
        Decimal("NaN"),
        Decimal("sNaN"),
    ),
)
def test_typed_canonical_decimal_identity_rejects_every_nonfinite_form(
    value: Decimal,
) -> None:
    with pytest.raises(ValueError, match="canonical_json_nonfinite_number"):
        canonical_identity_bytes(value)


def test_typed_canonical_timestamp_identity_preserves_unit_and_timezone() -> None:
    nanoseconds = pd.Timestamp("2026-01-01T00:00:00", tz=ZoneInfo("UTC")).as_unit("ns")
    milliseconds = nanoseconds.as_unit("ms")
    fixed_offset = pd.Timestamp(
        "2026-01-01T00:00:00",
        tz=UTC,
    ).as_unit("ns")

    assert canonical_json_hash(nanoseconds) != canonical_json_hash(milliseconds)
    assert canonical_json_hash(nanoseconds) != canonical_json_hash(fixed_offset)


def test_typed_canonical_timestamp_datetime_and_string_are_distinct() -> None:
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    python_datetime = timestamp.to_pydatetime()
    text = timestamp.isoformat()

    identities = {
        canonical_identity_bytes(timestamp),
        canonical_identity_bytes(python_datetime),
        canonical_identity_bytes(text),
    }

    assert len(identities) == 3


def test_typed_canonical_identity_rejects_nat_sentinel() -> None:
    assert pd.Timestamp("NaT") is pd.NaT
    with pytest.raises(ValueError, match="canonical_json_invalid_timestamp"):
        canonical_identity_bytes(pd.NaT)


def test_typed_canonical_identity_does_not_normalize_unicode() -> None:
    assert canonical_json_hash("\u00e9") != canonical_json_hash("e\u0301")


def test_strict_json_uses_native_integer_conversion_limits() -> None:
    import sys

    limit = sys.get_int_max_str_digits()
    value = 10**4096
    payload = {"value": value}
    assert parse_qrp_json_object(render_qrp_json_object(payload).encode()) == payload
    assert canonical_identity_bytes(value)
    if limit:
        huge = 10**limit
        with pytest.raises(ValueError, match="integer string conversion"):
            render_qrp_json_object({"value": huge})
        with pytest.raises(ValueError, match="integer string conversion"):
            canonical_identity_bytes(huge)
    assert sys.get_int_max_str_digits() == limit


def test_strict_json_rejects_cycles_and_excessive_recursion() -> None:
    cyclic: dict[str, object] = {"schema_id": "x"}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="qrp_json_cycle"):
        render_qrp_json_object(cyclic)
    with pytest.raises(ValueError, match="canonical_json_cycle"):
        canonical_identity_bytes(cyclic)

    nested: object = None
    for _ in range(100):
        nested = [nested]
    assert parse_qrp_json_object(
        render_qrp_json_object({"nested": nested}).encode()
    ) == {"nested": nested}
    assert canonical_identity_bytes(nested)
    for _ in range(2000):
        nested = [nested]
    with pytest.raises(ValueError, match="qrp_json_recursion_limit"):
        render_qrp_json_object({"schema_id": "x", "nested": nested})
    with pytest.raises(ValueError, match="canonical_json_recursion_limit"):
        canonical_identity_bytes(nested)


def test_strict_json_requires_exact_native_scalar_and_container_types() -> None:
    class IntegerKind(IntEnum):
        ONE = 1

    class ListKind(list[object]):
        pass

    class TupleKind(tuple[object, ...]):
        pass

    class DictKind(dict[str, object]):
        pass

    class DecimalKind(Decimal):
        pass

    class DateKind(date):
        pass

    class DatetimeKind(datetime):
        pass

    for value in (IntegerKind.ONE, np.int64(1), np.bool_(True), np.float64(1.0)):
        with pytest.raises(ValueError, match="qrp_json_unsupported_value"):
            render_qrp_json_object({"schema_id": "x", "value": value})
        with pytest.raises(ValueError, match="canonical_json_unsupported_value"):
            canonical_identity_bytes(value)
    with pytest.raises(ValueError, match="qrp_json_unsupported_value"):
        render_qrp_json_object({"schema_id": "x", "value": ListKind()})

    noncanonical_identity_values = (
        ListKind((1,)),
        TupleKind((1,)),
        DictKind({"key": 1}),
        MappingProxyType({"key": 1}),
        DecimalKind("1.0"),
        DateKind(2026, 1, 1),
        DatetimeKind(2026, 1, 1),
    )
    for value in noncanonical_identity_values:
        with pytest.raises(ValueError, match="canonical_json_unsupported_value"):
            canonical_identity_bytes(value)


def _identity_frame(tag: bytes, payload: bytes) -> bytes:
    return tag + len(payload).to_bytes(8, "big") + payload


@pytest.mark.parametrize("keys", [("a", "b"), ("a", "aa"), ("é", "量化")])
def test_mapping_identity_matches_independent_framing(keys) -> None:
    first, second = keys
    integer = _identity_frame(b"i", b"1")
    nested = _identity_frame(
        b"m",
        (2).to_bytes(8, "big")
        + _identity_frame(b"k", b"a")
        + integer
        + _identity_frame(b"k", b"aa")
        + _identity_frame(b"l", (1).to_bytes(8, "big") + integer),
    )
    for value, encoded in (
        (2, _identity_frame(b"i", b"2")),
        ({"aa": [1], "a": 1}, nested),
    ):
        expected = _identity_frame(
            b"m",
            (2).to_bytes(8, "big")
            + _identity_frame(b"k", first.encode("utf-8"))
            + integer
            + _identity_frame(b"k", second.encode("utf-8"))
            + encoded,
        )
        for pairs in permutations(((first, 1), (second, value))):
            mapping = dict(pairs)
            assert canonical_identity_bytes(mapping) == expected
            assert canonical_json_hash(mapping) == hashlib.sha256(expected).hexdigest()
        if keys == ("a", "b") and value == 2:
            assert hashlib.sha256(expected).hexdigest() == (
                "651bc9a60bd86e22fe01cfc417fb52be35c9c8e4c3085aa4b82b1af05905f56c"
            )
