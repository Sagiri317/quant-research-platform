from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import IntEnum
from types import MappingProxyType

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


def test_string_subclasses_preserve_json_content_and_identity():
    class Version(str):
        pass

    payload = {Version("version"): Version("2.12.0+cu130")}
    assert parse_qrp_json_object(render_qrp_json_object(payload).encode()) == {
        "version": "2.12.0+cu130"
    }
    assert canonical_identity_bytes(payload) == canonical_identity_bytes(
        {"version": "2.12.0+cu130"}
    )


def test_typed_canonical_identity_keeps_types_distinct() -> None:
    values = (
        None,
        False,
        0,
        0.0,
        "0",
        ["x"],
        ("x",),
        {"x": 1},
    )
    assert len({canonical_identity_bytes(value) for value in values}) == len(values)
    assert canonical_json_hash({"b": 2, "a": 1}) == canonical_json_hash(
        {"a": 1, "b": 2}
    )


def test_typed_canonical_identity_cannot_be_mimicked_by_tag_shaped_content() -> None:
    encoded_integer = canonical_identity_bytes(17)
    tag_shaped_string = encoded_integer.decode("utf-8")

    assert canonical_identity_bytes(tag_shaped_string) != encoded_integer
    assert canonical_json_hash(tag_shaped_string) != canonical_json_hash(17)


def test_typed_canonical_identity_normalizes_only_approved_zero_forms() -> None:
    assert canonical_json_hash(-0.0) == canonical_json_hash(0.0)
    assert canonical_json_hash(1) != canonical_json_hash(1.0)
    assert canonical_json_hash(True) != canonical_json_hash(1)


def test_typed_canonical_identity_rejects_ambiguous_or_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="canonical_json_non_string_key"):
        canonical_json_hash({1: "value"})
    with pytest.raises(ValueError, match="canonical_json_nonfinite_number"):
        canonical_json_hash(float("inf"))


def test_typed_canonical_identity_does_not_normalize_unicode() -> None:
    assert canonical_json_hash("\u00e9") != canonical_json_hash("e\u0301")


def test_strict_json_rejects_cycles() -> None:
    cyclic: dict[str, object] = {"schema_id": "x"}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="qrp_json_cycle"):
        render_qrp_json_object(cyclic)
    with pytest.raises(ValueError, match="canonical_json_cycle"):
        canonical_identity_bytes(cyclic)


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
        b"source bytes",
        Decimal("1.25"),
        date(2026, 1, 1),
        datetime(2026, 1, 1),
        pd.Timestamp("2026-01-01"),
        pd.NaT,
        DecimalKind("1.0"),
        DateKind(2026, 1, 1),
        DatetimeKind(2026, 1, 1),
    )
    for value in noncanonical_identity_values:
        with pytest.raises(ValueError, match="canonical_json_unsupported_value"):
            canonical_identity_bytes(value)


def test_nested_mapping_identity_is_independent_of_key_order():
    first = {"é": {"aa": [1], "a": 1}, "量化": 2}
    reordered = {"量化": 2, "é": {"a": 1, "aa": [1]}}
    changed = {"é": {"aa": [2], "a": 1}, "量化": 2}
    assert canonical_json_hash(first) == canonical_json_hash(reordered)
    assert canonical_json_hash(first) != canonical_json_hash(changed)
