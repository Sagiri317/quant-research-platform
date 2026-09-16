from __future__ import annotations

import hashlib
import importlib
import sys
from datetime import UTC, date, datetime
from decimal import MAX_EMAX, MIN_EMIN, Decimal
from enum import IntEnum
from itertools import permutations
from types import MappingProxyType
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.json_codec import (
    bounded_rendered_qrp_json_object_size,
    canonical_identity_bytes,
    canonical_json_hash,
    json_record_working_set_bytes,
    parse_qrp_json_object,
    render_qrp_json_object,
)


def test_json_record_working_set_bound_validates_bytes_and_grows() -> None:
    assert json_record_working_set_bytes(2) > json_record_working_set_bytes(1) > 1
    for value in (True, -1, 1.0):
        with pytest.raises(ValueError, match="json_record_bytes_invalid"):
            json_record_working_set_bytes(value)  # type: ignore[arg-type]


def test_bounded_strict_renderer_matches_existing_renderer_for_lazy_arrays() -> None:
    categories = [
        {"type": "str", "value": "量化"},
        {"type": "int", "value": 1},
        {"type": "bytes", "value": "ff"},
    ]
    skeleton = {
        "schema_id": "qrp-dataset-part-logical-dtypes",
        "columns": [
            {
                "name": "asset_id",
                "kind": "categorical",
                "ordered": False,
                "categories": [],
            },
            {"name": "value", "kind": "numpy_float", "dtype": "float64"},
        ],
    }
    full = {
        **skeleton,
        "columns": [
            {**skeleton["columns"][0], "categories": categories},
            skeleton["columns"][1],
        ],
    }
    expected_size = len(render_qrp_json_object(full).encode("utf-8"))

    assert (
        bounded_rendered_qrp_json_object_size(
            skeleton,
            lazy_arrays={("columns", 0, "categories"): iter(categories)},
            max_bytes=expected_size,
        )
        == expected_size
    )


def test_bounded_strict_renderer_stops_before_lazy_array_tail() -> None:
    skeleton = {"schema_id": "x", "values": []}
    seen = 0

    def values():
        nonlocal seen
        for index in range(10):
            seen += 1
            if index == 3:
                raise AssertionError("lazy_array_tail_materialized")
            yield {"index": index, "value": "x" * 128}

    one_item_size = len(
        render_qrp_json_object(
            {"schema_id": "x", "values": [{"index": 0, "value": "x" * 128}]}
        ).encode("utf-8")
    )
    with pytest.raises(ValueError, match="qrp_json_render_exceeds_max_bytes"):
        bounded_rendered_qrp_json_object_size(
            skeleton,
            lazy_arrays={("values",): values()},
            max_bytes=one_item_size,
        )

    assert seen == 2


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


def test_canonical_identity_writer_matches_existing_bytes_and_exact_peak_gate() -> None:
    json_owner = importlib.import_module("quant_research_platform.contracts.json_codec")
    writer_type = json_owner.CanonicalIdentityWriter
    value = ("nested", [None, 1, "x"], {"b": 2, "a": 1})
    expected = canonical_identity_bytes(value)

    writer = writer_type(max_bytes=len(expected) * 2)
    writer.value(value)
    assert writer.finish() == expected

    writer = writer_type(max_bytes=len(expected) * 2 - 1)
    writer.value(value)
    with pytest.raises(ValueError, match="canonical_identity_exceeds_max_bytes"):
        writer.finish()


def test_bounded_direct_mapping_rejects_before_copying_all_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_owner = importlib.import_module("quant_research_platform.contracts.json_codec")
    value = {f"key-{index:05d}": index for index in range(20_000)}
    encoded_keys = 0
    utf8_bytes = json_owner._utf8_bytes

    def bounded_utf8(item: str) -> bytes:
        nonlocal encoded_keys
        encoded_keys += 1
        if encoded_keys > 64:
            raise AssertionError("bounded_writer_copied_all_mapping_keys")
        return utf8_bytes(item)

    monkeypatch.setattr(json_owner, "_utf8_bytes", bounded_utf8)
    writer = json_owner.CanonicalIdentityWriter(max_bytes=1024)
    with pytest.raises(ValueError, match="canonical_identity_exceeds_max_bytes"):
        writer.value(value)
    assert encoded_keys <= 64


@pytest.mark.parametrize(
    "case",
    (
        "ascii-value",
        "non-ascii-value",
        "ascii-key",
        "non-ascii-key",
    ),
)
def test_bounded_writer_rejects_large_text_before_full_utf8_copy(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    json_owner = importlib.import_module("quant_research_platform.contracts.json_codec")
    utf8_bytes = json_owner._utf8_bytes
    text = "指标" * 4_000_000 if case.startswith("non-ascii") else "x" * 8_000_000
    value: object = {text: None} if case.endswith("key") else text

    def bounded_utf8(item: str) -> bytes:
        if len(item) > 1024:
            raise AssertionError("bounded_writer_preallocated_full_utf8_payload")
        return utf8_bytes(item)

    monkeypatch.setattr(json_owner, "_utf8_bytes", bounded_utf8)
    writer = json_owner.CanonicalIdentityWriter(max_bytes=1024)
    with pytest.raises(ValueError, match="canonical_identity_exceeds_max_bytes"):
        writer.value(value)


@pytest.mark.parametrize(
    "value",
    (
        Decimal("7" * 4_097),
        Decimal("1" + "0" * 4_096),
        Decimal("7" * 1_000_000),
    ),
    ids=("ceiling-plus-one", "ceiling-plus-one-trailing-zeros", "intermediate"),
)
def test_bounded_writer_rejects_oversized_decimal_before_as_tuple(
    value: Decimal,
) -> None:
    json_owner = importlib.import_module("quant_research_platform.contracts.json_codec")
    calls: list[str] = []
    previous_profile = sys.getprofile()

    def record_decimal_calls(frame: object, event: str, argument: object) -> None:
        del frame
        if event == "c_call" and getattr(argument, "__name__", None) == "as_tuple":
            calls.append("as_tuple")

    writer = json_owner.CanonicalIdentityWriter(max_bytes=500_000)
    sys.setprofile(record_decimal_calls)
    try:
        with pytest.raises(ValueError, match="canonical_identity_exceeds_max_bytes"):
            writer.value(value)
    finally:
        sys.setprofile(previous_profile)
    assert calls == []


@pytest.mark.parametrize("digit_count", (4_095, 4_096))
def test_bounded_writer_accepts_decimal_coefficient_ceiling_and_exact_budget(
    digit_count: int,
) -> None:
    json_owner = importlib.import_module("quant_research_platform.contracts.json_codec")
    value = Decimal("7" * digit_count)
    expected = canonical_identity_bytes(value)

    writer = json_owner.CanonicalIdentityWriter(max_bytes=len(expected) * 2)
    writer.value(value)
    assert writer.finish() == expected

    writer = json_owner.CanonicalIdentityWriter(max_bytes=len(expected) * 2 - 1)
    writer.value(value)
    with pytest.raises(ValueError, match="canonical_identity_exceeds_max_bytes"):
        writer.finish()


@pytest.mark.parametrize(
    "value",
    (
        Decimal("0"),
        Decimal("-0"),
        Decimal("0.00"),
        Decimal("-0.00"),
        Decimal("1.2300"),
        Decimal("-1.2300"),
        Decimal(f"1E+{MAX_EMAX}"),
        Decimal(f"1E{MIN_EMIN}"),
    ),
)
def test_bounded_writer_preserves_decimal_golden_within_runtime_domain(
    value: Decimal,
) -> None:
    json_owner = importlib.import_module("quant_research_platform.contracts.json_codec")
    expected = canonical_identity_bytes(value)
    writer = json_owner.CanonicalIdentityWriter(max_bytes=len(expected) * 2)

    writer.value(value)

    assert writer.finish() == expected


def test_bounded_writer_rejects_out_of_domain_decimal_before_as_tuple() -> None:
    json_owner = importlib.import_module("quant_research_platform.contracts.json_codec")
    value = Decimal((0, (1,), MIN_EMIN - 1))
    calls: list[str] = []
    previous_profile = sys.getprofile()

    def record_decimal_calls(frame: object, event: str, argument: object) -> None:
        del frame
        if event == "c_call" and getattr(argument, "__name__", None) == "as_tuple":
            calls.append("as_tuple")

    writer = json_owner.CanonicalIdentityWriter(max_bytes=100_000)
    sys.setprofile(record_decimal_calls)
    try:
        with pytest.raises(ValueError, match="canonical_identity_exceeds_max_bytes"):
            writer.value(value)
    finally:
        sys.setprofile(previous_profile)
    assert calls == []


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


@pytest.mark.parametrize(
    "value",
    (
        [Decimal("7" * 4_097)],
        (Decimal("7" * 4_097),),
        {"decimal": Decimal("7" * 4_097)},
    ),
    ids=("list", "tuple", "mapping-value"),
)
def test_bounded_writer_applies_decimal_ceiling_inside_contract_containers(
    value: object,
) -> None:
    json_owner = importlib.import_module("quant_research_platform.contracts.json_codec")
    writer = json_owner.CanonicalIdentityWriter(max_bytes=500_000)

    with pytest.raises(ValueError, match="canonical_identity_exceeds_max_bytes"):
        writer.value(value)


@pytest.mark.parametrize(
    "value",
    (Decimal("Infinity"), Decimal("-Infinity"), Decimal("NaN"), Decimal("sNaN")),
)
def test_bounded_writer_decimal_nonfinite_error_precedes_admission(
    value: Decimal,
) -> None:
    json_owner = importlib.import_module("quant_research_platform.contracts.json_codec")
    writer = json_owner.CanonicalIdentityWriter(max_bytes=1)

    with pytest.raises(ValueError, match="canonical_json_nonfinite_number"):
        writer.value(value)


@pytest.mark.parametrize("value", ("指标", {"指标": "值"}, Decimal("12.50")))
def test_bounded_writer_preserves_small_text_mapping_and_decimal_boundaries(
    value: object,
) -> None:
    json_owner = importlib.import_module("quant_research_platform.contracts.json_codec")
    expected = canonical_identity_bytes(value)
    accepted_bytes = len(expected) * 2 + (256 if type(value) is dict else 0)
    writer = json_owner.CanonicalIdentityWriter(max_bytes=accepted_bytes)
    writer.value(value)
    assert writer.finish() == expected

    writer = json_owner.CanonicalIdentityWriter(
        max_bytes=(len(expected) - 1 if type(value) is dict else accepted_bytes - 1)
    )
    if type(value) is dict:
        with pytest.raises(ValueError, match="canonical_identity_exceeds_max_bytes"):
            writer.value(value)
    else:
        writer.value(value)
        with pytest.raises(ValueError, match="canonical_identity_exceeds_max_bytes"):
            writer.finish()


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
    assert canonical_json_hash(Decimal("-0.00")) == canonical_json_hash(Decimal("0.00"))
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


@pytest.mark.parametrize(
    "value",
    (pd.NaT, pd.Timestamp("NaT")),
    ids=("pandas-nat", "timestamp-nat"),
)
def test_typed_canonical_identity_rejects_nat_sentinel(value: object) -> None:
    assert value is pd.NaT
    with pytest.raises(ValueError, match="canonical_json_invalid_timestamp"):
        canonical_identity_bytes(value)


def test_typed_canonical_identity_does_not_normalize_unicode() -> None:
    assert canonical_json_hash("\u00e9") != canonical_json_hash("e\u0301")


def test_strict_json_rejects_huge_integers_with_stable_errors() -> None:
    raw = b'{"schema_id":"x","value":' + (b"1" * 4097) + b"}"
    huge = 1 << 14_000
    boundary = (10**4096) - 1
    decimal_overflow = 10**4096

    assert (
        parse_qrp_json_object(
            b'{"schema_id":"x","value":' + str(boundary).encode("ascii") + b"}"
        )["value"]
        == boundary
    )
    render_qrp_json_object({"schema_id": "x", "value": boundary})
    canonical_identity_bytes(boundary)

    with pytest.raises(ValueError, match="qrp_json_integer_too_large"):
        parse_qrp_json_object(raw)
    with pytest.raises(ValueError, match="qrp_json_integer_too_large"):
        render_qrp_json_object({"schema_id": "x", "value": decimal_overflow})
    with pytest.raises(ValueError, match="canonical_json_integer_too_large"):
        canonical_identity_bytes(decimal_overflow)
    with pytest.raises(ValueError, match="qrp_json_integer_too_large"):
        render_qrp_json_object({"schema_id": "x", "value": huge})
    with pytest.raises(ValueError, match="canonical_json_integer_too_large"):
        canonical_identity_bytes(huge)


def test_strict_json_rejects_cycles_and_excessive_recursion() -> None:
    cyclic: dict[str, object] = {"schema_id": "x"}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="qrp_json_cycle"):
        render_qrp_json_object(cyclic)
    with pytest.raises(ValueError, match="canonical_json_cycle"):
        canonical_identity_bytes(cyclic)

    nested: object = None
    for _ in range(300):
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


@pytest.mark.parametrize(
    "exception_type", [KeyboardInterrupt, SystemExit, GeneratorExit]
)
@pytest.mark.parametrize("encode", [canonical_identity_bytes, canonical_json_hash])
def test_mapping_comparison_preserves_primary_exception(exception_type, encode) -> None:
    codec = importlib.import_module("quant_research_platform.contracts.json_codec")
    primary = exception_type("comparison interrupted")
    injected = False

    def trace(frame, event, arg):
        nonlocal injected
        if (
            frame.f_code is codec._compare_identity_ranges.__code__
            and event == "return"
        ):
            injected = True
            raise primary
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(exception_type) as caught:
            encode({"aa": [{"量化": 1, "é": 2}], "a": 1})
        assert caught.value is primary
        assert injected
    finally:
        sys.settrace(previous)
    assert sys.gettrace() is previous
