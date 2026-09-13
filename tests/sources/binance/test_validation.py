"""Exact scalar observation clocks shared by capture and archive decoding."""

import pytest

from quant_research_platform.sources.binance.validation import strict_iso_timestamp


@pytest.mark.parametrize(
    ("text", "nanoseconds"),
    [
        ("2026-09-01T00:00:00Z", 1788220800000000000),
        ("2026-09-01T00:00:00.000001Z", 1788220800000001000),
        ("2026-09-01T00:00:00.123456789Z", 1788220800123456789),
        ("2026-09-01T08:00:00.123456789+08:00", 1788220800123456789),
        ("1970-01-01T00:00:00.000000001Z", 1),
        ("1969-12-31T23:59:59.999999999Z", -1),
        ("2262-04-11T23:47:16.854775807Z", 2**63 - 1),
        ("1677-09-21T00:12:43.145224193Z", -(2**63) + 1),
    ],
)
def test_scalar_timestamp_preserves_nanoseconds_and_normalizes_real_offset(
    text, nanoseconds
):
    value = strict_iso_timestamp(text, error="clock_invalid")
    assert value.value == nanoseconds
    assert str(value.tz) == "UTC"


@pytest.mark.parametrize(
    "value",
    [
        None,
        0,
        "",
        "NaT",
        "NaTZ",
        "2026-09-01",
        "2026-09-01T00:00:00",
        "2025-02-29T00:00:00Z",
        "2026-09-01T23:59:60Z",
        "2026-09-01T24:00:00Z",
        "2026-09-01T00:00:00+99:00",
    ],
)
def test_scalar_timestamp_rejects_missing_offset_and_invalid_calendar(value):
    with pytest.raises(ValueError, match="^clock_invalid$"):
        strict_iso_timestamp(value, error="clock_invalid")


@pytest.mark.parametrize("text", ["1000-01-01T00:00:00Z", "3000-01-01T00:00:00Z"])
def test_valid_wide_timestamp_is_not_forced_into_nanosecond_storage(text):
    result = strict_iso_timestamp(text, error="clock_invalid")
    assert str(result.tz) == "UTC"
    assert result.year == int(text[:4])
    with pytest.raises(OverflowError):
        _ = result.value
