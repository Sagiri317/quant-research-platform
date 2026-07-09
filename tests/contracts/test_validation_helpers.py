from __future__ import annotations

import math
import warnings
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from quant_research_platform.contracts import validation


@pytest.mark.parametrize("value", [None, pd.NA, float("nan")])
def test_strict_validation_helpers_reject_nulls_by_default(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_contract_value"):
        validation.strict_string_series(
            pd.Series([value]),
            error="invalid_contract_value",
        )


def test_strict_string_series_rejects_non_strings_and_empty_values() -> None:
    for value in (1, [], {}, set(), ""):
        with pytest.raises(ValueError, match="invalid_contract_string"):
            validation.strict_string_series(
                pd.Series([value]),
                error="invalid_contract_string",
            )

    assert validation.strict_string_series(
        pd.Series([" value "]),
        error="invalid_contract_string",
    ).tolist() == ["value"]


@pytest.mark.parametrize("value", [True, 1.5, float("nan"), math.inf, -math.inf])
def test_strict_integer_series_rejects_non_integer_values(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_contract_integer"):
        validation.strict_integer_series(
            pd.Series([value]),
            error="invalid_contract_integer",
        )


def test_strict_integer_series_accepts_integral_numeric_values() -> None:
    assert validation.strict_integer_series(
        pd.Series([1, 1.0, "2"]),
        error="invalid_contract_integer",
    ).tolist() == [1, 1, 2]


@pytest.mark.parametrize("value", [float("nan"), math.inf, -math.inf, True, []])
def test_strict_finite_numeric_series_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_contract_numeric"):
        validation.strict_finite_numeric_series(
            pd.Series([value]),
            error="invalid_contract_numeric",
        )


@pytest.mark.parametrize("value", [[], {}, set(), "missing", None])
def test_strict_membership_series_rejects_invalid_members(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_contract_membership"):
        validation.strict_membership_series(
            pd.Series([value]),
            allowed={"known"},
            error="invalid_contract_membership",
        )


def test_strict_membership_series_accepts_known_hashable_members() -> None:
    assert validation.strict_membership_series(
        pd.Series(["known"]),
        allowed={"known"},
        error="invalid_contract_membership",
    ).tolist() == ["known"]


@pytest.mark.parametrize("value", [[], {}, set(), None, pd.NA])
def test_strict_hashable_key_series_rejects_invalid_keys(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_contract_key"):
        validation.strict_hashable_key_series(
            pd.Series([value]),
            error="invalid_contract_key",
        )


@pytest.mark.parametrize("value", [1, "yes", None, pd.NA])
def test_strict_bool_series_rejects_ambiguous_values(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_contract_bool"):
        validation.strict_bool_series(pd.Series([value]), error="invalid_contract_bool")


def test_strict_bool_series_accepts_explicit_bool_forms() -> None:
    assert validation.strict_bool_series(
        pd.Series([True, False, "true", "False", "1", "0"]),
        error="invalid_contract_bool",
    ).tolist() == [True, False, True, False, True, False]


def test_strict_timestamp_series_rejects_invalid_or_null_values() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="invalid_contract_timestamp"):
            validation.strict_timestamp_series(
                pd.Series(["not-a-date", "2024-01-02"]),
                error="invalid_contract_timestamp",
            )
    with pytest.raises(ValueError, match="invalid_contract_timestamp"):
        validation.strict_timestamp_series(
            pd.Series([pd.NaT]),
            error="invalid_contract_timestamp",
        )


def test_strict_timestamp_series_returns_naive_utc_for_aware_values() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = validation.strict_timestamp_series(
            pd.Series(
                [
                    "2024-01-01T00:00:00+09:00",
                    "2024-01-01T09:00:00+09:00",
                ]
            ),
            error="invalid_contract_timestamp",
        )

    assert result.dtype == "datetime64[ns]"
    assert result.dt.tz is None
    assert result.tolist() == [
        pd.Timestamp("2023-12-31T15:00:00"),
        pd.Timestamp("2024-01-01T00:00:00"),
    ]


def test_strict_timestamp_series_rejects_dst_mixed_offsets() -> None:
    zone = ZoneInfo("America/New_York")
    values = pd.Series(
        [
            pd.Timestamp("2024-01-15T12:00:00", tz=zone),
            pd.Timestamp("2024-07-15T12:00:00", tz=zone),
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="invalid_contract_timestamp"):
            validation.strict_timestamp_series(
                values,
                error="invalid_contract_timestamp",
            )


@pytest.mark.parametrize(
    "values",
    [
        ["2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00"],
        ["2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00+01:00"],
    ],
    ids=["aware-naive", "mixed-offsets"],
)
def test_strict_timestamp_series_rejects_mixed_timezone_inputs_without_warnings(
    values: list[str],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="invalid_contract_timestamp"):
            validation.strict_timestamp_series(
                pd.Series(values),
                error="invalid_contract_timestamp",
            )


def test_strict_timestamp_series_uses_vectorized_parse_for_large_string_input(
    monkeypatch,
) -> None:
    values = pd.Series(["2024-01-01T00:00:00"] * 2048)
    original_to_datetime = validation.pd.to_datetime
    calls: list[object] = []

    def spy_to_datetime(values, *args, **kwargs):
        calls.append(values)
        return original_to_datetime(values, *args, **kwargs)

    monkeypatch.setattr(validation.pd, "to_datetime", spy_to_datetime)
    result = validation.strict_timestamp_series(
        values,
        error="invalid_contract_timestamp",
    )

    assert len(calls) == 1
    assert isinstance(calls[0], pd.Series)
    assert result.dtype == "datetime64[ns]"
