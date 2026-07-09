from __future__ import annotations

import math

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
    with pytest.raises(ValueError, match="invalid_contract_timestamp"):
        validation.strict_timestamp_series(
            pd.Series(["2024-01-02", "not-a-date"]),
            error="invalid_contract_timestamp",
        )
    with pytest.raises(ValueError, match="invalid_contract_timestamp"):
        validation.strict_timestamp_series(
            pd.Series([pd.NaT]),
            error="invalid_contract_timestamp",
        )
