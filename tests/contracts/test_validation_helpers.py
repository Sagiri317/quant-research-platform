from __future__ import annotations

import math
import warnings
from datetime import timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import quant_research_platform.contracts.validation as validation


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
def test_real_numeric_string_dtype_uses_vectorized_parsing(monkeypatch, storage):
    def unexpected_scalar_check(_value):
        raise AssertionError("StringDtype cannot contain non-string objects")

    monkeypatch.setattr(validation, "_is_hashable", unexpected_scalar_check)
    values = pd.Series(["1.5", "bad", None, "inf"], dtype=pd.StringDtype(storage))
    result = validation.strict_real_numeric_series(
        values, error="numeric_invalid", errors="coerce"
    )
    pd.testing.assert_series_equal(result, pd.Series([1.5, np.nan, np.nan, np.inf]))
    with pytest.raises(ValueError, match="numeric_invalid"):
        validation.strict_real_numeric_series(values, error="numeric_invalid")


@pytest.mark.parametrize(
    "value",
    [True, None, "not-a-number", float("nan"), math.inf, -math.inf, 10**4000],
)
def test_strict_finite_float_maps_invalid_scalars_to_owner_error(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_contract_scalar"):
        validation.strict_finite_float(value, error="invalid_contract_scalar")


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


@pytest.mark.parametrize("dtype", ["int64", "Int64", "uint64", "object"])
def test_strict_integer_series_preserves_large_exact_values(dtype: str) -> None:
    values = pd.Series(
        [2**53 + 1, 2**63 - 1], dtype=dtype, index=["a", "b"], name="market_type"
    )
    result = validation.strict_integer_series(values, error="invalid_integer")
    pd.testing.assert_series_equal(result, values.astype("int64"))


def test_strict_integer_series_preserves_nullable_and_mixed_exact_values() -> None:
    values = pd.Series([2**53 + 1, None, str(2**63 - 1), "2.0", -(2**63)], dtype=object)
    result = validation.strict_integer_series(
        values, error="invalid_integer", allow_null=True
    )
    pd.testing.assert_series_equal(
        result, pd.Series([2**53 + 1, pd.NA, 2**63 - 1, 2, -(2**63)], dtype="Int64")
    )


@pytest.mark.parametrize("value", [2**63, -(2**63) - 1, str(2**63), float(2**63)])
def test_strict_integer_series_rejects_int64_overflow(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_integer"):
        validation.strict_integer_series(pd.Series([value]), error="invalid_integer")


def test_strict_integer_series_rejects_out_of_range_before_integer_allocation() -> None:
    class OutOfRange(Decimal):
        def __int__(self):
            pytest.fail("out-of-range integer conversion must not allocate")

    with pytest.raises(ValueError, match="invalid_integer"):
        validation.strict_integer_series(
            pd.Series([OutOfRange("1e1000000000")], dtype=object),
            error="invalid_integer",
        )


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        math.inf,
        -math.inf,
        True,
        [],
        pytest.param(10**400, id="huge-positive-integer"),
        pytest.param(-(10**400), id="huge-negative-integer"),
        pytest.param(Decimal("1e400"), id="huge-decimal"),
    ],
)
def test_strict_finite_numeric_series_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="invalid_contract_numeric"):
        validation.strict_finite_numeric_series(
            pd.Series([value], dtype=object),
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


@pytest.mark.parametrize("dtype", ["bool", "boolean"])
@pytest.mark.parametrize("allow_null", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_typed_bool_validation_preserves_values_index_and_detaches(
    dtype, allow_null, empty
) -> None:
    values = [] if empty else [True, False]
    index = pd.Index([] if empty else [7, 3], dtype="int64", name="row")
    source = pd.Series(values, index=index, dtype=dtype, name="named")
    result = validation.strict_bool_series(
        source, error="invalid_contract_bool", allow_null=allow_null
    )
    expected = pd.Series(values, index=index, dtype="boolean" if allow_null else "bool")
    pd.testing.assert_series_equal(result, expected)
    if not empty:
        result.iloc[0] = False
        assert source.iloc[0]


def test_nullable_bool_validation_keeps_null_policy() -> None:
    source = pd.Series([True, pd.NA], dtype="boolean", name="named")
    with pytest.raises(ValueError, match="invalid_contract_bool"):
        validation.strict_bool_series(source, error="invalid_contract_bool")
    result = validation.strict_bool_series(
        source, error="invalid_contract_bool", allow_null=True
    )
    pd.testing.assert_series_equal(result, source.rename(None))


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


@pytest.mark.parametrize("value", [True, 1, 1.5, 20240101, object()])
def test_strict_timestamp_series_rejects_non_datetime_numeric_or_object_values(
    value: object,
) -> None:
    with pytest.raises(ValueError, match="invalid_contract_timestamp"):
        validation.strict_timestamp_series(
            pd.Series([value]),
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


def test_strict_timestamp_series_normalizes_dst_offsets() -> None:
    zone = ZoneInfo("America/New_York")
    values = pd.Series(
        [
            pd.Timestamp("2024-01-15T12:00:00", tz=zone),
            pd.Timestamp("2024-07-15T12:00:00", tz=zone),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = validation.strict_timestamp_series(
            values, error="invalid_contract_timestamp"
        )
    assert result.tolist() == [
        pd.Timestamp("2024-01-15T17:00:00"),
        pd.Timestamp("2024-07-15T16:00:00"),
    ]


def test_strict_timestamp_series_normalizes_explicit_mixed_offsets_with_null():
    values = pd.Series(
        ["2024-01-01T00:00:00+00:00", None, "2024-01-01T00:00:00+01:00"],
        index=["first", "missing", "last"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = validation.strict_timestamp_series(
            values, error="invalid_contract_timestamp", allow_null=True
        )
    assert result.index.equals(values.index)
    assert result.iloc[0] == pd.Timestamp("2024-01-01T00:00:00")
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pd.Timestamp("2023-12-31T23:00:00")


@pytest.mark.parametrize(
    "values",
    [
        [
            pd.Timestamp("2024-01-15T12:00:00", tz=timezone(timedelta(hours=1))),
            pd.Timestamp("2024-01-15T12:00:00", tz=ZoneInfo("Europe/Paris")),
        ],
        [
            pd.Timestamp("2024-01-15T12:00:00", tz=ZoneInfo("Europe/Paris")),
            pd.Timestamp("2024-01-15T12:00:00", tz=ZoneInfo("Europe/Berlin")),
        ],
        [
            pd.Timestamp("2024-01-15T12:00:00", tz=timezone(timedelta(hours=-5))),
            pd.Timestamp("2024-01-15T12:00:00", tz=ZoneInfo("America/New_York")),
        ],
    ],
    ids=["fixed-paris", "paris-berlin", "fixed-new-york"],
)
def test_strict_timestamp_series_accepts_equivalent_timezone_representations(
    values: list[pd.Timestamp],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = validation.strict_timestamp_series(
            pd.Series(values, index=["first", "second"]),
            error="invalid_contract_timestamp",
        )

    assert result.index.tolist() == ["first", "second"]
    assert result.dtype == "datetime64[ns]"
    assert result.dt.tz is None
    assert result.iloc[0] == result.iloc[1]


def test_strict_timestamp_series_falls_back_when_vectorized_parse_loses_value(
    monkeypatch,
) -> None:
    values = pd.Series(
        ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
        index=["first", "second"],
    )
    original_to_datetime = validation.pd.to_datetime
    calls: list[object] = []

    def lossy_vectorized_parse(values, *args, **kwargs):
        calls.append(values)
        if len(calls) == 1:
            return pd.Series(
                [pd.NaT, pd.Timestamp("2024-01-02T00:00:00")],
                index=values.index,
            )
        return original_to_datetime(values, *args, **kwargs)

    monkeypatch.setattr(validation.pd, "to_datetime", lossy_vectorized_parse)

    result = validation.strict_timestamp_series(
        values,
        error="invalid_contract_timestamp",
    )

    assert len(calls) == 3
    assert isinstance(calls[0], pd.Series)
    assert result.index.tolist() == ["first", "second"]
    assert result.tolist() == [
        pd.Timestamp("2024-01-01T00:00:00"),
        pd.Timestamp("2024-01-02T00:00:00"),
    ]


@pytest.mark.parametrize(
    "values",
    [
        ["2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00"],
    ],
    ids=["aware-naive"],
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


@pytest.mark.parametrize(
    "invalid",
    [
        True,
        np.bool_(False),
        1 + 2j,
        "bad",
        10**4000,
        [1, 2],
        pd.Timestamp("2024-01-01"),
        pd.Timedelta(days=1),
    ],
)
def test_real_numeric_unavailable_cells_preserve_strict_default(invalid):
    from quant_research_platform.contracts.validation import strict_real_numeric_series

    values = pd.Series([2.5, invalid, None, "3.5", np.inf], dtype=object)
    with pytest.raises(ValueError, match="numeric_invalid"):
        strict_real_numeric_series(values, error="numeric_invalid")
    result = strict_real_numeric_series(
        values, error="numeric_invalid", errors="coerce"
    )
    assert result.iloc[0] == 2.5
    assert result.iloc[1:3].isna().all()
    assert result.iloc[3] == 3.5
    assert result.iloc[4] == np.inf


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([True, False]),
        pd.Series([1 + 2j, 3 + 0j]),
        pd.Series(pd.to_datetime(["2024-01-01", None])),
        pd.Series(pd.to_timedelta([1, 2], unit="D")),
    ],
)
def test_real_numeric_coercion_does_not_convert_nonreal_dtypes(values):
    from quant_research_platform.contracts.validation import strict_real_numeric_series

    with pytest.raises(ValueError, match="numeric_invalid"):
        strict_real_numeric_series(values, error="numeric_invalid")
    assert (
        strict_real_numeric_series(values, error="numeric_invalid", errors="coerce")
        .isna()
        .all()
    )
