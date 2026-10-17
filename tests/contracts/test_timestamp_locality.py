"""Opt-in timestamp locality preserves supported scalar and UTC semantics."""

import pandas as pd
import pytest

from quant_research_platform.contracts.validation import strict_timestamp_series


@pytest.mark.parametrize(
    "bad", [1, 1.5, True, complex(1, 2), pd.Timedelta(days=1), "bad", "9999-01-01", [1]]
)
def test_coerced_dates_keep_valid_neighbor_without_epoch_fabrication(bad):
    values = pd.Series([bad, "2025-01-02T23:00:00-05:00"], dtype=object)
    result = strict_timestamp_series(values, error="date_invalid", errors="coerce")
    assert result.dtype == "datetime64[ns]"
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pd.Timestamp("2025-01-03T04:00:00")
    with pytest.raises(ValueError, match="date_invalid"):
        strict_timestamp_series(values, error="date_invalid")


@pytest.mark.parametrize(
    "values",
    [
        pd.Series([1, 2]),
        pd.Series([True, False]),
        pd.Series(pd.to_timedelta([1, 2], unit="D")),
    ],
)
def test_numeric_and_timedelta_dtype_cannot_be_dates(values):
    result = strict_timestamp_series(values, error="date_invalid", errors="coerce")
    assert result.isna().all()
    assert result.dtype == "datetime64[ns]"


def test_datetime_fast_path_keeps_utc_nanoseconds_and_index():
    values = pd.Series(pd.to_datetime(["2025-01-01", None], utc=True), index=[3, 7])
    result = strict_timestamp_series(values, error="date_invalid", errors="coerce")
    assert result.index.equals(values.index)
    assert result.iloc[0] == pd.Timestamp("2025-01-01")
    assert pd.isna(result.iloc[1])
