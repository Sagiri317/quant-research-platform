from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ._helpers import numeric_series_diff_report


def test_numeric_diff_helper_treats_nan_pairs_and_value_nan() -> None:
    report = numeric_series_diff_report(
        pd.Series([1.0, np.nan, np.nan, 4.0]),
        pd.Series([1.0 + 5e-13, np.nan, 3.0, 4.1]),
        atol=1e-12,
        rtol=0.0,
    )

    assert report.mismatch_count == 2
    assert report.max_abs_diff == pytest.approx(0.1)
    assert report.mean_abs_diff == pytest.approx(0.1)
    assert report.quantiles["1.0"] == pytest.approx(0.1)


def test_integer_and_source_code_columns_compare_exactly_in_key_diff_helpers() -> None:
    left = pd.Series(["1", "2", "3"], dtype="string")
    right = pd.Series(["1", "2", "4"], dtype="string")

    assert (left == right).tolist() == [True, True, False]
