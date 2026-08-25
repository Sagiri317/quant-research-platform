"""Numeric safety helpers for price-volume calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import strict_finite_numeric_series


def safe_log(values: pd.Series) -> pd.Series:
    numeric = strict_finite_numeric_series(
        values,
        error="invalid_price_volume_numeric_values",
        allow_null=True,
    ).astype("float64")
    numeric = numeric.where(np.isfinite(numeric) & numeric.gt(0.0))
    return finite_or_nan(np.log(numeric))


def finite_or_nan(values: pd.Series) -> pd.Series:
    return values.replace([np.inf, -np.inf], np.nan)


def valid_denominator(values: pd.Series) -> pd.Series:
    numeric = strict_finite_numeric_series(
        values,
        error="invalid_price_volume_numeric_values",
        allow_null=True,
    ).astype("float64")
    return numeric.where(np.isfinite(numeric) & numeric.ne(0.0))
