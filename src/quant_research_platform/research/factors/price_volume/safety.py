"""Numeric safety helpers for price-volume calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_log(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.where(np.isfinite(numeric) & numeric.gt(0.0))
    return finite_or_nan(np.log(numeric))


def finite_or_nan(values: pd.Series) -> pd.Series:
    return values.replace([np.inf, -np.inf], np.nan)


def valid_denominator(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(np.isfinite(numeric) & numeric.ne(0.0))
