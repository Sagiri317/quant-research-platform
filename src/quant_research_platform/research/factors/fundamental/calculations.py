"""Pure common fundamental factor mathematics."""

from __future__ import annotations

from collections.abc import Iterable
from math import fsum

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import strict_finite_numeric_series

from .definitions import FUNDAMENTAL_FACTORS, FUNDAMENTAL_FORMULAS, fundamental_inputs


def calculate_fundamental_factors(
    panel: pd.DataFrame,
    *,
    factor_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Calculate the 19 registered factors from PIT/derived accounting facts."""

    requested = tuple(
        dict.fromkeys(FUNDAMENTAL_FACTORS if factor_columns is None else factor_columns)
    )
    unknown = sorted(set(requested).difference(FUNDAMENTAL_FACTORS))
    if unknown:
        raise ValueError("unknown_fundamental_factor: " + ", ".join(unknown))
    required = {"asset_id", "timestamp"}
    if not required.issubset(panel.columns):
        raise ValueError("fundamental_factor_keys_invalid")
    inputs = tuple(dict.fromkeys(c for f in requested for c in fundamental_inputs(f)))
    if not set(inputs).issubset(panel.columns):
        raise ValueError("fundamental_factor_input_columns_missing")
    result = panel.loc[:, ("asset_id", "timestamp")].copy()
    values = {
        column: strict_finite_numeric_series(
            panel[column], error=f"invalid_fundamental_value: {column}", allow_null=True
        ).astype("float64")
        for column in inputs
    }

    for column in requested:
        terms, denominator, adjustment = FUNDAMENTAL_FORMULAS[column]
        numerator = tuple(
            -values[term[1:]] if term.startswith("-") else values[term]
            for term in terms
        )
        if column == "capex_to_assets":
            numerator = (numerator[0].where(numerator[0].ge(0.0)),)
        divisor = values[denominator]
        if adjustment:
            numerator = (*numerator, adjustment * divisor)
        ratio = _ratio(numerator[0] if len(numerator) == 1 else numerator, divisor)
        result[column] = ratio.replace([np.inf, -np.inf], np.nan).astype("float64")
    return result


def _ratio(
    numerator: pd.Series | tuple[pd.Series, ...],
    denominator: pd.Series,
) -> pd.Series:
    if isinstance(numerator, pd.Series):
        return numerator.div(denominator.where(denominator.gt(0.0)))
    terms = np.column_stack(numerator)
    divisors = denominator.to_numpy(dtype=float)
    valid = (divisors > 0) & np.isfinite(terms).all(axis=1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        totals = terms.sum(axis=1)
        result = np.divide(
            totals, divisors, out=np.full(len(divisors), np.nan), where=valid
        )
    mixed = (terms > 0).any(axis=1) & (terms < 0).any(axis=1)
    if terms.shape[1] > 2:
        for index in np.flatnonzero(valid & mixed):
            try:
                result[index] = fsum(terms[index]) / divisors[index]
            except OverflowError:
                result[index] = np.nan
    return pd.Series(result, index=denominator.index)
