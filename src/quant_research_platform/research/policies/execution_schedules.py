"""Continuous-time Almgren--Chriss schedule and a causal signal adjustment."""

import numpy as np
import pandas as pd

from quant_research_platform.contracts.json_codec import _strict_config
from quant_research_platform.contracts.validation import strict_finite_float


def validate_execution_as_of(value, *, start, error):
    """Require an observed UTC-naive calibration time strictly before execution."""
    as_of = pd.Timestamp(value)
    if pd.isna(as_of) or as_of.tz is not None or as_of >= start:
        raise ValueError(error)


def execution_schedule_parameters(value, *, signal, start, feature_columns):
    names = {"risk_aversion", "price_volatility", "temporary_impact"}
    params = _strict_config(
        value,
        {"as_of", *names} | ({"signal_column", "signal_gain"} if signal else set()),
    )
    validate_execution_as_of(
        params["as_of"], start=start, error="execution_parameters_not_historical"
    )
    for name in names | ({"signal_gain"} if signal else set()):
        params[name] = strict_finite_float(
            params[name], error="execution_schedule_parameter_invalid"
        )
        if params[name] < 0:
            raise ValueError("execution_schedule_parameter_invalid")
    if (
        params["temporary_impact"] == 0
        or signal
        and params["signal_column"] not in feature_columns
    ):
        raise ValueError("execution_schedule_parameter_invalid")
    return params


def execution_fraction(fraction, duration_seconds, params, *, signal=0.0, side=1):
    """Price volatility is currency/share/sqrt(second); eta is currency*second/share².

    Risk aversion is inverse currency. Signal gain converts the declared signal's
    unit into a fraction. The adjustment is a heuristic, not the AC optimizer.
    """
    t = np.clip(fraction, 0.0, 1.0)
    k = (
        np.sqrt(params["risk_aversion"] / params["temporary_impact"])
        * params["price_volatility"]
    )
    scale = k * duration_seconds
    if scale == 0:
        desired = t
    else:
        remaining = (
            np.exp(-scale * t)
            * (-np.expm1(-2 * scale * (1 - t)))
            / (-np.expm1(-2 * scale))
        )
        desired = 1 - remaining
    if "signal_gain" in params:
        if not np.isfinite(signal):
            raise ValueError("execution_schedule_signal_missing")
        desired += side * params["signal_gain"] * signal * t * (1 - t)
    return float(np.clip(desired, 0.0, 1.0))
