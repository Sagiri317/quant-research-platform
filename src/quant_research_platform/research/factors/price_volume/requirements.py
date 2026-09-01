"""Price-volume factor input requirements."""

from __future__ import annotations

from collections.abc import Iterable

from .columns import price_volume_window_column
from .windows import parse_price_volume_windowed_factor_name

BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "reversal_1d": ("adj_close",),
    "reversal_5d": ("adj_close",),
    "reversal_20d": ("adj_close",),
    "momentum_60_20": ("adj_close",),
    "momentum_120_20": ("adj_close",),
    "momentum_252_20": ("adj_close",),
    "volatility_20d": ("period_return",),
    "volatility_60d": ("period_return",),
    "downside_volatility_20d": ("period_return",),
    "downside_volatility_60d": ("period_return",),
    "max_return_20d": ("period_return",),
    "skewness_60d": ("period_return",),
    "drawdown_60d": ("adj_close",),
    "amplitude_20d": ("high", "low", "close"),
    "log_avg_amount_20d": ("amount",),
    "turnover_20d": ("volume", "float_market_cap", "close"),
    "turnover_60d": ("volume", "float_market_cap", "close"),
    "turnover_volatility_20d": ("volume", "float_market_cap", "close"),
    "amihud_20d": ("period_return", "amount"),
    "amihud_60d": ("period_return", "amount"),
    "amount_shock_20d": ("amount",),
    "return_amount_corr_20d": ("period_return", "amount"),
    "trend_efficiency_60d": ("period_return",),
    "signed_amount_flow_20d": ("period_return", "amount"),
    "log_float_market_cap": ("float_market_cap",),
    price_volume_window_column("rev", 1): ("period_return",),
    "mom": ("adj_close",),
    "rev": ("adj_close",),
    "vol": ("period_return",),
    "downside_vol": ("period_return",),
    "upside_vol": ("period_return",),
    "max_return": ("period_return",),
    "min_return": ("period_return",),
    "drawdown": ("adj_close",),
    "skew": ("period_return",),
    "kurt": ("period_return",),
    "avg_amount": ("amount",),
    "amount_chg": ("amount",),
    "amount_shock": ("amount",),
    "amount_vol": ("amount",),
    "volume_chg": ("volume",),
    "volume_shock": ("volume",),
    "turnover": ("turnover",),
    "turnover_chg": ("turnover",),
    "turnover_vol": ("turnover",),
    "ret_amount_corr": ("period_return", "amount"),
    "ret_volume_corr": ("period_return", "volume"),
    "ret_turnover_corr": ("period_return", "turnover"),
    "trend_efficiency": ("period_return",),
    "signed_amount_flow": ("period_return", "amount"),
    "ret_amount_imbalance": ("period_return", "amount"),
    "amihud": ("period_return", "amount"),
    "zero_volume_ratio": ("amount",),
    "amplitude": ("high", "low", "prev_close"),
    "intraday_range": ("high", "low", "adj_close"),
    "log_total_market_cap": ("total_market_cap",),
}
_WINDOWED_PRICE_VOLUME_FACTOR_PREFIXES: frozenset[str] = frozenset(
    {
        "mom",
        "rev",
        "vol",
        "downside_vol",
        "upside_vol",
        "max_return",
        "min_return",
        "drawdown",
        "skew",
        "kurt",
        "avg_amount",
        "amount_chg",
        "amount_shock",
        "amount_vol",
        "volume_chg",
        "volume_shock",
        "turnover",
        "turnover_chg",
        "turnover_vol",
        "ret_amount_corr",
        "ret_volume_corr",
        "ret_turnover_corr",
        "trend_efficiency",
        "signed_amount_flow",
        "ret_amount_imbalance",
        "amihud",
        "zero_volume_ratio",
        "amplitude",
        "intraday_range",
    }
)


def price_volume_factor_input_columns(factor_id: str) -> tuple[str, ...]:
    """Return exact base input columns required by one price-volume factor."""

    required = _required_fields_for_factor(factor_id)
    if not required:
        raise ValueError(f"unknown_price_volume_factor: {factor_id}")
    return required


def unsupported_price_volume_factors(
    factor_columns: Iterable[str],
) -> tuple[str, ...]:
    """Return requested factor names that the price-volume engine cannot compute."""

    unsupported: list[str] = []
    for factor in tuple(dict.fromkeys(factor_columns)):
        if not _required_fields_for_factor(factor):
            unsupported.append(factor)
    return tuple(unsupported)


def _required_fields_for_factor(factor: str) -> tuple[str, ...]:
    if factor in BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS:
        return BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS[factor]
    parsed = parse_price_volume_windowed_factor_name(factor)
    if parsed is None:
        return ()
    prefix, window, frequency = parsed
    if prefix not in _WINDOWED_PRICE_VOLUME_FACTOR_PREFIXES:
        return ()
    if frequency.value != "1d":
        return ()
    if prefix == "mom" and window == 1:
        return ()
    return BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS.get(prefix, ())
