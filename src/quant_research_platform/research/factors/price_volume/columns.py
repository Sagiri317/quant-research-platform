"""Price-volume factor column names, presets, and requirements."""

from __future__ import annotations

import re
from collections.abc import Iterable

from quant_research_platform.contracts.data import DataFrequency, horizon_unit_suffix

PRICE_VOLUME_FACTOR_FAMILY = "price_volume"

PRICE_VOLUME_STANDARD_PRESET = "standard"
PRICE_VOLUME_CORE_PRESET = "core"
PRICE_VOLUME_SMOKE_PRESET = "smoke"

_PRICE_VOLUME_STANDARD_WINDOWS: tuple[int, ...] = (1, 5, 10, 20, 60, 120, 252)


def price_volume_window_column(
    prefix: str,
    window: int,
    frequency: DataFrequency | str = DataFrequency.DAILY,
) -> str:
    return f"{prefix}_{window}{horizon_unit_suffix(frequency)}"


CORE_PRICE_VOLUME_FACTORS: tuple[str, ...] = (
    price_volume_window_column("rev", 1),
    price_volume_window_column("rev", 5),
    price_volume_window_column("rev", 10),
    price_volume_window_column("rev", 20),
    price_volume_window_column("mom", 20),
    price_volume_window_column("mom", 60),
    price_volume_window_column("mom", 120),
    price_volume_window_column("vol", 20),
    price_volume_window_column("vol", 60),
    price_volume_window_column("downside_vol", 20),
    price_volume_window_column("downside_vol", 60),
    price_volume_window_column("avg_amount", 20),
    price_volume_window_column("avg_amount", 60),
    price_volume_window_column("turnover", 20),
    price_volume_window_column("turnover", 60),
    price_volume_window_column("amihud", 20),
    price_volume_window_column("amihud", 60),
    price_volume_window_column("amplitude", 20),
    price_volume_window_column("amplitude", 60),
    "log_float_market_cap",
    "log_total_market_cap",
)
SMOKE_PRICE_VOLUME_FACTORS: tuple[str, ...] = (
    price_volume_window_column("rev", 1),
    price_volume_window_column("mom", 20),
    price_volume_window_column("vol", 20),
    price_volume_window_column("avg_amount", 20),
    price_volume_window_column("turnover", 20),
    price_volume_window_column("amihud", 20),
    price_volume_window_column("amplitude", 20),
    "log_float_market_cap",
    "log_total_market_cap",
)

PRICE_VOLUME_FEATURE_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "forward_return",
    "target",
    "label",
)
PRICE_VOLUME_FACTOR_PREFIXES: tuple[str, ...] = (
    "mom_",
    "rev_",
    "vol_",
    "downside_vol_",
    "upside_vol_",
    "max_return_",
    "min_return_",
    "drawdown_",
    "skew_",
    "kurt_",
    "avg_amount_",
    "turnover_",
    "turnover_chg_",
    "turnover_vol_",
    "amihud_",
    "zero_volume_ratio_",
    "amplitude_",
    "intraday_range_",
    "log_float_market_cap",
    "log_total_market_cap",
)
PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS: frozenset[str] = frozenset(
    {
        "market_id",
        "asset_id",
        "timestamp",
    }
)
BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS: dict[str, tuple[str, ...]] = {
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
    "turnover": ("turnover",),
    "turnover_chg": ("turnover",),
    "turnover_vol": ("turnover",),
    "amihud": ("period_return", "amount"),
    "zero_volume_ratio": ("amount",),
    "amplitude": ("high", "low", "prev_close"),
    "intraday_range": ("high", "low", "adj_close"),
    "log_float_market_cap": ("float_market_cap",),
    "log_total_market_cap": ("total_market_cap",),
}
_WINDOWED_PRICE_VOLUME_FACTOR_PREFIXES: tuple[str, ...] = (
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
    "turnover",
    "turnover_chg",
    "turnover_vol",
    "amihud",
    "zero_volume_ratio",
    "amplitude",
    "intraday_range",
)


def price_volume_factor_columns_for_windows(
    windows: Iterable[int],
) -> tuple[str, ...]:
    columns: list[str] = [price_volume_window_column("rev", 1)]
    for window in windows:
        if window != 1:
            columns.extend(
                (
                    price_volume_window_column("rev", window),
                    price_volume_window_column("mom", window),
                )
            )
    for window in (5, 20, 60, 120):
        columns.append(price_volume_window_column("vol", window))
    for window in (20, 60):
        columns.extend(
            (
                price_volume_window_column("downside_vol", window),
                price_volume_window_column("upside_vol", window),
                price_volume_window_column("max_return", window),
                price_volume_window_column("min_return", window),
                price_volume_window_column("drawdown", window),
                price_volume_window_column("skew", window),
                price_volume_window_column("kurt", window),
            )
        )
    for window in (5, 20, 60, 120):
        columns.extend(
            (
                price_volume_window_column("avg_amount", window),
                price_volume_window_column("turnover", window),
            )
        )
    for window in (20, 60):
        columns.extend(
            (
                price_volume_window_column("turnover_chg", window),
                price_volume_window_column("turnover_vol", window),
            )
        )
    for window in (20, 60, 120):
        columns.append(price_volume_window_column("amihud", window))
    for window in (20, 60):
        columns.append(price_volume_window_column("zero_volume_ratio", window))
    for window in (5, 20, 60, 120):
        columns.append(price_volume_window_column("amplitude", window))
    for window in (20, 60):
        columns.append(price_volume_window_column("intraday_range", window))
    columns.extend(("log_float_market_cap", "log_total_market_cap"))
    return tuple(dict.fromkeys(columns))


STANDARD_PRICE_VOLUME_FACTORS: tuple[str, ...] = (
    price_volume_factor_columns_for_windows(_PRICE_VOLUME_STANDARD_WINDOWS)
)

PRICE_VOLUME_FACTOR_PRESETS: dict[str, tuple[str, ...]] = {
    PRICE_VOLUME_STANDARD_PRESET: STANDARD_PRICE_VOLUME_FACTORS,
    PRICE_VOLUME_CORE_PRESET: CORE_PRICE_VOLUME_FACTORS,
    PRICE_VOLUME_SMOKE_PRESET: SMOKE_PRICE_VOLUME_FACTORS,
}


def price_volume_factor_columns(*, preset: str = "standard") -> tuple[str, ...]:
    """Return the official price-volume factor columns for a named preset."""

    try:
        return PRICE_VOLUME_FACTOR_PRESETS[preset]
    except KeyError as exc:
        raise ValueError("invalid_price_volume_preset") from exc


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


def largest_price_volume_lookback_days(
    factor_columns: Iterable[str],
) -> int:
    """Return the largest rolling/trailing lookback encoded in factor names."""

    windows = []
    for column in factor_columns:
        if column == price_volume_window_column("rev", 1):
            windows.append(1)
        windows.extend(
            int(value)
            for value in re.findall(r"_(\d+)(?:d|m|tick)(?:_|$)", column)
        )
    return max(windows, default=0)


def _required_windows(factor_id: str) -> tuple[int, ...]:
    windows = tuple(
        int(match) for match in re.findall(r"_(\d+)(?:d|m|tick)\b", factor_id)
    )
    return windows


def _lookback_window(factor_id: str) -> int | None:
    windows = _required_windows(factor_id)
    if not windows:
        return None
    return max(windows)


def _required_fields_for_factor(factor: str) -> tuple[str, ...]:
    if factor in BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS:
        return BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS[factor]
    for prefix in _WINDOWED_PRICE_VOLUME_FACTOR_PREFIXES:
        if re.fullmatch(rf"{re.escape(prefix)}_\d+d", factor):
            return BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS[prefix]
    return ()


def _windows_from_requested_factors(requested: set[str]) -> tuple[int, ...]:
    values = {1} if price_volume_window_column("rev", 1) in requested else set()
    for factor in requested:
        for token in factor.split("_"):
            if token.endswith("d") and token[:-1].isdigit():
                values.add(int(token[:-1]))
    return tuple(sorted(values))


__all__ = [
    "BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS",
    "CORE_PRICE_VOLUME_FACTORS",
    "PRICE_VOLUME_CORE_PRESET",
    "PRICE_VOLUME_FACTOR_FAMILY",
    "PRICE_VOLUME_FACTOR_PRESETS",
    "PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS",
    "PRICE_VOLUME_FEATURE_EXCLUDE_PREFIXES",
    "PRICE_VOLUME_FACTOR_PREFIXES",
    "PRICE_VOLUME_SMOKE_PRESET",
    "PRICE_VOLUME_STANDARD_PRESET",
    "SMOKE_PRICE_VOLUME_FACTORS",
    "STANDARD_PRICE_VOLUME_FACTORS",
    "largest_price_volume_lookback_days",
    "price_volume_factor_columns",
    "price_volume_factor_columns_for_windows",
    "price_volume_factor_input_columns",
    "price_volume_window_column",
    "unsupported_price_volume_factors",
]
