"""Price-volume factor preset definitions."""

from __future__ import annotations

from .columns import price_volume_factor_columns_for_windows, price_volume_window_column

PRICE_VOLUME_STANDARD_PRESET = "standard"
PRICE_VOLUME_CORE_PRESET = "core"
PRICE_VOLUME_SMOKE_PRESET = "smoke"

_PRICE_VOLUME_STANDARD_WINDOWS: tuple[int, ...] = (1, 5, 10, 20, 60, 120, 252)

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


__all__ = [
    "CORE_PRICE_VOLUME_FACTORS",
    "PRICE_VOLUME_CORE_PRESET",
    "PRICE_VOLUME_FACTOR_PRESETS",
    "PRICE_VOLUME_SMOKE_PRESET",
    "PRICE_VOLUME_STANDARD_PRESET",
    "SMOKE_PRICE_VOLUME_FACTORS",
    "STANDARD_PRICE_VOLUME_FACTORS",
    "price_volume_factor_columns",
]
