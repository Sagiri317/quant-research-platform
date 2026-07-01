"""Price-volume factor column names and feature selection surfaces."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.contracts.data import DataFrequency, horizon_unit_suffix


def price_volume_window_column(
    prefix: str,
    window: int,
    frequency: DataFrequency | str = DataFrequency.DAILY,
) -> str:
    return f"{prefix}_{window}{horizon_unit_suffix(frequency)}"


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
    "amount_chg_",
    "amount_shock_",
    "amount_vol_",
    "volume_chg_",
    "volume_shock_",
    "turnover_",
    "turnover_chg_",
    "turnover_vol_",
    "ret_amount_corr_",
    "ret_volume_corr_",
    "ret_turnover_corr_",
    "trend_efficiency_",
    "signed_amount_flow_",
    "ret_amount_imbalance_",
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
    for window in (20, 60):
        columns.extend(
            (
                price_volume_window_column("amount_chg", window),
                price_volume_window_column("amount_shock", window),
                price_volume_window_column("amount_vol", window),
                price_volume_window_column("volume_chg", window),
                price_volume_window_column("volume_shock", window),
                price_volume_window_column("ret_amount_corr", window),
                price_volume_window_column("ret_volume_corr", window),
                price_volume_window_column("ret_turnover_corr", window),
                price_volume_window_column("trend_efficiency", window),
                price_volume_window_column("signed_amount_flow", window),
                price_volume_window_column("ret_amount_imbalance", window),
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

__all__ = [
    "PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS",
    "PRICE_VOLUME_FEATURE_EXCLUDE_PREFIXES",
    "PRICE_VOLUME_FACTOR_PREFIXES",
    "price_volume_factor_columns_for_windows",
    "price_volume_window_column",
]
