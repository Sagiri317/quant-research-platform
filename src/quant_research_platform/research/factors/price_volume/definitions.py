"""Canonical price-volume definitions, inputs and session windows."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId

PRICE_VOLUME_FACTOR_FAMILY = "price_volume"

# Each entry owns the base inputs and the required session windows.
_PRICE_VOLUME_INPUT_WINDOWS: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {
    "reversal_1d": (("adj_close",), (1,)),
    "reversal_5d": (("adj_close",), (5,)),
    "reversal_20d": (("adj_close",), (20,)),
    "momentum_60_20": (("adj_close",), (60, 20)),
    "momentum_120_20": (("adj_close",), (120, 20)),
    "momentum_252_20": (("adj_close",), (252, 20)),
    "volatility_20d": (("period_return",), (20,)),
    "volatility_60d": (("period_return",), (60,)),
    "downside_volatility_20d": (("period_return",), (20,)),
    "downside_volatility_60d": (("period_return",), (60,)),
    "max_return_20d": (("period_return",), (20,)),
    "skewness_60d": (("period_return",), (60,)),
    "drawdown_60d": (("adj_close",), (60,)),
    "amplitude_20d": (("high", "low", "close"), (20,)),
    "log_avg_amount_20d": (("amount",), (20,)),
    "turnover_20d": (("turnover",), (20,)),
    "turnover_60d": (("turnover",), (60,)),
    "turnover_volatility_20d": (("turnover",), (20,)),
    "amihud_20d": (("period_return", "amount"), (20,)),
    "amihud_60d": (("period_return", "amount"), (60,)),
    "amount_shock_20d": (("amount",), (20,)),
    "return_amount_corr_20d": (("period_return", "amount"), (20,)),
    "trend_efficiency_60d": (("period_return",), (60,)),
    "signed_amount_flow_20d": (("period_return", "amount"), (20,)),
    "log_float_market_cap": (("float_market_cap",), ()),
    "momentum_20d": (("adj_close",), (20,)),
    "momentum_60d": (("adj_close",), (60,)),
    "upside_volatility_20d": (("period_return",), (20,)),
    "min_return_20d": (("period_return",), (20,)),
    "kurtosis_60d": (("period_return",), (60,)),
    "avg_amount_20d": (("amount",), (20,)),
    "amount_change_20d": (("amount",), (20,)),
    "volume_change_20d": (("volume",), (20,)),
    "turnover_change_20d": (("turnover",), (20,)),
    "amount_volatility_20d": (("amount",), (20,)),
    "volume_shock_20d": (("volume",), (20,)),
    "return_volume_corr_20d": (("period_return", "volume"), (20,)),
    "return_turnover_corr_20d": (
        ("period_return", "turnover"),
        (20,),
    ),
    "return_amount_imbalance_20d": (("period_return", "amount"), (20,)),
    "zero_amount_ratio_20d": (("amount",), (20,)),
    "log_total_market_cap": (("total_market_cap",), ()),
    "share_price": (("close",), ()),
    "avg_amount_126d": (("amount",), (126,)),
    "amount_cv_126d": (("amount",), (126,)),
    "turnover_cv_126d": (("turnover",), (126,)),
    "amihud_126d": (("period_return", "amount"), (126,)),
    "skewness_20d": (("period_return",), (20,)),
    "top5_return_mean_20d": (("period_return",), (20,)),
    "turnover_250d": (("turnover",), (250,)),
    "abnormal_turnover_20_250": (("turnover",), (250, 20)),
}

PRICE_VOLUME_LIBRARY = tuple(_PRICE_VOLUME_INPUT_WINDOWS)
PRICE_VOLUME_PATH_FACTORS = (
    "positive_return_fraction_20d",
    "largest_daily_contribution_20d",
    "path_roughness_20d",
)
_PRICE_VOLUME_INPUT_WINDOWS.update(
    {column: (("period_return",), (20,)) for column in PRICE_VOLUME_PATH_FACTORS}
)


# Only sample statistics admit partial observations within the nominal window.
PRICE_VOLUME_PARTIAL_STATISTICS = frozenset(
    factor
    for factor in _PRICE_VOLUME_INPUT_WINDOWS
    if (
        factor in {"amount_shock_20d", "volume_shock_20d"}
        or factor.startswith(
            (
                "avg_amount_",
                "volatility_",
                "downside_volatility_",
                "amihud_",
                "skewness_",
            )
        )
        or factor.endswith("_corr_20d")
        or factor
        in {
            "log_avg_amount_20d",
            "turnover_20d",
            "turnover_60d",
            "turnover_250d",
            "amount_volatility_20d",
            "turnover_volatility_20d",
            "upside_volatility_20d",
            "kurtosis_60d",
            "amount_cv_126d",
            "turnover_cv_126d",
            "abnormal_turnover_20_250",
            "amplitude_20d",
        }
    )
)


def price_volume_factor_definitions(
    *,
    market_id: MarketId,
    required_input_data_id: str,
    factor_columns: Iterable[str] | None = None,
) -> tuple[FactorDefinition, ...]:
    """Bind canonical definitions without changing their identity across selections."""
    selected = (
        PRICE_VOLUME_LIBRARY
        if factor_columns is None
        else tuple(dict.fromkeys(factor_columns))
    )
    definitions = []
    for column in selected:
        try:
            inputs, windows = _PRICE_VOLUME_INPUT_WINDOWS[column]
        except KeyError as exc:
            raise ValueError(f"unknown_price_volume_factor: {column}") from exc
        lookback = max(windows, default=None)
        definitions.append(
            FactorDefinition(
                factor_id=f"price_volume_{column}",
                market_id=market_id,
                factor_set=FactorSet.CORE
                if column
                in {
                    "momentum_252_20",
                    "reversal_5d",
                    "volatility_20d",
                    "amihud_20d",
                    "turnover_20d",
                    "log_float_market_cap",
                }
                else FactorSet.STANDARD,
                input_columns=(*inputs, "filling_flag")
                if column in PRICE_VOLUME_PARTIAL_STATISTICS
                else inputs,
                output_column=column,
                display_name=column,
                description=(
                    (
                        "Share of nonpositive amount (amount <= 0) over 20 complete "
                        "sessions; missing amounts propagate."
                    )
                    if column == "zero_amount_ratio_20d"
                    else (
                        "Statistic over the full nominal own-market session window; "
                        "requires ceil(min_fresh_fraction * window) jointly valid "
                        "fresh "
                        "observations (default 0.75), plus mathematical sample minima. "
                        "Numeric denominators use actual valid samples; confirmed "
                        "suspension valuations do not increase the fresh count."
                    )
                    if column in PRICE_VOLUME_PARTIAL_STATISTICS
                    else (
                        "Price-volume factor using complete session windows or exact "
                        "endpoints; missing required observations and invalid "
                        "denominators preserve NULL."
                    )
                ),
                family=PRICE_VOLUME_FACTOR_FAMILY,
                required_windows=windows,
                lookback_window=lookback,
                grain=DataGrain.ASSET_TIMESTAMP,
                timing_semantics="same_session_close_available_at_next_session",
                null_policy="preserve_nulls_after_min_periods",
                source_layer="processed",
                required_input_data_id=required_input_data_id,
            )
        )
    return tuple(definitions)
