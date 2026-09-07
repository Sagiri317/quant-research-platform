"""Canonical price-volume definitions, inputs and session windows."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId

PRICE_VOLUME_FACTOR_FAMILY = "price_volume"

FORMAL_PRICE_VOLUME_FACTORS: tuple[str, ...] = (
    "reversal_1d",
    "reversal_5d",
    "reversal_20d",
    "momentum_60_20",
    "momentum_120_20",
    "momentum_252_20",
    "volatility_20d",
    "volatility_60d",
    "downside_volatility_20d",
    "downside_volatility_60d",
    "max_return_20d",
    "skewness_60d",
    "drawdown_60d",
    "amplitude_20d",
    "log_avg_amount_20d",
    "turnover_20d",
    "turnover_60d",
    "turnover_volatility_20d",
    "amihud_20d",
    "amihud_60d",
    "amount_shock_20d",
    "return_amount_corr_20d",
    "trend_efficiency_60d",
    "signed_amount_flow_20d",
    "log_float_market_cap",
)
FORMAL_CORE_PRICE_VOLUME_FACTORS: tuple[str, ...] = (
    "momentum_252_20",
    "reversal_5d",
    "volatility_20d",
    "amihud_20d",
    "turnover_20d",
    "log_float_market_cap",
)


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
    "turnover_20d": (("volume", "float_market_cap", "close"), (20,)),
    "turnover_60d": (("volume", "float_market_cap", "close"), (60,)),
    "turnover_volatility_20d": (("volume", "float_market_cap", "close"), (20,)),
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
    "turnover_change_20d": (("volume", "float_market_cap", "close"), (20,)),
    "amount_volatility_20d": (("amount",), (20,)),
    "volume_shock_20d": (("volume",), (20,)),
    "return_volume_corr_20d": (("period_return", "volume"), (20,)),
    "return_turnover_corr_20d": (
        ("period_return", "volume", "float_market_cap", "close"),
        (20,),
    ),
    "return_amount_imbalance_20d": (("period_return", "amount"), (20,)),
    "zero_amount_ratio_20d": (("amount",), (20,)),
    "log_total_market_cap": (("total_market_cap",), ()),
}


def price_volume_factor_definitions(
    *,
    market_id: MarketId,
    required_input_data_id: str,
    preset: str = "standard",
    factor_columns: Iterable[str] | None = None,
) -> tuple[FactorDefinition, ...]:
    """Bind canonical definitions without changing their identity across selections."""
    if preset in {"standard", "formal"}:
        selected = FORMAL_PRICE_VOLUME_FACTORS
    elif preset == "core":
        selected = FORMAL_CORE_PRICE_VOLUME_FACTORS
    else:
        raise ValueError("invalid_price_volume_preset")
    if factor_columns is not None:
        selected = tuple(dict.fromkeys(factor_columns))
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
                if column in FORMAL_CORE_PRICE_VOLUME_FACTORS
                else FactorSet.STANDARD,
                input_columns=inputs,
                base_input_columns=inputs,
                derived_input_columns=(),
                output_column=column,
                display_name=column,
                description=(
                    (
                        "Share of nonpositive amount (amount <= 0) over 20 complete "
                        "sessions; missing amounts propagate."
                    )
                    if column == "zero_amount_ratio_20d"
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
                window=lookback,
            )
        )
    return tuple(definitions)
