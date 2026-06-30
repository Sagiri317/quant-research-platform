"""Price-volume factor definition builders."""

from __future__ import annotations

from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId

from .columns import (
    CORE_PRICE_VOLUME_FACTORS,
    PRICE_VOLUME_FACTOR_FAMILY,
    SMOKE_PRICE_VOLUME_FACTORS,
    _lookback_window,
    _required_windows,
    price_volume_factor_columns,
    price_volume_factor_input_columns,
)


def price_volume_factor_definitions(
    *,
    market_id: MarketId,
    required_input_data_id: str,
    preset: str = "standard",
) -> tuple[FactorDefinition, ...]:
    """Return factor definitions for one market binding."""

    definitions: list[FactorDefinition] = []
    for factor_id in price_volume_factor_columns(preset=preset):
        base_inputs = price_volume_factor_input_columns(factor_id)
        lookback_window = _lookback_window(factor_id)
        definitions.append(
            FactorDefinition(
                factor_id=factor_id,
                market_id=market_id,
                factor_set=_factor_set_for_column(factor_id),
                input_columns=base_inputs,
                base_input_columns=base_inputs,
                derived_input_columns=(),
                output_column=factor_id,
                display_name=factor_id,
                description=(
                    "Daily price-volume factor registered by the QRP 0.1 "
                    "factor catalog."
                ),
                family=PRICE_VOLUME_FACTOR_FAMILY,
                required_windows=_required_windows(factor_id),
                lookback_window=lookback_window,
                grain=DataGrain.ASSET_TIMESTAMP,
                timing_semantics="same_session_close_available_at_next_session",
                null_policy="preserve_nulls_after_min_periods",
                source_layer="processed",
                required_input_data_id=required_input_data_id,
                window=lookback_window,
            )
        )
    return tuple(definitions)


def _factor_set_for_column(factor_id: str) -> FactorSet:
    if factor_id in SMOKE_PRICE_VOLUME_FACTORS:
        return FactorSet.SMOKE
    if factor_id in CORE_PRICE_VOLUME_FACTORS:
        return FactorSet.CORE
    return FactorSet.STANDARD

__all__ = ["price_volume_factor_definitions"]
