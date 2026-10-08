"""The formal four-factor expectations block."""

from __future__ import annotations

from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId

EXPECTATIONS_FACTORS = (
    "forward_earnings_yield_fy1",
    "forward_earnings_yield_fy2",
    "forecast_dispersion_fy1",
    "log_analyst_coverage",
)
EXPECTATIONS_MECHANISM_FACTORS = (
    "forecast_dispersion_price_scaled_fy1",
    "median_forecast_age_days_fy1",
    "newly_updated_institution_fraction_fy1",
)
ALL_EXPECTATIONS_FACTORS = (*EXPECTATIONS_FACTORS, *EXPECTATIONS_MECHANISM_FACTORS)
EXPECTATIONS_INPUT_COLUMNS = (
    "close",
    "forecast_eps",
    "forecast_end_date",
    "institution_id",
    "declare_date",
)


def expectations_factor_definitions(
    *, market_id: MarketId, required_input_data_id: str
) -> tuple[FactorDefinition, ...]:
    return tuple(
        FactorDefinition(
            factor_id=f"expectations_{name}",
            market_id=market_id,
            factor_set=FactorSet.CORE,
            input_columns=EXPECTATIONS_INPUT_COLUMNS,
            base_input_columns=EXPECTATIONS_INPUT_COLUMNS,
            output_column=name,
            display_name=name,
            description="PIT analyst forecast factor.",
            family="expectations",
            grain=DataGrain.ASSET_TIMESTAMP,
            timing_semantics="point_in_time_available_at",
            null_policy="preserve_nulls",
            source_layer="processed",
            required_input_data_id=required_input_data_id,
        )
        for name in ALL_EXPECTATIONS_FACTORS
    )
