"""A-share-specific factor definitions."""

from __future__ import annotations

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.markets.implementations.a_share.ids import (
    FUNDAMENTAL_PIT_PROCESSED_DATA_ID,
    MARKET_ID,
    market_panel_data_id,
)
from quant_research_platform.research.factors.price_volume.definitions import (
    price_volume_factor_definitions,
)
from quant_research_platform.research.factors.price_volume.presets import (
    PRICE_VOLUME_STANDARD_PRESET,
)


def factor_definitions() -> tuple[FactorDefinition, ...]:
    return (
        *price_volume_factor_definitions(
            market_id=MARKET_ID,
            required_input_data_id=market_panel_data_id(),
            preset=PRICE_VOLUME_STANDARD_PRESET,
        ),
        *fundamental_factor_definitions(),
    )


def factor_catalog() -> FactorCatalog:
    return FactorCatalog(factor_definitions())


def fundamental_factor_definitions() -> tuple[FactorDefinition, ...]:
    facts = (
        ("earnings_yield", ("net_income", "market_cap")),
        ("book_to_market", ("total_equity", "market_cap")),
        ("roe", ("net_income", "total_equity")),
        (
            "operating_cash_flow_to_assets",
            ("operating_cash_flow", "total_assets"),
        ),
        ("asset_growth", ("total_assets", "previous_total_assets")),
        ("leverage", ("total_liabilities", "total_assets")),
    )
    return tuple(
        FactorDefinition(
            factor_id=f"fundamental_raw_{name}",
            market_id=MARKET_ID,
            factor_set=FactorSet.SMOKE,
            input_columns=inputs,
            output_column=f"fundamental_raw_{name}",
            family="fundamental",
            timing_semantics="point_in_time_available_at",
            null_policy="preserve_nulls",
            source_layer="processed",
            required_input_data_id=FUNDAMENTAL_PIT_PROCESSED_DATA_ID,
        )
        for name, inputs in facts
    )
