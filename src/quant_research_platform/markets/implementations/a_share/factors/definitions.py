"""A-share-specific factor definitions."""

from __future__ import annotations

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_MARKET_ID,
    DAILY_PANEL_PROCESSED_DATA_ID,
)
from quant_research_platform.research.factors.price_volume import (
    PRICE_VOLUME_STANDARD_PRESET,
    price_volume_factor_definitions,
)


def a_share_factor_definitions() -> tuple[FactorDefinition, ...]:
    return price_volume_factor_definitions(
        market_id=A_SHARE_MARKET_ID,
        required_input_data_id=DAILY_PANEL_PROCESSED_DATA_ID,
        preset=PRICE_VOLUME_STANDARD_PRESET,
    )


def a_share_factor_catalog() -> FactorCatalog:
    catalog = FactorCatalog()
    for definition in a_share_factor_definitions():
        catalog.register(definition)
    return catalog


__all__ = [
    "a_share_factor_catalog",
    "a_share_factor_definitions",
]
