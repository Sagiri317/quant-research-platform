"""A-share-specific factor definitions."""

from __future__ import annotations

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.markets.implementations.a_share.ids import (
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
    return price_volume_factor_definitions(
        market_id=MARKET_ID,
        required_input_data_id=market_panel_data_id(),
        preset=PRICE_VOLUME_STANDARD_PRESET,
    )


def factor_catalog() -> FactorCatalog:
    return FactorCatalog(factor_definitions())


__all__ = [
    "factor_catalog",
    "factor_definitions",
]
