"""Crypto microstructure factor bindings."""

from __future__ import annotations

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.markets.implementations.crypto.ids import (
    MARKET_ID,
    MARKET_PANEL_DATA_ID,
)
from quant_research_platform.research.factors.microstructure.definitions import (
    microstructure_factor_definitions,
)


def factor_catalog() -> FactorCatalog:
    return FactorCatalog(
        microstructure_factor_definitions(
            market_id=MARKET_ID,
            required_input_data_id=MARKET_PANEL_DATA_ID,
        )
    )
