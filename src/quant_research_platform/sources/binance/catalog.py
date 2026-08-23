"""Binance source definition."""

from __future__ import annotations

from quant_research_platform.catalog.sources import SourceDefinition
from quant_research_platform.sources.binance.ids import SOURCE_ID


def source_definition() -> SourceDefinition:
    return SourceDefinition(
        source_id=SOURCE_ID,
        provider_id=str(SOURCE_ID),
        display_name="Binance public market data source",
        implementation_package="quant_research_platform.sources.binance",
    )
