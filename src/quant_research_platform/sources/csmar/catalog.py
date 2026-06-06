"""CSMAR source definition."""

from __future__ import annotations

from quant_research_platform.catalog.sources import SourceDefinition
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.sources.csmar.ids import CSMAR_SOURCE_ID


def csmar_source_definition() -> SourceDefinition:
    return SourceDefinition(
        source_id=SourceId(CSMAR_SOURCE_ID),
        provider_id=CSMAR_SOURCE_ID,
        display_name="CSMAR source adapter",
        implementation_package="quant_research_platform.sources.csmar",
    )


__all__ = ["csmar_source_definition"]
