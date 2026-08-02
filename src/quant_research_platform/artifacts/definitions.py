"""Output catalog definitions."""

from __future__ import annotations

from quant_research_platform.catalog.outputs import OutputCatalog
from quant_research_platform.contracts.outputs import (
    OutputDefinition,
    OutputScope,
)


def output_catalog() -> OutputCatalog:
    catalog = OutputCatalog()
    catalog.register(
        OutputDefinition(
            output_definition_id="prediction_outputs",
            display_name="Prediction outputs",
            run_type="prediction",
            scope=OutputScope.LOCAL_ONLY,
        )
    )
    catalog.register(
        OutputDefinition(
            output_definition_id="evaluation_outputs",
            display_name="Evaluation outputs",
            run_type="evaluation",
            scope=OutputScope.LOCAL_ONLY,
        )
    )
    catalog.register(
        OutputDefinition(
            output_definition_id="portfolio_outputs",
            display_name="Portfolio outputs",
            run_type="portfolio",
            scope=OutputScope.LOCAL_ONLY,
        )
    )
    return catalog


__all__ = ["output_catalog"]
