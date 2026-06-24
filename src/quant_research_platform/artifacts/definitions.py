"""Run output catalog definitions."""

from __future__ import annotations

from quant_research_platform.catalog.run_outputs import RunOutputCatalog
from quant_research_platform.contracts.run_outputs import (
    OutputScope,
    RunOutputDefinition,
)


def run_output_catalog() -> RunOutputCatalog:
    catalog = RunOutputCatalog()
    catalog.register(
        RunOutputDefinition(
            output_definition_id="stage_profile_diagnostic",
            display_name="Stage profile diagnostic",
            run_type="diagnostics",
            scope=OutputScope.LOCAL_ONLY,
        )
    )
    catalog.register(
        RunOutputDefinition(
            output_definition_id="prediction_outputs",
            display_name="Prediction outputs",
            run_type="prediction",
            scope=OutputScope.LOCAL_ONLY,
        )
    )
    catalog.register(
        RunOutputDefinition(
            output_definition_id="evaluation_outputs",
            display_name="Evaluation outputs",
            run_type="evaluation",
            scope=OutputScope.LOCAL_ONLY,
        )
    )
    catalog.register(
        RunOutputDefinition(
            output_definition_id="portfolio_outputs",
            display_name="Portfolio outputs",
            run_type="portfolio",
            scope=OutputScope.LOCAL_ONLY,
        )
    )
    return catalog


__all__ = ["run_output_catalog"]
