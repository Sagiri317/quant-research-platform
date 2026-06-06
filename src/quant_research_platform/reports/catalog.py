"""Report owner catalog factories."""

from __future__ import annotations

from quant_research_platform.catalog.reports import ReportCatalog
from quant_research_platform.contracts.reports import ReportDefinition


def report_catalog() -> ReportCatalog:
    catalog = ReportCatalog()
    catalog.register(
        ReportDefinition(
            report_definition_id="stage_profile_private_diagnostic",
            display_name="Private stage profile diagnostic",
            run_type="diagnostics",
        )
    )
    catalog.register(
        ReportDefinition(
            report_definition_id="prediction_bundle_private",
            display_name="Private prediction bundle",
            run_type="prediction",
        )
    )
    catalog.register(
        ReportDefinition(
            report_definition_id="evaluation_report_private",
            display_name="Private evaluation report",
            run_type="evaluation",
        )
    )
    return catalog


__all__ = ["report_catalog"]
