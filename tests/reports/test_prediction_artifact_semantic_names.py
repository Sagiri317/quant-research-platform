from __future__ import annotations

from pathlib import Path

from quant_research_platform.reports.catalog import report_catalog


def test_report_catalog_uses_prediction_artifacts_definition() -> None:
    definition = report_catalog().get("prediction_artifacts_private")

    assert definition.display_name == "Private prediction artifacts"
    assert "prediction_bundle_private" not in {
        item.report_definition_id for item in report_catalog().list()
    }


def test_prediction_report_errors_use_artifact_semantics() -> None:
    source = Path(
        "src/quant_research_platform/reports/report_artifacts.py"
    ).read_text(encoding="utf-8")

    assert "prediction_artifacts_private" in source
    assert "missing_prediction_artifact_columns" in source
    assert "unregistered_prediction_artifact_columns" in source
    assert "prediction_bundle_private" not in source
    assert "missing_prediction_bundle_columns" not in source
    assert "unregistered_prediction_bundle_columns" not in source
