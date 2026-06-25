from __future__ import annotations

from pathlib import Path

from quant_research_platform.artifacts.definitions import output_catalog


def test_output_catalog_uses_prediction_outputs_definition() -> None:
    definition = output_catalog().get("prediction_outputs")

    assert definition.display_name == "Prediction outputs"


def test_prediction_report_errors_use_artifact_semantics() -> None:
    source = Path("src/quant_research_platform/artifacts/outputs.py").read_text(
        encoding="utf-8"
    )

    assert "prediction_outputs" in source
    assert "missing_prediction_output_columns" in source
    assert "unregistered_prediction_output_columns" in source
