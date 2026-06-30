from __future__ import annotations

from quant_research_platform.workflows.diagnostics.validation_summary import (
    build_validation_summary,
)


def test_build_validation_summary_consumes_embedded_layer_summaries() -> None:
    summary = build_validation_summary(
        run_id="demo_run",
        dataset_roots=(),
        layer_summaries=(
            {
                "logical_layer": "research",
                "layer_validation_status": "pass",
            },
        ),
        generated_text_paths=(),
    )

    assert summary["status"] == "PASS"
    assert summary["data_layer_summary_completeness"] == {
        "missing": [],
        "checked_count": 1,
    }
