from __future__ import annotations

from quant_research_platform.reports.verification_evidence import (
    build_validation_summary,
    render_run_summary_markdown,
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
    assert summary["layer_manifest_completeness"] == {
        "missing": [],
        "checked_count": 1,
    }


def test_render_run_summary_markdown_uses_component_statuses() -> None:
    payload = {
        "run_id": "demo_run",
        "status": {
            "overall": "NOT_RUN",
            "validation": "PASS",
            "smoke": "PASS",
            "compare": "NOT_RUN",
            "share_safety": "PASS",
            "full_horizon": "NOT_RUN",
            "validation_matrix": "NOT_RUN",
        },
        "data": {"layers": [{}, {}]},
        "outputs": [{}, {}, {}],
    }

    text = render_run_summary_markdown(payload)

    assert "# Local Verification Run Summary" in text
    assert "- overall: `NOT_RUN`" in text
    assert "- validation: `PASS`" in text
    assert "- outputs: `3`" in text
