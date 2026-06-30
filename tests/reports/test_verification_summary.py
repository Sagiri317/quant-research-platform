from __future__ import annotations

from quant_research_platform.reports.verification_summary import (
    render_run_summary_markdown,
)


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
