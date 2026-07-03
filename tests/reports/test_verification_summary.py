from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.reports.execution_metrics import (
    format_optional_mb,
    render_stage_profile_report,
)
from quant_research_platform.reports.verification_summary import (
    render_local_verification_inventory_markdown,
    render_review_safe_aggregate_evidence_markdown,
    render_run_summary_markdown,
)
from quant_research_platform.runtime.profiling import (
    ProfileConfig,
    StageProfiler,
    active_stage_profiler,
    measure_stage,
)
from quant_research_platform.workflows.local_verification_layout import (
    LocalVerificationLayout,
)
from tests.helpers import frequency_ids


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


def test_local_verification_layout_uses_normal_internal_directory_names() -> None:
    run_root = Path("local-verification-output") / "s1"
    layout = LocalVerificationLayout.from_run_root(run_root)

    assert layout.run_root == run_root
    assert layout.run_json_path == run_root / "run.json"
    assert layout.data_root == run_root / "data"
    assert layout.interim_root == run_root / "data" / "interim"
    assert layout.processed_root == run_root / "data" / "processed"
    assert layout.research_root == run_root / "data" / "research"
    assert layout.outputs_root == run_root / "outputs"
    assert layout.logs_root == run_root / "logs"
    assert layout.summary_md_path == run_root / "summary.md"

    assert set(layout.__dataclass_fields__) == {"run_root"}


def test_review_safe_evidence_markdown_renders_status_rows() -> None:
    payload = {
        "output_scope": "REVIEW_SAFE",
        "schema_id": "qrp-a-share-aggregate-evidence",
        "verification_kind": "local_result_aggregate",
        "verification_scope": "aggregate_review_safe_parity",
        "evidence_generation_mode": "local_result_review_safe_aggregate",
        "source_data_visibility": "local_only",
        "handoff_ready": False,
        "areas": [
            {
                "area": "factor_value_parity",
                "row_count": 10,
                "column_count": 3,
                "period": "2024-01-01 through 2024-01-31",
                "human_approval_required": False,
            },
            {
                "area": "label_value_parity",
                "row_count": 8,
                "column_count": 3,
                "period": "not_applicable",
                "human_approval_required": True,
            },
        ],
    }

    markdown = render_review_safe_aggregate_evidence_markdown(payload)

    assert "factor value parity" in markdown
    assert "human approval required before final review status" in markdown
    assert "`handoff_ready | false`" in markdown
    assert "`schema_id=qrp-a-share-aggregate-evidence`" in markdown
    assert "Machine-readable run facts are recorded in run.json" in markdown


def test_local_verification_inventory_markdown_renders_inventory() -> None:
    payload = {
        "output_scope": "REVIEW_SAFE",
        "schema_id": "qrp-a-share-verification-inventory",
        "verification_kind": "local_result_aggregate_inventory",
        "source_data_visibility": "local_only",
        "dataset_inventory": [
            {
                "logical_root": frequency_ids.market_panel_logical_root(),
                "covered_by_evidence_area": "panel",
                "row_count": 10,
                "column_count": 3,
                "period": "2024-01-01 through 2024-01-31",
            },
            {
                "logical_root": "processed/a_share/other/demo",
                "covered_by_evidence_area": "",
                "row_count": 0,
                "column_count": 0,
                "period": "not_applicable",
            },
        ],
        "covered_roots": [frequency_ids.market_panel_logical_root()],
        "uncovered_roots": ["processed/a_share/other/demo"],
        "allowed_unverified_roots": [],
        "handoff_ready": False,
    }

    markdown = render_local_verification_inventory_markdown(payload)

    assert "processed/a_share/other/demo | uncovered" in markdown
    assert "uncovered dataset roots require review" in markdown
    assert "Machine-readable run facts are recorded in run.json" in markdown


def test_stage_profiler_records_and_renders() -> None:
    ticks = iter([1.0, 1.5, 2.0, 3.0])
    profiler = StageProfiler(
        ProfileConfig(enabled=True, sample_memory=False),
        clock=lambda: next(ticks),
        metadata={"run_id": "synthetic"},
    )

    with profiler:
        assert active_stage_profiler() is profiler
        assert measure_stage("metric", "one", lambda: "ok", sample_memory=False) == "ok"
        with profiler.stage("metric", "two", sample_memory=False):
            pass
    assert active_stage_profiler() is None

    frame = profiler.to_frame()
    assert frame["n_calls"].tolist() == [1, 1]
    assert frame["run_id"].tolist() == ["synthetic", "synthetic"]
    report = render_stage_profile_report(frame)
    assert "metric.one" in report
    assert format_optional_mb(None) == "unavailable"
    assert format_optional_mb(1.23456) == "1.235 MB"


def test_stage_profiler_disabled_and_invalid_config_paths() -> None:
    with pytest.raises(TypeError, match="enabled must be bool"):
        ProfileConfig(enabled="yes")  # type: ignore[arg-type]
    profiler = StageProfiler(ProfileConfig(enabled=False))
    with profiler.stage("metric", "disabled"):
        pass
    assert profiler.to_frame().empty
    assert measure_stage("metric", "standalone", lambda: 7) == 7
    with pytest.raises(ValueError, match="category must be"):
        with profiler.stage("", "bad"):
            pass

    empty_report = render_stage_profile_report(pd.DataFrame())
    assert "No stages were recorded" in empty_report
