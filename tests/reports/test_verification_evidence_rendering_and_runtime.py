from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.reports.verification_evidence import (
    render_private_verification_summary_markdown,
    render_public_safe_evidence_markdown,
)
from quant_research_platform.runtime.profiling import (
    ProfileConfig,
    StageProfiler,
    _format_optional_mb,
    _render_stage_profile_report,
    active_stage_profiler,
    measure_stage,
)
from tests.helpers import frequency_ids


def test_public_safe_evidence_markdown_renders_status_rows() -> None:
    payload = {
        "privacy_level": "public_safe_aggregate",
        "contract_version": "0.1",
        "verification_kind": "private_research_data_aggregate",
        "verification_scope": "aggregate_public_safe_parity",
        "evidence_generation_mode": "local_private_research_public_safe_aggregate",
        "source_data_visibility": "private_local_only",
        "merge_ready": False,
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

    markdown = render_public_safe_evidence_markdown(payload)

    assert "factor value parity" in markdown
    assert "human approval required before merge-ready status" in markdown
    assert "`merge_ready | false`" in markdown


def test_private_verification_summary_markdown_renders_inventory() -> None:
    payload = {
        "privacy_level": "public_safe_aggregate",
        "contract_version": "0.1",
        "verification_kind": "private_research_data_aggregate_inventory",
        "source_data_visibility": "private_local_only",
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
        "merge_ready": False,
    }

    markdown = render_private_verification_summary_markdown(payload)

    assert "processed/a_share/other/demo | uncovered" in markdown
    assert "uncovered dataset roots require review" in markdown


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
    report = _render_stage_profile_report(frame)
    assert "metric.one" in report
    assert _format_optional_mb(None) == "unavailable"
    assert _format_optional_mb(1.23456) == "1.235 MB"


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

    empty_report = _render_stage_profile_report(pd.DataFrame())
    assert "No stages were recorded" in empty_report
