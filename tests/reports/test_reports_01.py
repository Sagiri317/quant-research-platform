from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.reports.bundles import write_stage_profile_bundle
from quant_research_platform.reports.summaries import (
    ReportRunIndexRecord,
    ReportSummary,
)
from quant_research_platform.runtime.profiling import ProfileConfig, StageProfiler


def test_stage_profile_bundle_writes_only_inside_private_report_boundary(tmp_path):
    with StageProfiler(ProfileConfig(enabled=True)) as profiler:
        with profiler.stage("evaluation", "ic", sample_memory=False):
            pass

    output_dir = (
        tmp_path / "reports" / "private_real" / "a_share" / "diagnostics" / "run"
    )
    paths = profiler.write_artifacts(output_dir)

    assert paths["csv"].is_file()
    assert paths["json"].is_file()
    assert paths["report"].is_file()
    assert (output_dir / "manifest.json").is_file()


def test_stage_profile_bundle_rejects_public_output(tmp_path):
    profiler = StageProfiler(ProfileConfig(enabled=True))

    with pytest.raises(ValueError, match="outside_private_boundary"):
        profiler.write_artifacts(tmp_path / "public")


def test_stage_profile_bundle_requires_approved_report_spec(tmp_path):
    frame = pd.DataFrame.from_records(
        [
            {
                "category": r"C:\Users\DELL\private",
                "stage": "600000.SH",
                "seconds": 0.1,
                "n_calls": 1,
            }
        ]
    )
    output_dir = (
        tmp_path / "reports" / "private_real" / "a_share" / "diagnostics" / "run"
    )

    with pytest.raises(ValueError, match="invalid_stage_profile_report_definition"):
        write_stage_profile_bundle(
            frame,
            spec=ReportSpec(
                report_id="stage_profile",
                report_definition_id="other_report",
                run_id="run",
                market_id=MarketId("a_share"),
                run_type="diagnostics",
            ),
            output_root=output_dir,
            report_text="CSMAR vendor path /mnt/data/raw",
        )


def test_stage_profile_bundle_sanitizes_outputs(tmp_path):
    frame = pd.DataFrame.from_records(
        [
            {
                "category": r"C:\Users\DELL\private",
                "stage": "600000.SH",
                "seconds": 0.1,
                "n_calls": 1,
            }
        ]
    )
    output_dir = (
        tmp_path / "reports" / "private_real" / "a_share" / "diagnostics" / "run"
    )

    paths = write_stage_profile_bundle(
        frame,
        spec=ReportSpec(
            report_id="stage_profile",
            report_definition_id="stage_profile_private_diagnostic",
            run_id="run",
            market_id=MarketId("a_share"),
            run_type="diagnostics",
        ),
        output_root=output_dir,
        report_text="CSMAR vendor path /mnt/data/raw token=abcdef123456",
    )

    combined = (
        paths["csv"].read_text(encoding="utf-8")
        + paths["json"].read_text(encoding="utf-8")
        + paths["report"].read_text(encoding="utf-8")
        + json.dumps(json.loads((output_dir / "manifest.json").read_text("utf-8")))
    )
    assert "C:" not in combined
    assert "600000" not in combined
    assert "CSMAR" not in combined
    assert "abcdef123456" not in combined


def test_report_summary_records_are_database_ready_and_sanitized():
    summary = ReportSummary(
        run_id="run",
        market_id="a_share",
        run_type="evaluation",
        status="completed",
        aggregate_metadata={"path": r"C:\Users\DELL\000001.SZ"},
    )
    index = ReportRunIndexRecord(
        run_id="run",
        market_id="a_share",
        run_type="evaluation",
        status="completed",
        report_definition_id="evaluation_report_private",
    )

    assert "contract_version" in summary.to_record()
    assert "C:" not in str(summary.to_record())
    assert "000001" not in str(summary.to_record())
    assert index.to_record()["report_definition_id"] == "evaluation_report_private"
