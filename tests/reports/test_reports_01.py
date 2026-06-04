from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.reports import ReportSpec
from quant_research_platform.reports.bundles import write_stage_profile_bundle
from quant_research_platform.reports.index import ReportRunIndex
from quant_research_platform.reports.manifests import ReportManifest
from quant_research_platform.reports.summaries import (
    ManifestSummary,
    ParityEvidenceSummary,
    ReportRunIndexRecord,
    ReportSummary,
)
from quant_research_platform.runtime.profiling import ProfileConfig, StageProfiler
from quant_research_platform.storage.layout import StorageLayout


def test_stage_profile_bundle_writes_only_inside_private_report_boundary(tmp_path):
    with StageProfiler(ProfileConfig(enabled=True)) as profiler:
        with profiler.stage("evaluation", "ic", sample_memory=False):
            pass

    output_dir = profiler.write_artifacts(
        tmp_path / "reports" / "private_real" / "a_share" / "diagnostics" / "run"
    )

    assert output_dir["csv"].is_file()
    assert output_dir["json"].is_file()
    assert output_dir["report"].is_file()
    assert (
        tmp_path
        / "reports"
        / "private_real"
        / "a_share"
        / "diagnostics"
        / "run"
        / "manifest.json"
    ).is_file()


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
    with pytest.raises(ValueError, match="report_spec_mismatch"):
        write_stage_profile_bundle(
            frame,
            spec=ReportSpec(
                report_id="stage_profile",
                report_definition_id="other_report",
                run_id="run",
                market_id=MarketId("a_share"),
                run_type="diagnostics",
            ),
            layout=StorageLayout(project_root=tmp_path),
            report_text="CSMAR vendor path /mnt/data/raw",
        )


def test_report_spec_has_no_allowed_output_root_field() -> None:
    assert "allowed_output_root" not in ReportSpec.__dataclass_fields__


def test_stage_profile_bundle_uses_report_catalog_gate(tmp_path):
    frame = pd.DataFrame.from_records(
        [{"category": "evaluation", "stage": "ic", "seconds": 0.1, "n_calls": 1}]
    )

    with pytest.raises(ValueError, match="report_spec_mismatch"):
        write_stage_profile_bundle(
            frame,
            spec=ReportSpec(
                report_id="stage_profile",
                report_definition_id="stage_profile_private_diagnostic",
                run_id="run",
                market_id=MarketId("a_share"),
                run_type="prediction",
            ),
            layout=StorageLayout(project_root=tmp_path),
            report_text="safe",
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
    paths = write_stage_profile_bundle(
        frame,
        spec=ReportSpec(
            report_id="stage_profile",
            report_definition_id="stage_profile_private_diagnostic",
            run_id="run",
            market_id=MarketId("a_share"),
            run_type="diagnostics",
        ),
        layout=StorageLayout(project_root=tmp_path),
        report_text="CSMAR vendor path /mnt/data/raw token=abcdef123456",
    )
    output_dir = (
        tmp_path
        / "reports"
        / "private_real"
        / "a_share"
        / "diagnostics"
        / "run"
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


def test_stage_profile_bundle_rejects_arbitrary_output_root(tmp_path):
    frame = pd.DataFrame.from_records(
        [{"category": "evaluation", "stage": "ic", "seconds": 0.1, "n_calls": 1}]
    )

    with pytest.raises(ValueError, match="report_output_root_mismatch"):
        write_stage_profile_bundle(
            frame,
            spec=ReportSpec(
                report_id="stage_profile",
                report_definition_id="stage_profile_private_diagnostic",
                run_id="run",
                market_id=MarketId("a_share"),
                run_type="diagnostics",
            ),
            layout=StorageLayout(project_root=tmp_path),
            output_root=tmp_path / "reports" / "private_real" / "a_share" / "other",
            report_text="safe",
        )


def test_stage_profile_bundle_rejects_existing_run_root_without_target_writes(
    tmp_path,
):
    frame = pd.DataFrame.from_records(
        [{"category": "evaluation", "stage": "ic", "seconds": 0.1, "n_calls": 1}]
    )
    run_root = (
        tmp_path
        / "reports"
        / "private_real"
        / "a_share"
        / "diagnostics"
        / "run"
    )
    run_root.mkdir(parents=True)
    sentinel = run_root / "sentinel.txt"
    sentinel.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output_already_exists"):
        write_stage_profile_bundle(
            frame,
            spec=ReportSpec(
                report_id="stage_profile",
                report_definition_id="stage_profile_private_diagnostic",
                run_id="run",
                market_id=MarketId("a_share"),
                run_type="diagnostics",
            ),
            layout=StorageLayout(project_root=tmp_path),
            report_text="safe",
        )

    assert sentinel.read_text(encoding="utf-8") == "existing\n"
    assert not (run_root / "stage_profile.csv").exists()
    assert not (run_root / "manifest.json").exists()
    assert list(run_root.parent.glob("*backup*")) == []


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


def test_report_manifest_and_summaries_validate_run_id() -> None:
    with pytest.raises(ValueError, match="invalid_run_id"):
        ReportManifest(run_id="run 01")
    with pytest.raises(ValueError, match="invalid_run_id"):
        ReportSummary(
            run_id="600000.sh",
            market_id="a_share",
            run_type="evaluation",
            status="completed",
        )
    with pytest.raises(ValueError, match="invalid_run_id"):
        ReportRunIndexRecord(
            run_id="data/raw/run",
            market_id="a_share",
            run_type="evaluation",
            status="completed",
            report_definition_id="evaluation_report_private",
        )


def test_report_index_records_do_not_require_filesystem_scan():
    run_index = ReportRunIndex()
    record = ReportRunIndexRecord(
        run_id="run",
        market_id="a_share",
        run_type="prediction",
        status="completed",
        report_definition_id="prediction_bundle_private",
    )

    run_index.register(record)

    assert run_index.get("run") == record
    with pytest.raises(KeyError, match="unknown_report_run"):
        run_index.get("missing")


def test_manifest_and_parity_summaries_are_public_safe():
    manifest = ManifestSummary(
        artifact_id="artifact",
        market_id="a_share",
        status="completed",
        aggregate_metadata={"path": r"C:\Users\DELL\data\raw\csmar\000001.SZ"},
    )
    parity = ParityEvidenceSummary(
        evidence_id="evidence",
        market_id="a_share",
        data_product_id="a_share_daily_bar",
        privacy_level="public_safe_aggregate",
        status="accepted",
        aggregate_metadata={"provider_id": "csmar"},
    )

    combined = str(manifest.to_record()) + str(parity.to_record())

    assert "C:" not in combined
    assert "000001" not in combined
    assert "csmar" not in combined.lower()
    assert "provider_id" not in combined
