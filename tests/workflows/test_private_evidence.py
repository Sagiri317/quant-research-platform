from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_research_platform.storage.manifests import write_internal_artifact_manifest
from quant_research_platform.workflows import private_evidence as module
from quant_research_platform.workflows.private_evidence import (
    PrivateEvidenceRunConfig,
    run_private_evidence,
)

LEGACY_RSS_PEAK_FIELD = "rss_" + "peak_mb"


def test_private_evidence_workflow_rejects_output_root_outside_run(tmp_path):
    config = PrivateEvidenceRunConfig(
        output_root=tmp_path / "outside",
        start_date="2014-01-01",
        end_date="2016-12-31",
    )

    with pytest.raises(ValueError, match="output_root_must_be_under_run"):
        run_private_evidence(config)


def test_private_evidence_workflow_rejects_existing_output_root(tmp_path):
    output_root = tmp_path / ".run" / "s1"
    output_root.mkdir(parents=True)
    config = PrivateEvidenceRunConfig(
        output_root=output_root,
        start_date="2014-01-01",
        end_date="2016-12-31",
    )

    with pytest.raises(ValueError, match="output_root_already_exists"):
        run_private_evidence(config)


def test_private_evidence_workflow_rejects_invalid_date_range(tmp_path):
    config = PrivateEvidenceRunConfig(
        output_root=tmp_path / ".run" / "s1",
        start_date="2016-12-31",
        end_date="2014-01-01",
    )

    with pytest.raises(ValueError, match="start_date_after_end_date"):
        run_private_evidence(config)


def test_private_evidence_run_id_can_be_explicit_long_semantic_value(tmp_path):
    config = PrivateEvidenceRunConfig(
        output_root=tmp_path / ".run" / "s1",
        start_date="2014-01-01",
        end_date="2016-12-31",
        run_id="private-evidence-smoke-a-share-2014-2016",
    )

    assert config.run_id == "private-evidence-smoke-a-share-2014-2016"


def test_private_evidence_workflow_runs_injected_generation_and_writes_artifacts(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []
    evidence_years: list[tuple[int, ...]] = []
    evaluation_run_ids: list[str] = []
    prediction_ids: list[str] = []

    def import_interim(**kwargs):
        calls.append("interim")
        artifact_root = (
            kwargs["layout"].project_root
            / "data"
            / "interim"
            / "csmar"
            / "daily_market"
        )
        (artifact_root / "data").mkdir(parents=True)
        _write_artifact_manifest(artifact_root, "daily_market", "interim")
        return object()

    def build_processed(**kwargs):
        calls.append("processed")
        processed = Path(kwargs["processed_root"])
        artifact_root = processed / "factors" / "price_volume"
        (artifact_root / "data").mkdir(parents=True)
        _write_artifact_manifest(artifact_root, "price_volume", "processed")
        return type(
            "Result",
            (),
            {
                "row_counts": {"price_volume": 2},
                "data_roots": {
                    "price_volume": processed
                    / "factors"
                    / "price_volume"
                    / "data"
                },
            },
        )()

    def assemble_dataset(**kwargs):
        calls.append("research")
        processed = Path(kwargs["processed_root"])
        artifact_root = (
            processed
            / "research_datasets"
            / "a_share_price_volume_forward_returns"
        )
        (artifact_root / "data").mkdir(parents=True)
        _write_artifact_manifest(
            artifact_root,
            "a_share_price_volume_forward_returns",
            "research",
        )
        return artifact_root / "data"

    monkeypatch.setattr(module, "a_share_csmar_import_interim_data", import_interim)
    monkeypatch.setattr(module, "build_a_share_processed_layers", build_processed)
    monkeypatch.setattr(
        module,
        "assemble_research_dataset_from_processed",
        assemble_dataset,
    )
    monkeypatch.setattr(
        module,
        "run_evaluation_from_research_dataset",
        lambda **kwargs: (
            calls.append("evaluation")
            or evidence_years.append(kwargs["years"])
            or evaluation_run_ids.append(kwargs["run_id"])
            or Path("evaluation")
        ),
    )
    monkeypatch.setattr(
        module,
        "run_prediction_from_research_dataset",
        lambda **kwargs: (
            calls.append("prediction")
            or evidence_years.append(kwargs["years"])
            or prediction_ids.append(kwargs["prediction_id"])
            or Path("prediction")
        ),
    )

    output_root = tmp_path / ".run" / "s1"
    result = run_private_evidence(
        PrivateEvidenceRunConfig(
            output_root=output_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
        )
    )

    assert calls == ["interim", "processed", "research", "evaluation", "prediction"]
    assert evidence_years == [(2014,), (2014,)]
    assert evaluation_run_ids == ["a_share_mom_20d_forward_return_20d_2014"]
    assert prediction_ids == ["a_share_linear_mom_20d_prediction_2014"]
    assert result.status == "PASS"
    assert result.run_id == "s1"
    assert (output_root / "config" / "run_config.json").is_file()
    assert (output_root / "manifests" / "run_manifest.json").is_file()
    assert (output_root / "timings" / "timings.json").is_file()
    assert (output_root / "validations" / "validation_summary.json").is_file()
    assert (output_root / "data" / "evidence" / "evidence_summary.json").is_file()
    for layer in ("interim", "processed", "research", "evidence"):
        layer_manifest = (
            output_root / "data" / layer / "_manifests" / "layer_manifest.json"
        )
        assert layer_manifest.is_file()
        assert list((output_root / "data" / layer).rglob("layer_manifest.json")) == [
            layer_manifest
        ]
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (output_root / "validations" / "validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    timings = json.loads(
        (output_root / "timings" / "timings.json").read_text(encoding="utf-8")
    )
    timings_csv = (output_root / "timings" / "timings.csv").read_text(
        encoding="utf-8"
    )
    assert summary["status"] == "PASS"
    assert summary["status"] == validation["status"]
    assert summary["run_id"] == "s1"
    timing_steps = [record["step_name"] for record in timings["records"]]
    summary_timing_steps = [
        record["step_name"] for record in summary["timings"]["records"]
    ]
    validation_record = next(
        record
        for record in timings["records"]
        if record["step_name"] == "validation_summary"
    )
    assert "validation_summary" in timing_steps
    assert "validation_summary" in summary_timing_steps
    assert "validation_summary" in timings_csv
    assert "rss_start_end_max_mb" in validation_record
    assert LEGACY_RSS_PEAK_FIELD not in validation_record


def test_private_evidence_workflow_preserves_dataset_manifests(
    tmp_path,
    monkeypatch,
):
    def import_interim(**kwargs):
        artifact_root = (
            kwargs["layout"].project_root
            / "data"
            / "interim"
            / "csmar"
            / "daily_market"
        )
        (artifact_root / "data").mkdir(parents=True)
        _write_artifact_manifest(
            artifact_root,
            "daily_market",
            "interim",
            marker="artifact-writer-owned",
        )

    monkeypatch.setattr(module, "a_share_csmar_import_interim_data", import_interim)
    monkeypatch.setattr(
        module,
        "build_a_share_processed_layers",
        lambda **kwargs: type("Result", (), {"row_counts": {}})(),
    )
    monkeypatch.setattr(
        module,
        "assemble_research_dataset_from_processed",
        lambda **kwargs: Path("research"),
    )
    monkeypatch.setattr(
        module,
        "run_evaluation_from_research_dataset",
        lambda **kwargs: Path("evaluation"),
    )
    monkeypatch.setattr(
        module,
        "run_prediction_from_research_dataset",
        lambda **kwargs: Path("prediction"),
    )

    output_root = tmp_path / ".run" / "s8"
    result = run_private_evidence(
        PrivateEvidenceRunConfig(
            output_root=output_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
        )
    )
    manifest_path = (
        output_root
        / "data"
        / "interim"
        / "csmar"
        / "daily_market"
        / "_manifest"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result.status == "PASS"
    assert manifest["marker"] == "artifact-writer-owned"
    assert "factor_nan_profile" not in manifest


def test_private_evidence_workflow_fails_when_dataset_manifest_is_missing(
    tmp_path,
    monkeypatch,
):
    def import_interim(**kwargs):
        artifact_root = (
            kwargs["layout"].project_root
            / "data"
            / "interim"
            / "csmar"
            / "daily_market"
        )
        (artifact_root / "data").mkdir(parents=True)

    monkeypatch.setattr(module, "a_share_csmar_import_interim_data", import_interim)
    monkeypatch.setattr(
        module,
        "build_a_share_processed_layers",
        lambda **kwargs: type("Result", (), {"row_counts": {}})(),
    )
    monkeypatch.setattr(
        module,
        "assemble_research_dataset_from_processed",
        lambda **kwargs: Path("research"),
    )
    monkeypatch.setattr(
        module,
        "run_evaluation_from_research_dataset",
        lambda **kwargs: Path("evaluation"),
    )
    monkeypatch.setattr(
        module,
        "run_prediction_from_research_dataset",
        lambda **kwargs: Path("prediction"),
    )

    output_root = tmp_path / ".run" / "s9"
    result = run_private_evidence(
        PrivateEvidenceRunConfig(
            output_root=output_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
        )
    )
    validation = json.loads(
        (output_root / "validations" / "validation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))

    assert result.status == "FAIL"
    assert "dataset_manifest_missing" in validation["failure_codes"]
    assert validation["dataset_manifest_completeness"]["missing"] == [
        ".run/s9/data/interim/csmar/daily_market/_manifest/manifest.json"
    ]
    assert summary["status"] == validation["status"] == "FAIL"
    summary_md = (output_root / "summary.md").read_text(encoding="utf-8")
    assert "- status: `FAIL`" in summary_md
    assert "- validation_status: `FAIL`" in summary_md
    assert "`PASS`" not in summary_md


def test_no_comparison_engine_is_introduced_in_src():
    hits = [
        str(path)
        for path in Path("src").rglob("*comparison*.py")
        if "private_evidence" not in str(path)
    ]

    assert hits == []


def _write_artifact_manifest(
    artifact_root: Path,
    dataset_id: str,
    logical_layer: str,
    *,
    marker: str | None = None,
) -> None:
    payload = {
        "schema_version": "0.1",
        "run_id": "artifact-writer",
        "logical_layer": logical_layer,
        "dataset_id": dataset_id,
        "primary_key": ["market_id", "asset_id", "timestamp"],
        "partition_columns": ["year"],
        "columns": ["market_id", "asset_id", "timestamp", "value"],
        "row_count": 0,
        "partition_count": 0,
        "date_min": None,
        "date_max": None,
        "null_count_summary": {
            "market_id": 0,
            "asset_id": 0,
            "timestamp": 0,
        },
        "non_finite_count_summary": {},
        "duplicate_primary_key_count": 0,
        "validation_status": "pass",
    }
    if marker is not None:
        payload["marker"] = marker
    write_internal_artifact_manifest(
        artifact_root / "_manifest" / "manifest.json",
        payload,
    )
