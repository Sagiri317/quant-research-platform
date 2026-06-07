from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_research_platform.workflows import private_evidence as module
from quant_research_platform.workflows.private_evidence import (
    PrivateEvidenceRunConfig,
    run_private_evidence,
)


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


def test_private_evidence_workflow_runs_injected_generation_and_writes_artifacts(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []
    evidence_years: list[tuple[int, ...]] = []

    def import_interim(**kwargs):
        calls.append("interim")
        data_root = kwargs["layout"].project_root / "data" / "interim" / "csmar"
        (data_root / "daily_market" / "data").mkdir(parents=True)
        return object()

    def build_processed(**kwargs):
        calls.append("processed")
        processed = Path(kwargs["processed_root"])
        (processed / "factors" / "price_volume" / "data").mkdir(parents=True)
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
        data_root = (
            processed
            / "research_datasets"
            / "a_share_price_volume_forward_returns"
            / "data"
        )
        data_root.mkdir(parents=True)
        return data_root

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
            or Path("evaluation")
        ),
    )
    monkeypatch.setattr(
        module,
        "run_prediction_from_research_dataset",
        lambda **kwargs: (
            calls.append("prediction")
            or evidence_years.append(kwargs["years"])
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
    assert result.status == "PASS"
    assert (output_root / "config" / "run_config.json").is_file()
    assert (output_root / "manifests" / "run_manifest.json").is_file()
    assert (output_root / "timings" / "timings.json").is_file()
    assert (output_root / "validations" / "validation_summary.json").is_file()
    assert (output_root / "data" / "evidence" / "evidence_summary.json").is_file()
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"


def test_no_comparison_engine_is_introduced_in_src():
    hits = [
        str(path)
        for path in Path("src").rglob("*comparison*.py")
        if "private_evidence" not in str(path)
    ]

    assert hits == []
