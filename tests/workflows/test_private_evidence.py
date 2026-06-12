from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.storage.datasets import inspect_dataset, write_dataset
from quant_research_platform.storage.layout import PrivateEvidenceLayout
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


def test_private_evidence_output_root_rejects_traversal_outside_run(tmp_path):
    with pytest.raises(ValueError, match="output_root_must_be_under_run"):
        module._assert_output_root_under_run(tmp_path / ".run" / ".." / "outside")


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
        _persist_fake_artifact(artifact_root, "daily_market", "interim")
        return type(
            "Result",
            (),
            {"artifact_roots": {"daily_market": artifact_root}},
        )()

    def build_processed(**kwargs):
        calls.append("processed")
        artifact_root = (
            kwargs["layout"].project_root
            / "data"
            / "processed"
            / "a_share"
            / "factor"
            / "price_volume"
        )
        _persist_fake_artifact(artifact_root, "price_volume", "processed")
        inspection = inspect_dataset(artifact_root)
        return {"price_volume": inspection}

    def assemble_dataset(**kwargs):
        calls.append("research")
        artifact_root = (
            kwargs["layout"].project_root
            / "data"
            / "research"
            / "a_share"
            / "a_share_price_volume_forward_returns"
        )
        _persist_fake_artifact(
            artifact_root,
            "a_share_price_volume_forward_returns",
            "research",
        )
        return artifact_root

    monkeypatch.setattr(
        module,
        "_local_a_share_csmar_materialize_interim_data",
        import_interim,
    )
    monkeypatch.setattr(
        module,
        "_local_a_share_materialize_processed_data",
        build_processed,
    )
    monkeypatch.setattr(module, "assemble_research_dataset", assemble_dataset)
    monkeypatch.setattr(
        module,
        "run_evaluation_from_research_dataset",
        lambda **kwargs: (
            calls.append("evaluation")
            or evidence_years.append((kwargs["time_window"].start.year,))
            or evaluation_run_ids.append(kwargs["run_id"])
            or Path("evaluation")
        ),
    )
    monkeypatch.setattr(
        module,
        "run_prediction_from_research_dataset",
        lambda **kwargs: (
            calls.append("prediction")
            or evidence_years.append((kwargs["time_window"].start.year,))
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
    layout = PrivateEvidenceLayout.from_output_root(output_root)
    assert (layout.evidence_root / "evidence_summary.json").is_file()
    for layer in ("interim", "processed", "research", "evidence"):
        layer_root = getattr(layout, f"{layer}_root")
        layer_manifest = layer_root / "_manifests" / "layer_manifest.json"
        assert layer_manifest.is_file()
        assert list(layer_root.rglob("layer_manifest.json")) == [layer_manifest]
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
    captured_roots: list[Path] = []
    captured_manifests: list[dict[str, object]] = []

    def import_interim(**kwargs):
        artifact_root = (
            kwargs["layout"].project_root
            / "data"
            / "interim"
            / "csmar"
            / "daily_market"
        )
        _persist_fake_artifact(artifact_root, "daily_market", "interim")
        captured_roots.append(artifact_root)
        captured_manifests.append(inspect_dataset(artifact_root).manifest)
        return type(
            "Result",
            (),
            {"artifact_roots": {"daily_market": artifact_root}},
        )()

    monkeypatch.setattr(
        module,
        "_local_a_share_csmar_materialize_interim_data",
        import_interim,
    )
    monkeypatch.setattr(
        module,
        "_local_a_share_materialize_processed_data",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        module,
        "assemble_research_dataset",
        _persist_fake_research_root,
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
    manifest = inspect_dataset(captured_roots[0]).manifest

    assert result.status == "PASS"
    assert manifest == captured_manifests[0]


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
        return type(
            "Result",
            (),
            {"artifact_roots": {"daily_market": artifact_root}},
        )()

    monkeypatch.setattr(
        module,
        "_local_a_share_csmar_materialize_interim_data",
        import_interim,
    )
    monkeypatch.setattr(
        module,
        "_local_a_share_materialize_processed_data",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        module,
        "assemble_research_dataset",
        _persist_fake_research_root,
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
        ".run/s9/data/interim/csmar/daily_market"
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


def _persist_fake_artifact(
    artifact_root: Path,
    dataset_id: str,
    logical_layer: str,
) -> None:
    frame = _interim_frame() if logical_layer == "interim" else _dataset_frame()
    write_dataset(
        artifact_root,
        frame,
        spec=_dataset_spec(dataset_id, logical_layer),
    )


def _persist_fake_research_root(**kwargs) -> Path:
    artifact_root = (
        kwargs["layout"].project_root
        / "data"
        / "research"
        / "a_share"
        / "a_share_price_volume_forward_returns"
    )
    _persist_fake_artifact(
        artifact_root,
        "a_share_price_volume_forward_returns",
        "research",
    )
    return artifact_root


def _dataset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market_id": ["a_share"],
            "asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "value": [1.0],
        }
    )


def _interim_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_asset_id": ["asset_a"],
            "timestamp": pd.to_datetime(["2014-01-02"]),
            "value": [1.0],
        }
    )


def _dataset_spec(
    dataset_id: str,
    logical_layer: str,
) -> InterimDataSpec | ProcessedDataSpec | ResearchDatasetSpec:
    source_base = {
        "frequency": DataFrequency.DAILY,
        "grain": DataGrain.ASSET_TIMESTAMP,
        "column_roles": {
            "source_asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.FACTOR,
        },
        "primary_key": ("source_asset_id", "timestamp"),
        "partition_columns": ("year",),
    }
    market_base = {
        "frequency": DataFrequency.DAILY,
        "grain": DataGrain.ASSET_TIMESTAMP,
        "column_roles": {
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.FACTOR,
        },
        "primary_key": ("market_id", "asset_id", "timestamp"),
        "partition_columns": ("year",),
    }
    if logical_layer == "interim":
        return InterimDataSpec(
            **source_base,
            data_id=dataset_id,
            source_id=SourceId("csmar"),
        )
    if logical_layer == "processed":
        return ProcessedDataSpec(
            **market_base,
            data_id=dataset_id,
            market_id=MarketId("a_share"),
            kind=ProcessedDataKind.FACTOR,
        )
    if logical_layer == "research":
        return ResearchDatasetSpec(
            **market_base,
            data_id=dataset_id,
            market_id=MarketId("a_share"),
            input_data_ids=("price_volume",),
            base_input_data_id="price_volume",
            feature_columns=("value",),
            label_columns=(),
        )
    raise ValueError(f"unsupported_logical_layer: {logical_layer}")
