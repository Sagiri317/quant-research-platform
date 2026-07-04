from __future__ import annotations

import json

from quant_research_platform.storage.datasets import inspect_dataset
from quant_research_platform.workflows import local_verification as module
from quant_research_platform.workflows.local_verification import run_local_verification
from quant_research_platform.workflows.local_verification_layout import (
    LocalVerificationLayout,
)
from tests.helpers import frequency_ids
from tests.workflows.local_verification_fixtures import (
    LEGACY_RSS_PEAK_FIELD,
    _fake_evaluation_result,
    _fake_portfolio_result,
    _fake_prediction_result,
    _persist_fake_artifact,
    _profile_with_materializers,
    _repo_run_root,
    _smoke_prediction_specs,
    _stub_output_writers,
    _verification_config,
)


def test_local_verification_workflow_runs_injected_generation_and_writes_artifacts(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []
    run_years: list[tuple[int, ...]] = []
    evaluation_run_ids: list[str] = []
    prediction_ids: list[str] = []
    portfolio_ids: list[str] = []

    def import_interim(
        layout,
        raw_root,
        start_date,
        end_date,
    ):
        del raw_root, start_date, end_date
        calls.append("interim")
        artifact_root = (
            layout.project_root
            / "data"
            / "interim"
            / "csmar"
            / frequency_ids.MARKET_BARS_DATA_ID
        )
        _persist_fake_artifact(
            artifact_root,
            frequency_ids.MARKET_BARS_DATA_ID,
            "interim",
        )
        return type(
            "Result",
            (),
            {"artifact_roots": {frequency_ids.MARKET_BARS_DATA_ID: artifact_root}},
        )()

    def build_processed(
        layout,
        source_id,
        time_window,
        kinds,
    ):
        del source_id, time_window, kinds
        calls.append("processed")
        artifact_root = (
            layout.project_root
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
            / "price_volume_forward_returns"
        )
        _persist_fake_artifact(
            artifact_root,
            "price_volume_forward_returns",
            "research",
        )
        return artifact_root

    monkeypatch.setattr(module, "assemble_research_dataset", assemble_dataset)
    _stub_output_writers(monkeypatch)
    monkeypatch.setattr(
        module,
        "build_evaluation_result_from_research_dataset",
        lambda **kwargs: (
            calls.append("evaluation")
            or run_years.append((kwargs["time_window"].start.year,))
            or evaluation_run_ids.append(kwargs["run_id"])
            or _fake_evaluation_result(kwargs)
        ),
    )
    monkeypatch.setattr(
        module,
        "build_prediction_result_from_research_dataset",
        lambda **kwargs: (
            calls.append("prediction")
            or run_years.append((kwargs["time_window"].start.year,))
            or prediction_ids.append(kwargs["prediction_id"])
            or _fake_prediction_result(kwargs)
        ),
    )
    monkeypatch.setattr(
        module,
        "build_portfolio_result_from_research_dataset",
        lambda **kwargs: (
            calls.append("portfolio")
            or run_years.append((kwargs["time_window"].start.year,))
            or portfolio_ids.append(kwargs["portfolio_id"])
            or _fake_portfolio_result(kwargs)
        ),
    )

    profile = _profile_with_materializers(import_interim, build_processed)
    expected_prediction_ids = [
        prediction_spec.prediction_id for prediction_spec in _smoke_prediction_specs()
    ]
    expected_prediction_count = len(expected_prediction_ids)
    run_root = _repo_run_root(tmp_path, monkeypatch, ".run", "s1")
    result = run_local_verification(
        _verification_config(
            run_root=run_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
            profile=profile,
        )
    )

    assert calls == [
        "interim",
        "processed",
        "research",
        "evaluation",
        *(["prediction"] * expected_prediction_count),
        "portfolio",
    ]
    assert run_years == [(2014,)] * (expected_prediction_count + 2)
    assert evaluation_run_ids == ["signal_forward_return_20d"]
    assert prediction_ids == expected_prediction_ids
    assert portfolio_ids == ["signal_forward_return_20d_portfolio"]
    assert result.status == "NOT_RUN"
    assert result.run_id == "s1"
    assert result.run_root == "s1"
    layout = LocalVerificationLayout.from_run_root(run_root)
    run_payload = json.loads(layout.run_json_path.read_text(encoding="utf-8"))
    assert layout.summary_md_path.is_file()
    assert run_payload["schema_id"] == "qrp-run"
    assert run_payload["status"] == {
        "overall": "NOT_RUN",
        "validation": "PASS",
        "smoke": "PASS",
        "compare": "NOT_RUN",
        "share_safety": "PASS",
        "full_horizon": "NOT_RUN",
        "validation_matrix": "NOT_RUN",
    }
    assert run_payload["run_id"] == "s1"
    assert run_payload["root_ownership"]["replace_safe"] is True
    dataset_manifest_paths = sorted(
        (run_root / "data").rglob("_manifest/manifest.json")
    )
    expected_layers = {
        path.relative_to(run_root).parts[1] for path in dataset_manifest_paths
    }
    actual_layers = {
        layer["logical_layer"] for layer in run_payload["data"]["layers"]
    }
    represented_manifests = {
        dataset["manifest_path"]
        for layer in run_payload["data"]["layers"]
        for dataset in layer["datasets"]
    }
    expected_manifests = {
        path.relative_to(run_root).as_posix() for path in dataset_manifest_paths
    }
    assert actual_layers == expected_layers
    assert "evidence" not in actual_layers
    assert represented_manifests == expected_manifests
    assert len(run_payload["outputs"]) == expected_prediction_count + 2
    assert all(
        output["schema_id"] == "qrp-output-manifest"
        for output in run_payload["outputs"]
    )
    timing_steps = [
        record["step_name"] for record in run_payload["execution_metrics"]["records"]
    ]
    validation_record = next(
        record
        for record in run_payload["execution_metrics"]["records"]
        if record["step_name"] == "validation_summary"
    )
    assert "validation_summary" in timing_steps
    assert "rss_start_end_max_mb" in validation_record
    assert LEGACY_RSS_PEAK_FIELD not in validation_record
    assert {path.name for path in run_root.iterdir()} == {
        "data",
        "logs",
        "outputs",
        "run.json",
        "summary.md",
    }
