from __future__ import annotations

import json

import pytest

from quant_research_platform.workflows import local_verification as module
from quant_research_platform.workflows.local_verification import run_local_verification
from quant_research_platform.workflows.local_verification_layout import (
    LocalVerificationLayout,
)
from tests.workflows.local_verification_fixtures import (
    LEGACY_RSS_PEAK_FIELD,
    fake_evaluation_result,
    fake_portfolio_result,
    fake_prediction_result,
    make_local_verification_config,
    persist_fake_artifact,
    profile_with_empty_prediction_and_portfolio_builders,
    profile_with_materializers,
    repo_run_root,
    smoke_prediction_specs,
    stub_interim_importer,
    stub_output_writers,
    stub_processed_builder,
)


@pytest.mark.parametrize("summary_failure", (False, True))
def test_local_verification_publishes_completed_run_only_after_summary(
    tmp_path,
    monkeypatch,
    summary_failure,
):
    calls: list[str] = []
    publish_order: list[str] = []
    run_years: list[tuple[int, ...]] = []
    evaluation_run_ids: list[str] = []
    prediction_ids: list[str] = []
    portfolio_ids: list[str] = []
    original_write_summary = module.write_run_summary_markdown
    original_replace_json = module.replace_qrp_json

    def track_summary_publish(path, text):
        publish_order.append("summary")
        if summary_failure:
            raise OSError("summary_publish_failed")
        return original_write_summary(path, text)

    def track_run_publish(path, payload):
        if "completed_at" in payload:
            publish_order.append("completed_run")
        return original_replace_json(path, payload)

    monkeypatch.setattr(module, "write_run_summary_markdown", track_summary_publish)
    monkeypatch.setattr(module, "replace_qrp_json", track_run_publish)

    def assemble_dataset(**kwargs):
        calls.append("research")
        artifact_root = (
            kwargs["layout"].project_root
            / "data"
            / "research"
            / "a_share"
            / "price_volume_forward_returns"
        )
        persist_fake_artifact(
            artifact_root,
            "price_volume_forward_returns",
            "research",
        )
        return artifact_root

    monkeypatch.setattr(module, "assemble_research_dataset", assemble_dataset)
    stub_output_writers(monkeypatch)
    monkeypatch.setattr(
        module,
        "build_evaluation_result_from_research_dataset",
        lambda **kwargs: (
            calls.append("evaluation")
            or run_years.append((kwargs["time_window"].start.year,))
            or evaluation_run_ids.append(kwargs["run_id"])
            or fake_evaluation_result(kwargs)
        ),
    )
    monkeypatch.setattr(
        module,
        "build_prediction_result_from_research_dataset",
        lambda **kwargs: (
            calls.append("prediction")
            or run_years.append((kwargs["time_window"].start.year,))
            or prediction_ids.append(kwargs["prediction_id"])
            or fake_prediction_result(kwargs)
        ),
    )
    monkeypatch.setattr(
        module,
        "build_portfolio_result_from_research_dataset",
        lambda **kwargs: (
            calls.append("portfolio")
            or run_years.append((kwargs["time_window"].start.year,))
            or portfolio_ids.append(kwargs["portfolio_id"])
            or fake_portfolio_result(kwargs)
        ),
    )

    profile = profile_with_materializers(
        stub_interim_importer(on_call=lambda: calls.append("interim")),
        stub_processed_builder(on_call=lambda: calls.append("processed")),
    )
    expected_prediction_ids = [
        prediction_spec.prediction_id for prediction_spec in smoke_prediction_specs()
    ]
    expected_prediction_count = len(expected_prediction_ids)
    run_root = repo_run_root(tmp_path, monkeypatch, ".run", "s1")
    config = make_local_verification_config(
        run_root=run_root,
        start_date="2014-01-01",
        end_date="2016-12-31",
        profile=profile,
    )
    if summary_failure:
        with pytest.raises(OSError, match="summary_publish_failed"):
            run_local_verification(config)
        seed_payload = json.loads(
            (run_root / "run.json").read_text(encoding="utf-8")
        )
        assert "completed_at" not in seed_payload
        assert publish_order == ["summary"]
        return
    result = run_local_verification(config)

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
    assert publish_order == ["summary", "completed_run"]
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


def test_local_verification_skips_empty_prediction_and_portfolio_catalogs(
    tmp_path,
    monkeypatch,
):
    calls: list[str] = []

    def assemble_dataset(**kwargs):
        calls.append("research")
        artifact_root = (
            kwargs["layout"].project_root
            / "data"
            / "research"
            / "a_share"
            / "price_volume_forward_returns"
        )
        persist_fake_artifact(
            artifact_root,
            "price_volume_forward_returns",
            "research",
        )
        return artifact_root

    monkeypatch.setattr(module, "assemble_research_dataset", assemble_dataset)
    stub_output_writers(monkeypatch)
    monkeypatch.setattr(
        module,
        "build_evaluation_result_from_research_dataset",
        lambda **kwargs: calls.append("evaluation") or fake_evaluation_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_prediction_result_from_research_dataset",
        lambda **kwargs: calls.append("prediction") or fake_prediction_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_portfolio_result_from_research_dataset",
        lambda **kwargs: calls.append("portfolio") or fake_portfolio_result(kwargs),
    )

    profile = profile_with_empty_prediction_and_portfolio_builders(
        profile_with_materializers(
            stub_interim_importer(on_call=lambda: calls.append("interim")),
            stub_processed_builder(on_call=lambda: calls.append("processed")),
        )
    )
    run_root = repo_run_root(tmp_path, monkeypatch, ".run", "s-empty-optional")
    result = run_local_verification(
        make_local_verification_config(
            run_root=run_root,
            start_date="2014-01-01",
            end_date="2016-12-31",
            profile=profile,
        )
    )
    run_payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))

    assert result.status == "NOT_RUN"
    assert calls == ["interim", "processed", "research", "evaluation"]
    assert [output["output_definition_id"] for output in run_payload["outputs"]] == [
        "evaluation_outputs"
    ]
