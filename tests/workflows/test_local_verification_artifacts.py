from __future__ import annotations

import json

import pytest

from quant_research_platform.workflows import local_verification as module
from quant_research_platform.workflows.local_verification import run_local_verification
from quant_research_platform.workflows.local_verification_layout import (
    LocalVerificationLayout,
)
from tests.helpers.runtime_resources import capture_test_policy
from tests.workflows.local_verification_fixtures import (
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
    final_reserved_bytes: list[int] = []
    original_write_summary = module.write_markdown_evidence
    original_replace_json = module.replace_qrp_json
    original_finish_memory = module.RuntimeResourcePolicy.finish_memory_measurement

    def record_final_memory(self, **kwargs):  # noqa: ANN001, ANN003
        final_reserved_bytes.append(self.reserved_memory_bytes)
        return original_finish_memory(self, **kwargs)

    def track_summary_publish(path, text):
        publish_order.append("summary")
        if summary_failure:
            raise OSError("summary_publish_failed")
        return original_write_summary(path, text)

    def track_run_publish(path, payload):
        if "completed_at" in payload:
            publish_order.append("completed_run")
        return original_replace_json(path, payload)

    monkeypatch.setattr(module, "write_markdown_evidence", track_summary_publish)
    monkeypatch.setattr(module, "replace_qrp_json", track_run_publish)
    monkeypatch.setattr(
        module.RuntimeResourcePolicy,
        "finish_memory_measurement",
        record_final_memory,
    )

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
    representative_prediction_id = next(
        prediction_id
        for prediction_id in expected_prediction_ids
        if prediction_id.startswith("linear_regression_")
    )
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
        assert not run_root.exists()
        assert publish_order == ["summary"]
        assert final_reserved_bytes == []
        return
    result = run_local_verification(config)

    assert calls == [
        "interim",
        "processed",
        "research",
        "evaluation",
        *(["prediction"] * expected_prediction_count),
        "portfolio",
        "portfolio",
    ]
    assert run_years == [(2014,)] * (expected_prediction_count + 3)
    assert evaluation_run_ids == ["signal_forward_return_20d"]
    assert prediction_ids == expected_prediction_ids
    assert portfolio_ids == [
        f"{representative_prediction_id}_portfolio",
        "signal_forward_return_20d_portfolio",
    ]
    assert result.status == "NOT_RUN"
    assert publish_order == ["summary", "completed_run"]
    assert len(final_reserved_bytes) == 1 and final_reserved_bytes[0] > 0
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
    assert len(run_payload["outputs"]) == expected_prediction_count + 3
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
    assert {path.name for path in run_root.iterdir()} == {
        "data",
        "logs",
        "outputs",
        "run.json",
        "summary.md",
    }


def test_local_verification_replace_failure_preserves_old_good(
    tmp_path,
    monkeypatch,
) -> None:
    run_root = repo_run_root(tmp_path, monkeypatch, ".run", "old_good")
    module._prepare_run_root(
        run_root,
        run_id="old_good",
        repo_root=run_root.parents[1],
    )
    marker = run_root / "old-good.txt"
    marker.write_text("old-good\n", encoding="utf-8")
    old_run_json = (run_root / "run.json").read_bytes()

    def fail_replacement(**kwargs):
        candidate = kwargs["run_root"]
        (candidate / "partial.txt").write_text("partial\n", encoding="utf-8")
        raise RuntimeError("replacement_failed")

    monkeypatch.setattr(
        module,
        "_run_local_verification_with_policy",
        fail_replacement,
    )
    config = make_local_verification_config(
        run_root=run_root,
        run_id="old_good",
        start_date="2014-01-01",
        end_date="2014-12-31",
        replace_output=True,
        profile=profile_with_empty_prediction_and_portfolio_builders(),
    )

    with pytest.raises(RuntimeError, match="replacement_failed"):
        run_local_verification(config)

    assert (run_root / "run.json").read_bytes() == old_run_json
    assert marker.read_text(encoding="utf-8") == "old-good\n"
    assert not (run_root / "partial.txt").exists()
    assert tuple(path.name for path in run_root.parent.iterdir()) == ("old_good",)

    replacement_count = 0

    def complete_replacement(**kwargs):
        nonlocal replacement_count
        replacement_count += 1
        candidate = kwargs["run_root"]
        (candidate / "new-good.txt").write_text(
            f"new-good-{replacement_count}\n",
            encoding="utf-8",
        )
        return module.LocalVerificationRunResult(
            status="PASS",
            run_id="old_good",
            run_root=candidate.name,
            summary={},
        )

    monkeypatch.setattr(
        module,
        "_run_local_verification_with_policy",
        complete_replacement,
    )
    original_move = module.move_path_no_replace

    def fail_candidate_publish(source, target):
        if (
            source.name.startswith(".lv-")
            and not source.name.startswith(".lv-old-")
            and target == run_root
        ):
            raise OSError("replacement_publish_failed")
        return original_move(source, target)

    monkeypatch.setattr(module, "move_path_no_replace", fail_candidate_publish)
    with pytest.raises(OSError, match="replacement_publish_failed"):
        run_local_verification(config)

    assert (run_root / "run.json").read_bytes() == old_run_json
    assert marker.read_text(encoding="utf-8") == "old-good\n"
    assert tuple(path.name for path in run_root.parent.iterdir()) == ("old_good",)

    monkeypatch.setattr(module, "move_path_no_replace", original_move)
    policies = iter(
        (
            capture_test_policy(available_memory_bytes=16 * 1024 * 1024),
            capture_test_policy(available_memory_bytes=16 * 1024 * 1024),
            capture_test_policy(available_memory_bytes=16 * 1024 * 1024),
        )
    )
    monkeypatch.setattr(
        module.RuntimeResourcePolicy,
        "capture",
        staticmethod(lambda **_kwargs: next(policies)),
    )
    result = run_local_verification(config)
    previous_root = module._previous_generation_root(run_root)

    assert result.run_root == "old_good"
    assert not marker.exists()
    assert (run_root / "new-good.txt").read_text(encoding="utf-8") == "new-good-2\n"
    assert (previous_root / "old-good.txt").read_text(encoding="utf-8") == "old-good\n"
    assert {path.name for path in run_root.parent.iterdir()} == {
        "old_good",
        previous_root.name,
    }

    run_local_verification(config)

    assert not (previous_root / "old-good.txt").exists()
    assert (previous_root / "new-good.txt").read_text(encoding="utf-8") == (
        "new-good-2\n"
    )
    assert {path.name for path in run_root.parent.iterdir()} == {
        "old_good",
        previous_root.name,
    }

    original_remove = module.remove_owned_scratch

    def fail_previous_cleanup(path, identity):  # noqa: ANN001, ANN202
        if path == previous_root:
            raise OSError("previous_generation_cleanup_failed")
        return original_remove(path, identity)

    monkeypatch.setattr(module, "remove_owned_scratch", fail_previous_cleanup)
    with pytest.raises(OSError, match="previous_generation_cleanup_failed"):
        run_local_verification(config)

    assert (run_root / "new-good.txt").read_text(encoding="utf-8") == "new-good-3\n"
    assert (previous_root / "new-good.txt").read_text(encoding="utf-8") == (
        "new-good-2\n"
    )


def test_final_rss_rejection_rolls_back_published_replacement(
    tmp_path,
    monkeypatch,
) -> None:
    run_root = repo_run_root(tmp_path, monkeypatch, ".run", "rss_old_good")
    module._prepare_run_root(
        run_root,
        run_id="rss_old_good",
        repo_root=run_root.parents[1],
    )
    old_run_json = (run_root / "run.json").read_bytes()
    marker = run_root / "old-good.txt"
    marker.write_text("old-good\n", encoding="utf-8")
    policy = capture_test_policy(
        available_memory_bytes=1024,
        rss_bytes=100,
        explicit_cap_bytes=600,
    )
    measurement = policy._measurement  # noqa: SLF001
    original_payload = measurement._payload  # type: ignore[attr-defined]  # noqa: SLF001

    def failed_evidence() -> dict[str, int | float | str]:
        payload = original_payload()
        payload["sampled_peak_rss_bytes"] = 701
        payload["peak_incremental_rss_bytes"] = 601
        payload["status"] = "FAIL"
        return payload

    measurement.checkpoint_evidence = failed_evidence  # type: ignore[method-assign]
    monkeypatch.setattr(
        module.RuntimeResourcePolicy,
        "capture",
        staticmethod(lambda **_kwargs: policy),
    )

    def replacement(**kwargs):
        candidate = kwargs["run_root"]
        (candidate / "new-good.txt").write_text("new-good\n", encoding="utf-8")
        return module.LocalVerificationRunResult(
            status="PASS",
            run_id="rss_old_good",
            run_root=candidate.name,
            summary={},
        )

    monkeypatch.setattr(module, "_run_local_verification_with_policy", replacement)
    config = make_local_verification_config(
        run_root=run_root,
        run_id="rss_old_good",
        start_date="2014-01-01",
        end_date="2014-12-31",
        replace_output=True,
        profile=profile_with_empty_prediction_and_portfolio_builders(),
    )

    with pytest.raises(MemoryError, match="runtime_rss_budget_exceeded"):
        run_local_verification(config)

    assert (run_root / "run.json").read_bytes() == old_run_json
    assert marker.read_text(encoding="utf-8") == "old-good\n"
    assert tuple(path.name for path in run_root.parent.iterdir()) == (
        "rss_old_good",
    )

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
