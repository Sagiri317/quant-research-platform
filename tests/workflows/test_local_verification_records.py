from __future__ import annotations

import json

from quant_research_platform.runtime import resources as resource_module
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.workflows import local_verification as module
from quant_research_platform.workflows.local_verification import run_local_verification
from tests.helpers.runtime_resources import capture_test_policy
from tests.workflows.local_verification_fixtures import (
    GIB,
    fake_portfolio_result,
    fake_prediction_result,
    make_local_verification_config,
    persist_fake_research_root,
    profile_with_materializers,
    repo_run_root,
    smoke_prediction_specs,
    stub_interim_importer,
    stub_output_writers,
    stub_processed_builder,
)


def test_local_verification_workflow_uses_one_runtime_policy_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    snapshots: list[RuntimeResourcePolicy] = []
    capture_calls = 0
    explicit_cap = 5 * GIB
    policy = capture_test_policy(
        available_memory_bytes=10 * GIB,
        explicit_cap_bytes=explicit_cap,
    )

    def fake_capture(*, explicit_cap_bytes=None) -> RuntimeResourcePolicy:  # noqa: ANN001
        nonlocal capture_calls
        assert explicit_cap_bytes == explicit_cap
        capture_calls += 1
        return policy

    def capture_policy() -> None:
        snapshots.append(resource_module.current_runtime_resource_policy())

    def assemble_dataset(**kwargs):
        capture_policy()
        return persist_fake_research_root(**kwargs)

    monkeypatch.setattr(
        resource_module,
        "_capture_runtime_resource_policy",
        fake_capture,
    )
    monkeypatch.setattr(module, "assemble_research_dataset", assemble_dataset)
    real_publish = module.publish_no_replace

    def publish(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        capture_policy()
        return real_publish(*args, **kwargs)

    monkeypatch.setattr(module, "publish_no_replace", publish)
    evaluation_writer = stub_output_writers(monkeypatch)
    monkeypatch.setattr(
        module,
        "_write_evaluation_outputs_from_research_dataset",
        lambda **kwargs: capture_policy() or evaluation_writer(**kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_prediction_result_from_research_dataset",
        lambda **kwargs: capture_policy() or fake_prediction_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_portfolio_result_from_research_dataset",
        lambda **kwargs: capture_policy() or fake_portfolio_result(kwargs),
    )

    interim = stub_interim_importer(on_call=capture_policy)
    processed = stub_processed_builder(on_call=capture_policy)

    def policy_aware_interim(*args, resource_policy, **kwargs):  # noqa: ANN002, ANN003, ANN202
        assert resource_policy is policy
        return interim(*args, resource_policy=resource_policy, **kwargs)

    def policy_aware_processed(*args, resource_policy, **kwargs):  # noqa: ANN002, ANN003, ANN202
        assert resource_policy is policy
        return processed(*args, resource_policy=resource_policy, **kwargs)

    profile = profile_with_materializers(
        policy_aware_interim,
        policy_aware_processed,
    )
    expected_prediction_count = len(tuple(smoke_prediction_specs()))
    run_root = repo_run_root(
        tmp_path,
        monkeypatch,
        ".run",
        "local",
        "s-policy",
    )
    result = run_local_verification(
        make_local_verification_config(
            run_root=run_root,
            start_date="2014-01-01",
                end_date="2016-12-31",
                profile=profile,
                explicit_memory_cap_bytes=explicit_cap,
            )
    )
    run_config = json.loads((run_root / "run.json").read_text(encoding="utf-8"))[
        "config"
    ]

    assert result.status == "NOT_RUN"
    assert capture_calls == 1
    assert snapshots == [policy] * (7 + expected_prediction_count)
    policy_payload = run_config["runtime_resource_policy"]
    assert set(policy_payload) == {
        "captured_available_memory_bytes",
        "memory_budget_bytes",
        "explicit_memory_cap_bytes",
        "used_memory_bytes",
        "planning_headroom_bytes",
    }
    assert policy_payload["memory_budget_bytes"] == explicit_cap
    assert policy_payload["explicit_memory_cap_bytes"] == explicit_cap
    assert (
        policy_payload["used_memory_bytes"]
        + policy_payload["planning_headroom_bytes"]
        == policy_payload["memory_budget_bytes"]
    )


def test_local_verification_failure_cleanup_keeps_the_root_policy_active(
    tmp_path,
    monkeypatch,
) -> None:
    policy = capture_test_policy(available_memory_bytes=10 * GIB)
    observed: list[RuntimeResourcePolicy] = []
    real_remove = module.remove_owned_scratch

    monkeypatch.setattr(
        resource_module,
        "_capture_runtime_resource_policy",
        lambda **_kwargs: policy,
    )
    monkeypatch.setattr(
        module,
        "_run_local_verification_with_policy",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    def remove(path):  # noqa: ANN001, ANN202
        observed.append(resource_module.current_runtime_resource_policy())
        return real_remove(path)

    monkeypatch.setattr(module, "remove_owned_scratch", remove)
    run_root = repo_run_root(tmp_path, monkeypatch, ".run", "local", "s-cleanup")

    import pytest

    with pytest.raises(RuntimeError, match="boom"):
        run_local_verification(
            make_local_verification_config(
                run_root=run_root,
                start_date="2014-01-01",
                end_date="2014-12-31",
                profile=profile_with_materializers(
                    stub_interim_importer(),
                    stub_processed_builder(),
                ),
            )
        )

    assert observed == [policy]
