from __future__ import annotations

import json

from quant_research_platform.runtime import resources as resource_module
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.workflows import local_verification as module
from quant_research_platform.workflows.local_verification import run_local_verification
from tests.helpers.runtime_resources import capture_test_policy
from tests.workflows.local_verification_fixtures import (
    GIB,
    fake_evaluation_result,
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
        snapshots.append(resource_module.active_runtime_resource_policy())

    def assemble_dataset(**kwargs):
        capture_policy()
        return persist_fake_research_root(**kwargs)

    monkeypatch.setattr(
        resource_module.RuntimeResourcePolicy,
        "capture",
        staticmethod(fake_capture),
    )
    monkeypatch.setattr(module, "assemble_research_dataset", assemble_dataset)
    stub_output_writers(monkeypatch)
    monkeypatch.setattr(
        module,
        "build_evaluation_result_from_research_dataset",
        lambda **kwargs: capture_policy() or fake_evaluation_result(kwargs),
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

    profile = profile_with_materializers(
        stub_interim_importer(on_call=capture_policy),
        stub_processed_builder(on_call=capture_policy),
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
    assert snapshots == [policy] * (6 + expected_prediction_count)
    assert run_config["runtime_resource_policy"] == policy.to_payload()
