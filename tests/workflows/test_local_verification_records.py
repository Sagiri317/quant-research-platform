from __future__ import annotations

import json
from pathlib import Path

from quant_research_platform.runtime import resources as resource_module
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.workflows import local_verification as module
from quant_research_platform.workflows.local_verification import run_local_verification
from tests.workflows.local_verification_fixtures import (
    GIB,
    MIB,
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
    from_environment_calls = 0
    policy = RuntimeResourcePolicy.from_environment(
        available_memory_bytes=10 * GIB,
        environ={
            "QRP_TASK_MEMORY_BUDGET_MB": str(5 * GIB // MIB),
            "QRP_PAYLOAD_TARGET_MB": "128",
        },
    )

    def fake_from_environment() -> RuntimeResourcePolicy:
        nonlocal from_environment_calls
        from_environment_calls += 1
        return policy

    def capture_policy() -> None:
        snapshots.append(resource_module.active_runtime_resource_policy())

    def assemble_dataset(**kwargs):
        capture_policy()
        return persist_fake_research_root(**kwargs)

    monkeypatch.setattr(
        resource_module.RuntimeResourcePolicy,
        "from_environment",
        staticmethod(fake_from_environment),
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
        )
    )
    run_config = json.loads((run_root / "run.json").read_text(encoding="utf-8"))[
        "config"
    ]

    assert result.status == "NOT_RUN"
    assert from_environment_calls == 1
    assert snapshots == [policy] * (5 + expected_prediction_count)
    assert run_config["runtime_resource_policy"] == policy.to_payload()


def test_no_comparison_engine_is_introduced_in_src():
    allowed_owner_paths = {
        Path("src/quant_research_platform/storage/_dataset_comparison.py"),
    }
    hits = [
        str(path)
        for path in Path("src").rglob("*comparison*.py")
        if path not in allowed_owner_paths and "verification_summary" not in str(path)
    ]
    assert hits == []
