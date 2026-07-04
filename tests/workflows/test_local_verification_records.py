from __future__ import annotations

import json
from pathlib import Path

from quant_research_platform.runtime import resources as resource_module
from quant_research_platform.runtime.resources import RuntimeResourcePolicy
from quant_research_platform.storage.datasets import inspect_dataset
from quant_research_platform.workflows import local_verification as module
from quant_research_platform.workflows.local_verification import run_local_verification
from tests.helpers import frequency_ids
from tests.workflows.local_verification_fixtures import (
    GIB,
    MIB,
    _fake_evaluation_result,
    _fake_portfolio_result,
    _fake_prediction_result,
    _persist_fake_artifact,
    _persist_fake_research_root,
    _profile_with_materializers,
    _repo_run_root,
    _smoke_prediction_specs,
    _stub_output_writers,
    _verification_config,
)


def test_local_verification_workflow_uses_one_runtime_policy_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    snapshots: list[RuntimeResourcePolicy] = []
    from_environment_calls = 0
    policy = RuntimeResourcePolicy(
        available_memory_bytes=10 * GIB,
        memory_budget_fraction=0.5,
        chunk_target_bytes=128 * MIB,
        memory_safety_factor=0.5,
    )

    def fake_from_environment() -> RuntimeResourcePolicy:
        nonlocal from_environment_calls
        from_environment_calls += 1
        return policy

    def capture_policy() -> None:
        snapshots.append(resource_module.active_runtime_resource_policy())

    def import_interim(layout, raw_root, start_date, end_date):
        del raw_root, start_date, end_date
        capture_policy()
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

    def build_processed(layout, source_id, time_window, kinds):
        del source_id, time_window, kinds
        capture_policy()
        artifact_root = (
            layout.project_root
            / "data"
            / "processed"
            / "a_share"
            / "factor"
            / "price_volume"
        )
        _persist_fake_artifact(artifact_root, "price_volume", "processed")
        return {"price_volume": inspect_dataset(artifact_root)}

    def assemble_dataset(**kwargs):
        capture_policy()
        return _persist_fake_research_root(**kwargs)

    monkeypatch.setattr(
        resource_module.RuntimeResourcePolicy,
        "from_environment",
        staticmethod(fake_from_environment),
    )
    monkeypatch.setattr(module, "assemble_research_dataset", assemble_dataset)
    _stub_output_writers(monkeypatch)
    monkeypatch.setattr(
        module,
        "build_evaluation_result_from_research_dataset",
        lambda **kwargs: capture_policy() or _fake_evaluation_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_prediction_result_from_research_dataset",
        lambda **kwargs: capture_policy() or _fake_prediction_result(kwargs),
    )
    monkeypatch.setattr(
        module,
        "build_portfolio_result_from_research_dataset",
        lambda **kwargs: capture_policy() or _fake_portfolio_result(kwargs),
    )

    profile = _profile_with_materializers(import_interim, build_processed)
    expected_prediction_count = len(tuple(_smoke_prediction_specs()))
    run_root = _repo_run_root(
        tmp_path,
        monkeypatch,
        ".run",
        "local",
        "s-policy",
    )
    result = run_local_verification(
        _verification_config(
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
