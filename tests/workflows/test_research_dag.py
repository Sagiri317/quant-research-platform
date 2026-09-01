from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.contracts.targets import TargetSpec
from quant_research_platform.profiles.research import (
    CRYPTO_PROFILE_ID,
    research_profile,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research
from quant_research_platform.workflows.smoke import run_research_synthetic_smoke


def test_target_graph_is_permutation_stable() -> None:
    first = (
        TargetSpec("root", ("b", "a"), producer_id="root"),
        TargetSpec("a", producer_id="a"),
        TargetSpec("b", ("leaf",), producer_id="b"),
        TargetSpec("leaf", producer_id="leaf"),
    )
    second = (
        TargetSpec("root", ("a", "b"), producer_id="root"),
        TargetSpec("b", ("leaf",), producer_id="b"),
        TargetSpec("leaf", producer_id="leaf"),
        TargetSpec("a", producer_id="a"),
    )

    assert tuple(
        spec.target_id for spec in research.resolve_target_graph(first, ("root",))
    ) == (
        "a",
        "leaf",
        "b",
        "root",
    )
    assert tuple(
        spec.target_id for spec in research.resolve_target_graph(second, ("root",))
    ) == (
        "a",
        "leaf",
        "b",
        "root",
    )


def test_target_graph_rejects_cycles_and_unknown_dependencies() -> None:
    with pytest.raises(ValueError, match="target_dependency_cycle"):
        research.resolve_target_graph(
            (
                TargetSpec("a", ("b",)),
                TargetSpec("b", ("a",)),
            ),
            ("a",),
        )
    with pytest.raises(ValueError, match="target_unknown"):
        research.resolve_target_graph((TargetSpec("a", ("missing",)),), ("a",))


def test_crypto_plan_is_blocked_when_generic_producers_are_unbound(
    tmp_path: Path,
) -> None:
    profile = research_profile(CRYPTO_PROFILE_ID)
    dataset_id = profile.research_datasets.list()[0].data_id

    plan = research.plan_targets(
        profile=profile,
        action="dataset build",
        terminal_targets=(dataset_id,),
        layout=StorageLayout(project_root=tmp_path),
    )

    assert plan["status"] == "BLOCKED"
    assert any(
        item["reason"] == "producer_binding_missing" for item in plan["blockers"]
    )


def test_research_and_report_terminals_are_honestly_blocked(tmp_path: Path) -> None:
    profile = research_profile()
    layout = StorageLayout(project_root=tmp_path)

    research_plan = research.research_run(
        profile=profile,
        layout=layout,
        scenario_id="pv_all",
        start_date="2019-01-01",
        end_date="2019-12-31",
        plan=True,
    )
    report_plan = research.report_build(
        profile=profile,
        layout=layout,
        scenario_id="pv_all",
        plan=True,
    )

    assert research_plan["status"] == "BLOCKED"
    assert report_plan["status"] == "BLOCKED"
    assert all(
        target["decision"] != "REUSE"
        for target in (*research_plan["targets"], *report_plan["targets"])
    )

    research_result = research.research_run(
        profile=profile,
        layout=layout,
        scenario_id="pv_all",
        start_date="2019-01-01",
        end_date="2019-12-31",
    )
    report_result = research.report_build(
        profile=profile,
        layout=layout,
        scenario_id="pv_all",
    )
    assert research_result["status"] == "BLOCKED"
    assert report_result["status"] == "BLOCKED"
    assert "report" not in research_result
    assert "report" not in report_result

    rebuild = research.dataset_build(
        profile=profile,
        layout=layout,
        dataset_id="pv_all",
        start_date="2019-01-01",
        end_date="2019-12-31",
        rebuild=True,
    )
    assert rebuild["status"] == "BLOCKED"
    assert any(
        item["blocker"] == "semantic_identity_not_available"
        for item in rebuild["target_results"]
    )


def test_inspect_and_status_never_treat_a_directory_as_a_complete_target(
    tmp_path: Path,
) -> None:
    profile = research_profile()
    layout = StorageLayout(project_root=tmp_path)
    root = layout.research_dataset_root("a_share", "pv_all")
    root.mkdir(parents=True)

    inspected = research.inspect(
        profile=profile,
        layout=layout,
        dataset_id="pv_all",
    )
    status = research.status(profile=profile, layout=layout)

    assert inspected["status"] == "BLOCKED"
    assert inspected["blocker"] == "dataset_manifest_missing"
    assert status["status"] == "BLOCKED"
    assert not (root / "spec.json").exists()
    assert not (root / "_manifest").exists()


def test_a_share_synthetic_chain_uses_the_single_resolver(tmp_path: Path) -> None:
    result = run_research_synthetic_smoke(tmp_path)

    assert result["status"] == "SYNTHETIC_SMOKE_PASS"
    assert result["dataset_status"] == "COMPLETE"
    assert result["research_status"] == "BLOCKED"
    assert result["report_status"] == "BLOCKED"
