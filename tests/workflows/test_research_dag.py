from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from quant_research_platform.contracts.targets import (
    ProducerBinding,
    ProducerKind,
    TargetSpec,
)
from quant_research_platform.profiles.research import (
    CRYPTO_PROFILE_ID,
    research_profile,
)
from quant_research_platform.sources.csmar.interim import InterimImportResult
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


def test_protocol_owns_formal_research_and_report_graphs() -> None:
    a_share = research_profile().protocol("pv_all")
    a_share_specs = {
        spec.target_id: spec for spec in a_share.target_specs("research run")
    }
    assert a_share.terminal_targets("research run") == ("pv_all_research",)
    assert a_share_specs["pv_all_evaluation"].dependencies == ("pv_all",)
    assert a_share_specs["pv_all_model_selection_refit"].dependencies == (
        "pv_all_evaluation",
    )
    assert a_share_specs["pv_all_persisted_predictions"].dependencies == (
        "pv_all_model_selection_refit",
    )
    assert a_share_specs["pv_all_statistics"].dependencies == (
        "pv_all_persisted_predictions",
    )
    ledger_ids = tuple(
        target_id for target_id in a_share_specs if target_id.endswith("_ledger")
    )
    assert len(ledger_ids) == 5
    assert set(a_share_specs["pv_all_research"].dependencies) == {
        "pv_all_statistics",
        *ledger_ids,
    }
    assert a_share_specs["pv_all_report"].dependencies == ("pv_all_report_artifact",)

    crypto_profile = research_profile(CRYPTO_PROFILE_ID)
    crypto = crypto_profile.protocol(crypto_profile.research_datasets.list()[0].data_id)
    crypto_specs = {
        spec.target_id: spec for spec in crypto.target_specs("research run")
    }
    assert {
        "binance_futures_hft_research_100ms_maker_execution_diagnostics",
        "binance_futures_hft_research_100ms_taker_execution_diagnostics",
    }.issubset(crypto_specs)
    assert crypto_specs["binance_futures_hft_research_100ms_artifact"].dependencies == (
        "binance_futures_hft_research_100ms_maker_execution_diagnostics",
        "binance_futures_hft_research_100ms_taker_execution_diagnostics",
    )


def test_declared_processed_dependencies_are_not_removed_from_the_plan(
    tmp_path: Path,
) -> None:
    profile = research_profile()
    panel_spec = profile.processed.get("market_panel_1d")
    panel_root = StorageLayout(project_root=tmp_path).processed_data_root(
        panel_spec.market_id,
        panel_spec.kind,
        panel_spec.data_id,
    )
    panel_root.mkdir(parents=True)
    plan = research.plan_targets(
        profile=profile,
        action="dataset build",
        terminal_targets=("pv_all",),
        layout=StorageLayout(project_root=tmp_path),
    )
    panel = next(
        target for target in plan["targets"] if target["target_id"] == "market_panel_1d"
    )
    assert panel["decision"] == "BLOCKED"
    assert panel["blocker"] == "semantic_identity_not_available"
    assert {
        "industry_classification",
        "industry_classification_sets",
        "industry_classification_versions",
        "listing_status_changes",
        "stock_info_changes",
        "stock_info_annual",
        "trading_no_limit",
        "special_treatment_changes",
    }.issubset(panel["upstream_target_ids"])


def test_interim_binding_receives_one_requested_target(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []
    data_id = "market_bars_1d"

    def producer(**kwargs: object) -> InterimImportResult:
        requested = kwargs["data_ids"]
        assert type(requested) is tuple
        calls.append(requested)
        return InterimImportResult(
            artifact_roots={data_id: tmp_path / data_id},
            skipped=(),
            pruning_report_text="test",
        )

    profile = replace(
        research_profile(),
        producer_bindings={
            data_id: ProducerBinding(ProducerKind.INTERIM, producer),
        },
    )
    result = research.data_materialize(
        profile=profile,
        layout=StorageLayout(project_root=tmp_path),
        raw_root=tmp_path / "raw",
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
    )

    assert calls == [(data_id,)]
    assert result["status"] == "BLOCKED"


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

    assert result["status"] == "BLOCKED"
    assert result["dataset_status"] == "BLOCKED"
    assert result["research_status"] == "BLOCKED"
    assert result["report_status"] == "BLOCKED"

    inspection = research.inspect(
        profile=research_profile(),
        layout=StorageLayout(project_root=tmp_path),
        dataset_id="pv_all",
    )
    assert inspection["physical_manifest_valid"] is True
    assert inspection["status"] == "BLOCKED"
    assert inspection["blocker"] == "semantic_identity_not_available"
