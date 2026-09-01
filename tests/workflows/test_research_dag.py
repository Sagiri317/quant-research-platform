from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.targets import (
    ProducerBinding,
    ProducerKind,
    TargetSpec,
)
from quant_research_platform.profiles.research import (
    CRYPTO_PROFILE_ID,
    research_profile,
)
from quant_research_platform.research.protocols import a_share
from quant_research_platform.sources.csmar.interim import InterimImportResult
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import content_hash_for_files
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
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
        item["reason"] == "capability_not_implemented" for item in plan["blockers"]
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
    assert panel["blocker"] == "dependency_blocked: market_bars_1d"
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
        spec = research_profile().interim.get(data_id)
        frame = pd.DataFrame(
            {
                column: pd.Series(
                    dtype="datetime64[ns]" if role is ColumnRole.TIMESTAMP else "string"
                )
                for column, role in spec.column_roles.items()
            }
        )
        producer_layout = kwargs["layout"]
        output_root = producer_layout.target_root(
            data_id,
            producer_layout.interim_data_root(spec.source_id, data_id),
        )
        write_dataset(
            output_root,
            frame,
            spec=spec,
            instance_metadata=kwargs["instance_metadata"],
            resource_policy=kwargs["resource_policy"],
            borrowed_input=True,
        )
        return InterimImportResult(
            artifact_roots={data_id: output_root},
            skipped=(),
            pruning_report_text="test",
        )

    def source_identity(_raw_root: Path, _data_id: str) -> str:
        return "a" * 64

    profile = replace(
        research_profile(),
        producer_bindings={
            data_id: ProducerBinding(
                ProducerKind.INTERIM,
                producer,
                source_identity=source_identity,
            ),
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
    assert result["status"] == "COMPLETE"


def test_source_identity_mutation_builds_a_new_immutable_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_id = "market_bars_1d"
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    marker = raw_root / "source.raw"
    marker.write_bytes(b"before")
    calls = 0

    def producer(**kwargs: object) -> InterimImportResult:
        nonlocal calls
        calls += 1
        producer_layout = kwargs["layout"]
        spec = research_profile().interim.get(data_id)
        output_root = producer_layout.target_root(
            data_id,
            producer_layout.interim_data_root(spec.source_id, data_id),
        )
        frame = pd.DataFrame(
            {
                column: pd.Series(
                    dtype="datetime64[ns]" if role is ColumnRole.TIMESTAMP else "string"
                )
                for column, role in spec.column_roles.items()
            }
        )
        write_dataset(
            output_root,
            frame,
            spec=spec,
            instance_metadata=kwargs["instance_metadata"],
            resource_policy=kwargs["resource_policy"],
            borrowed_input=True,
        )
        return InterimImportResult(
            artifact_roots={data_id: output_root},
            skipped=(),
            pruning_report_text="test",
        )

    def source_identity(raw: Path, _data_id: str) -> str:
        return content_hash_for_files(
            (("market_bars_1d", raw / marker.name),),
            error="source_identity_invalid",
        )

    profile = replace(
        research_profile(),
        producer_bindings={
            data_id: ProducerBinding(
                ProducerKind.INTERIM,
                producer,
                source_identity=source_identity,
            ),
        },
    )
    layout = StorageLayout(project_root=tmp_path)
    first = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
    )
    marker.write_bytes(b"after")
    second = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
    )

    first_identity = first["target_results"][0]["instance_identity"]
    second_identity = second["target_results"][0]["instance_identity"]
    assert first["status"] == second["status"] == "COMPLETE"
    assert first_identity != second_identity
    assert calls == 2

    instances = tuple(
        (layout.interim_data_root("csmar", data_id) / "instances").iterdir()
    )
    assert {path.name for path in instances} == {first_identity, second_identity}

    plan = research._plan_targets

    def mutate_after_plan(**kwargs: object) -> dict[str, object]:
        result = plan(**kwargs)
        marker.write_bytes(b"between-plan-and-reuse")
        return result

    monkeypatch.setattr(research, "_plan_targets", mutate_after_plan)
    third = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
    )
    assert third["status"] == "BLOCKED"
    assert third["target_results"] == [
        {
            "target_id": data_id,
            "status": "BLOCKED",
            "blocker": "source_identity_changed",
            "instance_identity": None,
        }
    ]
    assert calls == 2


def test_data_target_identity_ignores_unrelated_protocol_factor_scope(
    tmp_path: Path,
) -> None:
    data_id = "market_bars_1d"

    def source_identity(_raw_root: Path, _data_id: str) -> str:
        return "a" * 64

    binding = ProducerBinding(
        ProducerKind.INTERIM,
        lambda **_: None,
        source_identity=source_identity,
    )
    base = research_profile()
    full_profile = replace(
        base,
        protocol=lambda *, scenario_id, access_window: a_share.protocol(
            scenario_id,
            access_window,
            factor_scope="full",
        ),
        producer_bindings={data_id: binding},
    )
    core_profile = replace(base, producer_bindings={data_id: binding})

    core_plan = research.plan_targets(
        profile=core_profile,
        action="data materialize",
        terminal_targets=(data_id,),
        layout=StorageLayout(project_root=tmp_path / "core"),
        raw_root=tmp_path / "raw",
    )
    full_plan = research.plan_targets(
        profile=full_profile,
        action="data materialize",
        terminal_targets=(data_id,),
        layout=StorageLayout(project_root=tmp_path / "full"),
        raw_root=tmp_path / "raw",
    )
    core_target = core_plan["targets"][0]
    full_target = full_plan["targets"][0]
    assert core_target["requested_identity"] == full_target["requested_identity"]


def test_csmar_dataset_action_resolves_a_valid_auxiliary_root(
    tmp_path: Path,
) -> None:
    profile = research_profile()
    layout = StorageLayout(project_root=tmp_path)
    raw_root = layout.raw_source_root("csmar")
    auxiliary_root = raw_root.with_name("csmar_aux")
    auxiliary_root.mkdir(parents=True)
    (auxiliary_root / "native-calendar.csv").write_text(
        "Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n",
        encoding="utf-8",
    )
    (
        auxiliary_root / "csmar__special_treatment_changes__all_market_types__full__"
        "export-20260605.csv"
    ).write_text(
        "bad,header\n",
        encoding="utf-8",
    )

    result = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=("trading_calendar",),
    )

    assert result["status"] == "COMPLETE"
    assert result["target_results"][0]["status"] == "BUILT"


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
    assert inspected["blocker"] == "dataset_instances_missing"
    assert status["status"] == "BLOCKED"
    assert not (root / "spec.json").exists()
    assert not (root / "_manifest").exists()


def test_a_share_synthetic_chain_uses_the_single_resolver(tmp_path: Path) -> None:
    result = run_research_synthetic_smoke(tmp_path)

    assert result["status"] == "BLOCKED"
    assert result["dataset_status"] == "COMPLETE"
    assert result["dataset_repeat_status"] == "COMPLETE"
    assert result["dataset_target_status"] == "BUILT"
    assert result["dataset_repeat_target_status"] == "REUSED"
    assert result["source_producer_calls_on_repeat"] == 0
    assert result["research_status"] == "BLOCKED"
    assert result["report_status"] == "BLOCKED"

    inspection = research.inspect(
        profile=research_profile(),
        layout=StorageLayout(project_root=tmp_path),
        dataset_id="pv_all",
    )
    assert inspection["instances"]
    assert inspection["status"] == "BLOCKED"
    assert inspection["blocker"] == "request_identity_required"

    instance_root = next(
        StorageLayout(project_root=tmp_path)
        .research_dataset_root("a_share", "pv_all")
        .glob("instances/*")
    )
    manifest = read_dataset_manifest_evidence(instance_root)
    envelope = manifest["common_manifest"]
    assert envelope["research_window"]["access"] == "not_applicable"
    assert envelope["semantic_metadata"] == {}
