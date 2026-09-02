from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
    ProcessedDataCatalog,
)
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    InterimDataSpec,
)
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.signals import SignalSpec
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
from quant_research_platform.storage.datasets.types import DatasetComparison
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import content_hash_for_files
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import read_dataset_manifest_evidence
from quant_research_platform.workflows import research
from quant_research_platform.workflows.smoke import run_research_synthetic_smoke
from tests.helpers.research_specs import make_demo_research_dataset_spec


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
    assert a_share_specs["pv_all_evaluation"].producer_id == "pv_all_evaluation"
    assert a_share_specs["pv_all_model_selection_refit"].dependencies == ("pv_all",)
    assert (
        a_share_specs["pv_all_model_selection_refit"].producer_id
        == "pv_all_model_selection_refit"
    )
    assert a_share_specs["pv_all_persisted_predictions"].dependencies == (
        "pv_all",
        "pv_all_model_selection_refit",
    )
    assert (
        a_share_specs["pv_all_persisted_predictions"].producer_id
        == "pv_all_persisted_predictions"
    )
    assert research_profile().evaluations.get("pv_all_evaluation").evaluation_id == (
        "pv_all_evaluation"
    )
    assert (
        research_profile().producer_bindings["pv_all_evaluation"].kind
        is ProducerKind.EVALUATION
    )
    assert (
        research_profile().producer_bindings["pv_all_model_selection_refit"].kind
        is ProducerKind.MODEL
    )
    assert (
        research_profile().producer_bindings["pv_all_persisted_predictions"].kind
        is ProducerKind.PREDICTION
    )
    assert a_share_specs["pv_all_statistics"].dependencies == (
        "pv_all_persisted_predictions",
    )
    ledger_ids = tuple(
        target_id for target_id in a_share_specs if target_id.endswith("_ledger")
    )
    assert len(ledger_ids) == 5
    assert set(a_share_specs["pv_all_research"].dependencies) == {
        "pv_all_evaluation",
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
            "reason": "source_identity_changed",
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


def test_same_identity_rebuild_matches_and_preserves_old_good(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_id = "market_bars_1d"
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    marker = raw_root / "source.raw"
    marker.write_bytes(b"stable")
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
            ((data_id, raw / marker.name),),
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
    old_root = layout.instance_root(
        layout.interim_data_root("csmar", data_id),
        first["target_results"][0]["instance_identity"],
    )
    before = tuple(
        (
            path.relative_to(old_root).as_posix(),
            path.read_bytes(),
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in old_root.rglob("*")
        if path.is_file()
    )

    rebuild_plan = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
        plan=True,
        rebuild=True,
    )
    planned_target = rebuild_plan["targets"][0]
    assert planned_target["decision"] == "BUILD"
    assert planned_target["matching_instance_identity"] == old_root.name
    assert planned_target["requested_identity"] == old_root.name
    normal_plan = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
        plan=True,
    )
    assert normal_plan["targets"][0]["decision"] == "REUSE"
    assert normal_plan["targets"][0]["matching_instance_identity"] == old_root.name

    rebuilt = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
        rebuild=True,
    )

    after = tuple(
        (
            path.relative_to(old_root).as_posix(),
            path.read_bytes(),
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in old_root.rglob("*")
        if path.is_file()
    )
    assert rebuilt["status"] == "COMPLETE"
    assert rebuilt["target_results"][0]["status"] == "BUILT"
    assert rebuilt["target_results"][0]["instance_identity"] == old_root.name
    assert rebuilt["rebuild_evidence"][0]["comparison_status"] == "MATCH"
    assert before == after
    assert calls == 2
    assert not tuple((tmp_path / ".run").glob(".r-*"))

    def mismatch(*_args: object, **_kwargs: object) -> DatasetComparison:
        return DatasetComparison(
            status="CONTENT_MISMATCH",
            matches=False,
            summary={},
            layout_delta={},
        )

    monkeypatch.setattr(research, "compare_datasets", mismatch)
    failed = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
        rebuild=True,
    )
    assert failed["status"] == "FAILED"
    assert failed["target_results"][0]["status"] == "FAILED"
    assert failed["target_results"][0]["reason"] == (
        "rebuild_comparison_content_mismatch"
    )
    assert failed["rebuild_evidence"][0]["comparison_status"] == "CONTENT_MISMATCH"
    assert tuple((tmp_path / ".run").glob(".r-*"))
    failed_snapshot = tuple(
        (
            path.relative_to(old_root).as_posix(),
            path.read_bytes(),
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in old_root.rglob("*")
        if path.is_file()
    )
    assert failed_snapshot == before
    assert calls == 3


def _calendar_rebuild_profile(
    variant: str,
    *,
    marker: Path,
    failure: BaseException | None = None,
) -> tuple[object, str, dict[str, int]]:
    data_id = "trading_calendar"
    calls = {"count": 0}

    def producer(**kwargs: object) -> InterimImportResult:
        calls["count"] += 1
        if calls["count"] > 1 and failure is not None:
            raise failure
        producer_layout = kwargs["layout"]
        spec = research_profile().interim.get(data_id)
        output_root = producer_layout.target_root(
            data_id,
            producer_layout.interim_data_root(spec.source_id, data_id),
        )
        count = 1 if calls["count"] == 1 or variant != "row" else 2
        values = {
            "timestamp": pd.date_range("2020-01-02", periods=count),
            "calendar_date": pd.date_range("2020-01-02", periods=count),
            "source_market_type": [4 if calls["count"] == 1 else 5] * count,
            "is_trading_day": [True] * count,
            "source_day_of_week": [4] * count,
            "source_state": ["O" if calls["count"] == 1 else "C"] * count,
        }
        frame = pd.DataFrame(values)
        if variant == "schema" and calls["count"] > 1:
            frame["is_trading_day"] = pd.Series([1] * count, dtype="int64")
        if variant == "value" and calls["count"] > 1:
            frame["source_market_type"] = [4] * count
            frame["source_state"] = ["C"] * count
        if variant == "key" and calls["count"] > 1:
            frame["source_state"] = ["O"] * count
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
            ((data_id, raw / marker.name),),
            error="source_identity_invalid",
        )

    return (
        replace(
            research_profile(),
            producer_bindings={
                data_id: ProducerBinding(
                    ProducerKind.INTERIM,
                    producer,
                    source_identity=source_identity,
                ),
            },
        ),
        data_id,
        calls,
    )


@pytest.mark.parametrize(
    ("variant", "comparison_status"),
    (
        ("value", "CONTENT_MISMATCH"),
        ("row", "ROW_COUNT_MISMATCH"),
        ("key", "CONTENT_MISMATCH"),
        ("schema", "SCHEMA_MISMATCH"),
    ),
)
def test_rebuild_comparison_mismatch_families_preserve_old_good(
    tmp_path: Path,
    variant: str,
    comparison_status: str,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    marker = raw_root / "source.raw"
    marker.write_bytes(b"stable")
    profile, data_id, calls = _calendar_rebuild_profile(variant, marker=marker)
    layout = StorageLayout(project_root=tmp_path)
    first = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
    )
    old_root = layout.instance_root(
        layout.interim_data_root("csmar", data_id),
        first["target_results"][0]["instance_identity"],
    )
    before = tuple(
        (
            path.relative_to(old_root).as_posix(),
            path.read_bytes(),
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in old_root.rglob("*")
        if path.is_file()
    )

    rebuilt = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
        rebuild=True,
    )

    after = tuple(
        (
            path.relative_to(old_root).as_posix(),
            path.read_bytes(),
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in old_root.rglob("*")
        if path.is_file()
    )
    assert rebuilt["status"] == "FAILED"
    assert rebuilt["target_results"][0]["status"] == "FAILED"
    assert rebuilt["target_results"][0]["reason"] == (
        f"rebuild_comparison_{comparison_status.lower()}"
    )
    assert rebuilt["rebuild_evidence"][0]["comparison_status"] == comparison_status
    assert before == after
    assert calls["count"] == 2
    assert tuple((tmp_path / ".run").glob(".r-*"))


def test_rebuild_corrupt_old_good_returns_inspection_failed(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    marker = raw_root / "source.raw"
    marker.write_bytes(b"stable")
    profile, data_id, calls = _calendar_rebuild_profile("value", marker=marker)
    layout = StorageLayout(project_root=tmp_path)
    first = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
    )
    old_root = layout.instance_root(
        layout.interim_data_root("csmar", data_id),
        first["target_results"][0]["instance_identity"],
    )
    (old_root / "_manifest" / "manifest.json").unlink()

    rebuilt = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
        rebuild=True,
    )

    assert rebuilt["status"] == "FAILED"
    assert rebuilt["target_results"][0]["status"] == "FAILED"
    assert rebuilt["target_results"][0]["reason"] == (
        "rebuild_comparison_inspection_failed"
    )
    assert rebuilt["rebuild_evidence"][0]["comparison_status"] == ("INSPECTION_FAILED")
    assert calls["count"] == 2
    assert tuple((tmp_path / ".run").glob(".r-*"))


def test_rebuild_compare_primary_is_preserved_with_candidate_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    marker = raw_root / "source.raw"
    marker.write_bytes(b"stable")
    profile, data_id, calls = _calendar_rebuild_profile("value", marker=marker)
    layout = StorageLayout(project_root=tmp_path)
    first = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
    )
    old_root = layout.instance_root(
        layout.interim_data_root("csmar", data_id),
        first["target_results"][0]["instance_identity"],
    )
    before = tuple(
        (
            path.relative_to(old_root).as_posix(),
            path.read_bytes(),
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in old_root.rglob("*")
        if path.is_file()
    )

    class PrimaryError(Exception):
        pass

    primary = PrimaryError("comparison failure")

    def raise_primary(*_args: object, **_kwargs: object) -> DatasetComparison:
        raise primary

    monkeypatch.setattr(research, "compare_datasets", raise_primary)
    with pytest.raises(PrimaryError) as caught:
        research.data_materialize(
            profile=profile,
            layout=layout,
            raw_root=raw_root,
            start_date="2020-01-01",
            end_date="2020-12-31",
            data_ids=(data_id,),
            rebuild=True,
        )

    after = tuple(
        (
            path.relative_to(old_root).as_posix(),
            path.read_bytes(),
            path.stat().st_ino,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in old_root.rglob("*")
        if path.is_file()
    )
    assert caught.value is primary
    assert before == after
    assert calls["count"] == 2
    assert tuple((tmp_path / ".run").glob(".r-*"))


def test_rebuild_producer_primary_is_preserved_with_candidate_evidence(
    tmp_path: Path,
) -> None:
    class PrimaryError(Exception):
        pass

    primary = PrimaryError("producer failure")
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    marker = raw_root / "source.raw"
    marker.write_bytes(b"stable")
    profile, data_id, calls = _calendar_rebuild_profile(
        "value",
        marker=marker,
        failure=primary,
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
    old_root = layout.instance_root(
        layout.interim_data_root("csmar", data_id),
        first["target_results"][0]["instance_identity"],
    )

    with pytest.raises(PrimaryError) as caught:
        research.data_materialize(
            profile=profile,
            layout=layout,
            raw_root=raw_root,
            start_date="2020-01-01",
            end_date="2020-12-31",
            data_ids=(data_id,),
            rebuild=True,
        )

    assert caught.value is primary
    assert read_dataset_manifest_evidence(old_root)["data_id"] == data_id
    assert calls["count"] == 2
    assert tuple((tmp_path / ".run").glob(".r-*"))


def test_rebuild_producer_validation_failure_preserves_candidate(
    tmp_path: Path,
) -> None:
    primary = ValueError("producer validation failure")
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    marker = raw_root / "source.raw"
    marker.write_bytes(b"stable")
    profile, data_id, calls = _calendar_rebuild_profile(
        "value",
        marker=marker,
        failure=primary,
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
    old_root = layout.instance_root(
        layout.interim_data_root("csmar", data_id),
        first["target_results"][0]["instance_identity"],
    )

    with pytest.raises(ValueError) as caught:
        research.data_materialize(
            profile=profile,
            layout=layout,
            raw_root=raw_root,
            start_date="2020-01-01",
            end_date="2020-12-31",
            data_ids=(data_id,),
            rebuild=True,
        )

    assert caught.value is primary
    assert read_dataset_manifest_evidence(old_root)["data_id"] == data_id
    assert calls["count"] == 2
    assert tuple((tmp_path / ".run").glob(".r-*"))


@pytest.mark.parametrize("exception_type", (KeyboardInterrupt, SystemExit))
def test_rebuild_producer_control_base_exception_preserves_candidate(
    tmp_path: Path,
    exception_type: type[BaseException],
) -> None:
    primary = exception_type("producer control failure")
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    marker = raw_root / "source.raw"
    marker.write_bytes(b"stable")
    profile, data_id, calls = _calendar_rebuild_profile(
        "value",
        marker=marker,
        failure=primary,
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
    old_root = layout.instance_root(
        layout.interim_data_root("csmar", data_id),
        first["target_results"][0]["instance_identity"],
    )

    with pytest.raises(exception_type) as caught:
        research.data_materialize(
            profile=profile,
            layout=layout,
            raw_root=raw_root,
            start_date="2020-01-01",
            end_date="2020-12-31",
            data_ids=(data_id,),
            rebuild=True,
        )

    assert caught.value is primary
    assert read_dataset_manifest_evidence(old_root)["data_id"] == data_id
    assert calls["count"] == 2
    assert tuple((tmp_path / ".run").glob(".r-*"))


def test_rebuild_raw_mutation_before_candidate_build_blocks_without_empty_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    marker = raw_root / "source.raw"
    marker.write_bytes(b"stable")
    profile, data_id, calls = _calendar_rebuild_profile("value", marker=marker)
    layout = StorageLayout(project_root=tmp_path)
    first = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
    )
    old_identity = first["target_results"][0]["instance_identity"]

    plan = research._plan_targets

    def mutate_after_plan(**kwargs: object) -> dict[str, object]:
        result = plan(**kwargs)
        marker.write_bytes(b"changed")
        return result

    monkeypatch.setattr(research, "_plan_targets", mutate_after_plan)
    rebuilt = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
        rebuild=True,
    )

    assert rebuilt["status"] == "BLOCKED"
    assert rebuilt["target_results"] == [
        {
            "target_id": data_id,
            "status": "BLOCKED",
            "reason": "source_identity_changed",
            "instance_identity": None,
        }
    ]
    assert rebuilt["rebuild_evidence"] == []
    assert calls["count"] == 1
    assert not tuple((tmp_path / ".run").glob(".r-*"))
    assert old_identity is not None


def test_rebuild_without_old_good_is_a_normal_immutable_build(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    marker = raw_root / "source.raw"
    marker.write_bytes(b"stable")
    profile, data_id, calls = _calendar_rebuild_profile("value", marker=marker)

    result = research.data_materialize(
        profile=profile,
        layout=StorageLayout(project_root=tmp_path),
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
        rebuild=True,
    )

    assert result["status"] == "COMPLETE"
    assert result["target_results"][0]["status"] == "BUILT"
    assert result["rebuild_evidence"] == []
    assert calls["count"] == 1
    assert not tuple((tmp_path / ".run").glob(".r-*"))


def test_rebuild_blocked_before_materialization_cleans_empty_candidate(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    marker = raw_root / "source.raw"
    marker.write_bytes(b"stable")
    profile, data_id, calls = _calendar_rebuild_profile("value", marker=marker)
    layout = StorageLayout(project_root=tmp_path)
    first = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
    )
    binding = profile.producer_bindings[data_id]

    def blocked_producer(**_kwargs: object) -> InterimImportResult:
        calls["count"] += 1
        return InterimImportResult(
            artifact_roots={},
            skipped=(data_id,),
            pruning_report_text="blocked",
        )

    blocked_profile = replace(
        profile,
        producer_bindings={
            data_id: ProducerBinding(
                ProducerKind.INTERIM,
                blocked_producer,
                source_identity=binding.source_identity,
            )
        },
    )
    result = research.data_materialize(
        profile=blocked_profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
        rebuild=True,
    )

    assert first["status"] == "COMPLETE"
    assert result["status"] == "BLOCKED"
    assert result["target_results"] == [
        {
            "target_id": data_id,
            "status": "BLOCKED",
            "reason": "external_data_unavailable",
            "instance_identity": None,
        }
    ]
    assert result["rebuild_evidence"] == []
    assert calls["count"] == 2
    assert not tuple((tmp_path / ".run").glob(".r-*"))


def test_rebuild_mismatch_blocks_real_descendant_before_producer_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    marker = raw_root / "source.raw"
    marker.write_bytes(b"stable")
    profile, source_id, source_calls = _calendar_rebuild_profile(
        "value",
        marker=marker,
    )
    descendant_id = "market_bars_1d"
    descendant_calls = {"count": 0}

    def descendant_producer(**kwargs: object) -> InterimImportResult:
        descendant_calls["count"] += 1
        spec = profile.interim.get(descendant_id)
        frame = pd.DataFrame(
            {
                column: pd.Series(
                    dtype=(
                        "datetime64[ns]" if role is ColumnRole.TIMESTAMP else "string"
                    )
                )
                for column, role in spec.column_roles.items()
            }
        )
        producer_layout = kwargs["layout"]
        output_root = producer_layout.target_root(
            descendant_id,
            producer_layout.interim_data_root(spec.source_id, descendant_id),
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
            artifact_roots={descendant_id: output_root},
            skipped=(),
            pruning_report_text="test",
        )

    profile = replace(
        profile,
        producer_bindings={
            **profile.producer_bindings,
            descendant_id: ProducerBinding(
                ProducerKind.INTERIM,
                descendant_producer,
                source_identity=lambda raw, _data_id: content_hash_for_files(
                    ((descendant_id, raw / marker.name),),
                    error="source_identity_invalid",
                ),
            ),
        },
    )
    declared_specs = (
        TargetSpec(source_id, producer_id=source_id),
        TargetSpec(
            descendant_id,
            dependencies=(source_id,),
            producer_id=descendant_id,
        ),
    )

    def target_specs(
        _profile: object,
        *,
        action: str,
        protocol: object,
    ) -> tuple[TargetSpec, ...]:
        del action, protocol
        return declared_specs

    monkeypatch.setattr(research, "_target_specs", target_specs)
    layout = StorageLayout(project_root=tmp_path)
    first = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(descendant_id,),
    )
    assert first["status"] == "COMPLETE"
    assert source_calls["count"] == 1
    assert descendant_calls["count"] == 1

    failed = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(descendant_id,),
        rebuild=True,
    )

    results = {item["target_id"]: item for item in failed["target_results"]}
    assert failed["status"] == "FAILED"
    assert results[source_id]["status"] == "FAILED"
    assert results[source_id]["reason"] == ("rebuild_comparison_content_mismatch")
    assert results[descendant_id] == {
        "target_id": descendant_id,
        "status": "BLOCKED",
        "reason": f"dependency_blocked: {source_id}",
        "instance_identity": None,
    }
    assert source_calls["count"] == 2
    assert descendant_calls["count"] == 1
    assert tuple((tmp_path / ".run").glob(".r-*"))


def test_csmar_dataset_action_discovers_nested_single_root(
    tmp_path: Path,
) -> None:
    profile = research_profile()
    layout = StorageLayout(project_root=tmp_path)
    raw_root = layout.raw_source_root("csmar")
    nested_root = raw_root / "calendar" / "history"
    nested_root.mkdir(parents=True)
    (nested_root / "native-calendar.csv").write_text(
        "Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n",
        encoding="utf-8",
    )
    (
        nested_root / "csmar__special_treatment_changes__all_market_types__full__"
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


def test_csmar_dataset_action_missing_target_is_blocked(tmp_path: Path) -> None:
    profile = research_profile()
    layout = StorageLayout(project_root=tmp_path)
    result = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=layout.raw_source_root("csmar"),
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=("trading_calendar",),
    )

    assert result["status"] == "BLOCKED"
    assert result["target_results"] == [
        {
            "target_id": "trading_calendar",
            "status": "BLOCKED",
            "reason": "raw_content_identity_unavailable",
            "instance_identity": None,
        }
    ]


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
    assert all("reason" in item for item in rebuild["target_results"])


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
    assert result["dataset_rebuild_status"] == "COMPLETE"
    assert result["dataset_rebuild_target_status"] == "BUILT"
    assert result["dataset_rebuild_evidence_count"] > 0
    assert result["source_producer_calls_on_repeat"] == 0
    assert result["source_producer_calls_on_rebuild"] > 0
    assert not tuple((tmp_path / ".run").glob(".r-*"))
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


def test_synthetic_research_run_builds_evaluation_model_and_prediction_targets(
    tmp_path: Path,
) -> None:
    base = research_profile()
    dataset_id = "demo"
    source_id = "synthetic_base"
    interim_spec = InterimDataSpec(
        data_id=source_id,
        source_id=SourceId("csmar"),
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            "source_asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "value": ColumnRole.CONTEXT,
        },
        primary_key=("source_asset_id", "timestamp"),
        partition_columns=("year",),
    )
    dataset_spec = make_demo_research_dataset_spec(
        data_id=dataset_id,
        input_data_ids=(source_id,),
        base_input_data_id=source_id,
        factor_columns=("factor_a",),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=(),
    )
    evaluation_spec = EvaluationSpec(
        evaluation_id=f"{dataset_id}_evaluation",
        market_id=MarketId("a_share"),
        research_dataset_id=dataset_id,
        signals=(
            SignalSpec(
                signal_id="factor_a",
                source_kind="factor",
                source_column="factor_a",
            ),
        ),
        label_columns=("label",),
        n_groups=2,
    )
    model_spec = ModelSpec(
        model_id=f"{dataset_id}_linear_regression",
        model_algorithm="linear_regression",
        input_factor_columns=("factor_a",),
        label_column="label",
    )
    rank_model_spec = ModelSpec(
        model_id=f"{dataset_id}_equal_weight_rank_ensemble",
        model_algorithm="equal_weight_rank_ensemble",
        input_factor_columns=("factor_a",),
        label_column="label",
    )

    @dataclass(frozen=True)
    class SyntheticProtocol:
        scenario_id: str
        access_window: str = "development"

        def manifest(self) -> dict[str, object]:
            return {
                "protocol_id": "synthetic_protocol",
                "scenario_id": self.scenario_id,
                "access_window": self.access_window,
                "selection_tie_break": [
                    "linear_regression",
                    "equal_weight_rank_ensemble",
                ],
            }

        def target_specs(self, action: str) -> tuple[TargetSpec, ...]:
            if action != "research run":
                return ()
            evaluation = f"{self.scenario_id}_evaluation"
            model = f"{self.scenario_id}_model_selection_refit"
            prediction = f"{self.scenario_id}_persisted_predictions"
            return (
                TargetSpec(evaluation, (self.scenario_id,), evaluation),
                TargetSpec(model, (self.scenario_id,), model),
                TargetSpec(prediction, (self.scenario_id, model), prediction),
                TargetSpec(
                    f"{self.scenario_id}_research",
                    (evaluation, prediction),
                ),
            )

        def terminal_targets(self, action: str) -> tuple[str, ...]:
            if action == "research run":
                return (f"{self.scenario_id}_research",)
            raise ValueError("unsupported_synthetic_action")

    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    dataset_frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [asset for timestamp in dates for asset in ("a", "b", "c")],
            "timestamp": [timestamp for timestamp in dates for _ in range(3)],
            "factor_a": [
                float(asset_index + date_index)
                for date_index in range(1, len(dates) + 1)
                for asset_index in range(1, 4)
            ],
            "label": [
                float(asset_index + date_index) / 10.0
                for date_index in range(1, len(dates) + 1)
                for asset_index in range(1, 4)
            ],
            "eligible": True,
        }
    )
    source_frame = pd.DataFrame(
        {
            "source_asset_id": ["a"],
            "timestamp": [dates[0]],
            "value": [1.0],
        }
    )

    def source_identity(raw_root: Path, data_id: str) -> str | None:
        path = Path(raw_root) / f"{data_id}.raw"
        if not path.is_file():
            return None
        return content_hash_for_files(
            ((data_id, path),),
            error="synthetic_source_identity_invalid",
        )

    def materialize_source(**kwargs: object) -> InterimImportResult:
        producer_layout = kwargs["layout"]
        output_root = producer_layout.target_root(
            source_id,
            producer_layout.interim_data_root(
                interim_spec.source_id,
                source_id,
            ),
        )
        write_dataset(
            output_root,
            source_frame,
            spec=interim_spec,
            instance_metadata=kwargs["instance_metadata"],
            resource_policy=kwargs["resource_policy"],
            borrowed_input=True,
        )
        return InterimImportResult(
            artifact_roots={source_id: output_root},
            skipped=(),
            pruning_report_text="synthetic source",
        )

    def materialize_dataset(**kwargs: object) -> object:
        producer_layout = kwargs["layout"]
        output_root = producer_layout.target_root(
            dataset_id,
            producer_layout.research_dataset_root(
                dataset_spec.market_id,
                dataset_id,
            ),
        )
        return write_dataset(
            output_root,
            dataset_frame,
            spec=dataset_spec,
            instance_metadata=kwargs["instance_metadata"],
            resource_policy=kwargs["resource_policy"],
            borrowed_input=True,
        )

    raw_root = StorageLayout(project_root=tmp_path).raw_source_root(
        interim_spec.source_id
    )
    raw_root.mkdir(parents=True)
    (raw_root / f"{source_id}.raw").write_bytes(b"synthetic")
    profile = replace(
        base,
        protocol=lambda *, scenario_id, access_window: SyntheticProtocol(
            scenario_id,
            access_window,
        ),
        interim=InterimDataCatalog((interim_spec,)),
        processed=ProcessedDataCatalog(()),
        research_datasets=ResearchDatasetCatalog((dataset_spec,)),
        evaluations=EvaluationCatalog((evaluation_spec,)),
        models=ModelCatalog((model_spec, rank_model_spec)),
        producer_bindings={
            source_id: ProducerBinding(
                ProducerKind.INTERIM,
                materialize_source,
                source_identity=source_identity,
            ),
            dataset_id: ProducerBinding(ProducerKind.DATASET, materialize_dataset),
            f"{dataset_id}_evaluation": base.producer_bindings["pv_all_evaluation"],
            f"{dataset_id}_model_selection_refit": base.producer_bindings[
                "pv_all_model_selection_refit"
            ],
            f"{dataset_id}_persisted_predictions": base.producer_bindings[
                "pv_all_persisted_predictions"
            ],
        },
    )

    result = research.research_run(
        profile=profile,
        layout=StorageLayout(project_root=tmp_path),
        scenario_id=dataset_id,
        start_date="2020-01-01",
        end_date="2020-01-06",
    )
    targets = {item["target_id"]: item for item in result["target_results"]}
    assert result["status"] == "BLOCKED"
    assert targets[f"{dataset_id}_evaluation"]["status"] == "BUILT"
    assert targets[f"{dataset_id}_model_selection_refit"]["status"] == "BUILT"
    assert targets[f"{dataset_id}_persisted_predictions"]["status"] == "BUILT"
    assert (tmp_path / "data/artifacts/a_share").is_dir()
    for target_id in (
        f"{dataset_id}_evaluation",
        f"{dataset_id}_model_selection_refit",
        f"{dataset_id}_persisted_predictions",
    ):
        instance = (
            StorageLayout(project_root=tmp_path).semantic_artifact_root(
                "a_share", target_id
            )
            / "instances"
            / targets[target_id]["instance_identity"]
        )
        assert (instance / "manifest.json").is_file()
    assert (
        StorageLayout(project_root=tmp_path)
        .semantic_artifact_root("a_share", f"{dataset_id}_model_selection_refit")
        .joinpath(
            "instances",
            targets[f"{dataset_id}_model_selection_refit"]["instance_identity"],
            "model.json",
        )
        .is_file()
    )

    repeated = research.research_run(
        profile=profile,
        layout=StorageLayout(project_root=tmp_path),
        scenario_id=dataset_id,
        start_date="2020-01-01",
        end_date="2020-01-06",
    )
    repeated_targets = {item["target_id"]: item for item in repeated["target_results"]}
    assert repeated_targets[f"{dataset_id}_evaluation"]["status"] == "REUSED"
    assert repeated_targets[f"{dataset_id}_model_selection_refit"]["status"] == "REUSED"
    assert repeated_targets[f"{dataset_id}_persisted_predictions"]["status"] == "REUSED"
