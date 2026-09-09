from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.artifacts.research import read_research_run_record
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.processed import (
    InterimDataCatalog,
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
    ResearchRunTargetResult,
    TargetSpec,
    TargetStatus,
)
from quant_research_platform.contracts.time import TimeWindow
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
from quant_research_platform.storage.manifests import (
    read_dataset_manifest_evidence,
    read_manifest,
)
from quant_research_platform.workflows import research
from tests.helpers.research_specs import make_demo_research_dataset_spec


def test_a_share_model_prediction_fixture_reports_formal_statistics_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.research.models import production as model_production

    model_read_windows: list[TimeWindow] = []
    real_iter_dataset_chunks = model_production.iter_dataset_chunks

    def observed_iter_dataset_chunks(session, **kwargs):
        model_read_windows.append(session.time_window)
        return real_iter_dataset_chunks(session, **kwargs)

    monkeypatch.setattr(
        model_production, "iter_dataset_chunks", observed_iter_dataset_chunks
    )
    base = research_profile()
    source_id = "formal_source"
    dataset_spec = make_demo_research_dataset_spec(
        data_id="pv_all",
        input_data_ids=(source_id,),
        base_input_data_id=source_id,
        factor_columns=("factor_a",),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=("label_available_at",),
    )
    source_spec = InterimDataSpec(
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
    evaluation_spec = EvaluationSpec(
        evaluation_id="pv_all_evaluation",
        market_id=MarketId("a_share"),
        research_dataset_id="pv_all",
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
    model_catalog = ModelCatalog(
        tuple(
            ModelSpec(
                model_id=f"pv_all_{algorithm}",
                model_algorithm=algorithm,
                input_factor_columns=("factor_a",),
                label_column="label",
                random_seed=42,
            )
            for algorithm in a_share.MODEL_FAMILY_ORDER
        )
    )
    sessions = pd.date_range("2014-01-01", "2019-06-30", freq="B")
    label_calendar = pd.bdate_range(sessions[0], periods=len(sessions) + 21)
    dataset_frame = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [asset for timestamp in sessions for asset in ("a", "b", "c")],
            "timestamp": [timestamp for timestamp in sessions for _ in range(3)],
            "label_available_at": [
                label_calendar[index + 21]
                for index in range(len(sessions))
                for _ in range(3)
            ],
            "factor_a": [
                float(asset_index + (date_index % 5))
                for date_index in range(len(sessions))
                for asset_index in range(1, 4)
            ],
            "label": [
                float(asset_index + (date_index % 5)) / 10.0
                for date_index in range(len(sessions))
                for asset_index in range(1, 4)
            ],
            "eligible": True,
        }
    )
    dataset_frame = pd.concat(
        [
            dataset_frame,
            pd.DataFrame(
                {
                    "market_id": ["a_share"],
                    "asset_id": ["holdout"],
                    "timestamp": pd.to_datetime(["2024-01-02"]),
                    "label_available_at": pd.to_datetime(["2024-01-31"]),
                    "factor_a": [float("nan")],
                    "label": [float("nan")],
                    "eligible": [True],
                }
            ),
        ],
        ignore_index=True,
    )
    source_frame = pd.DataFrame(
        {
            "source_asset_id": ["a"],
            "timestamp": [sessions[0]],
            "value": [1.0],
        }
    )

    def source_producer(**kwargs: object) -> InterimImportResult:
        output_layout = kwargs["layout"]
        output_root = output_layout.target_root(
            source_id,
            output_layout.interim_data_root(source_spec.source_id, source_id),
        )
        write_dataset(
            output_root,
            source_frame,
            spec=source_spec,
            instance_metadata=kwargs["instance_metadata"],
            resource_policy=kwargs["resource_policy"],
            borrowed_input=True,
        )
        return InterimImportResult(
            artifact_roots={source_id: output_root},
            skipped=(),
            pruning_report_text="formal synthetic source",
        )

    def dataset_producer(**kwargs: object) -> object:
        output_layout = kwargs["layout"]
        output_root = output_layout.target_root(
            "pv_all",
            output_layout.research_dataset_root(dataset_spec.market_id, "pv_all"),
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
        source_spec.source_id
    )
    raw_root.mkdir(parents=True)
    (raw_root / "formal_source.raw").write_bytes(b"formal")
    profile = replace(
        base,
        protocol=a_share.protocol,
        interim=InterimDataCatalog((*base.interim.list(), source_spec)),
        processed=base.processed,
        research_datasets=ResearchDatasetCatalog((dataset_spec,)),
        evaluations=EvaluationCatalog((evaluation_spec,)),
        models=model_catalog,
        producer_bindings={
            **base.producer_bindings,
            source_id: ProducerBinding(
                ProducerKind.INTERIM,
                source_producer,
                source_identity=lambda raw, data_id: content_hash_for_files(
                    ((data_id, raw / "formal_source.raw"),),
                    error="formal_source_identity_invalid",
                ),
            ),
            "pv_all": ProducerBinding(ProducerKind.DATASET, dataset_producer),
        },
    )
    result = research.research_run(
        profile=profile,
        layout=StorageLayout(project_root=tmp_path),
        scenario_id="pv_all",
        start_date="2019-01-01",
        end_date="2019-06-30",
    )
    targets = {item["target_id"]: item for item in result["target_results"]}
    assert result["status"] == "FAILED"
    assert targets["pv_all_statistics"] == {
        "target_id": "pv_all_statistics",
        "status": "FAILED",
        "reason": "statistics_build_failed:statistics_fdr_family_factors_missing",
        "instance_identity": None,
    }
    assert targets["pv_all_evaluation"]["status"] == "BUILT"
    assert targets["pv_all_model_selection_refit"]["status"] == "BUILT"
    assert "pv_all_ew_rank_model" not in targets
    assert targets["pv_all_persisted_predictions"]["status"] == "BUILT"
    assert "pv_all_ew_rank_predictions" not in targets
    model_root = (
        StorageLayout(project_root=tmp_path).semantic_artifact_root(
            "a_share", "pv_all_model_selection_refit"
        )
        / "instances"
        / targets["pv_all_model_selection_refit"]["instance_identity"]
    )
    from quant_research_platform.artifacts.outputs.bundle_validation import (
        read_prediction_output_bundle,
    )
    from quant_research_platform.research.models.persisted import (
        read_quarterly_model_artifact,
    )

    model_payload = json.loads((model_root / "model.json").read_text(encoding="utf-8"))
    assert set(model_payload) == {"schema_id", "model_id", "quarters"}
    assert model_payload["quarters"] == ["2019Q1", "2019Q2"]
    artifact = read_quarterly_model_artifact(
        model_root, trial_layout=StorageLayout(tmp_path)
    )
    assert len(artifact.quarterly_models) == 16
    assert set(item["family_id"] for item in artifact.quarterly_models) == set(
        protocol_family_ids := tuple(
            f"pv_all_{algorithm}" for algorithm in a_share.MODEL_FAMILY_ORDER
        )
    )
    for item in artifact.quarterly_models:
        metadata = item["model_metadata"]
        assert type(metadata["random_seed"]) is int
        assert "model_selection" not in metadata
        assert not {"test_start", "test_end", "oos_start", "oos_end"}.intersection(
            metadata
        )
    assert all(
        tuple(item["candidate_family_ids"]) == protocol_family_ids
        for item in artifact.quarterly_selections.values()
    )
    assert model_read_windows
    assert any(
        pd.Timestamp(window.start) == pd.Timestamp("2014-01-01")
        for window in model_read_windows
    )
    assert {
        window
        for window in model_read_windows
        if pd.Timestamp(window.start) > pd.Timestamp("2014-01-01")
    } == {
        TimeWindow(item["validation_start"], item["validation_end"])
        for item in artifact.quarterly_selections.values()
    }
    assert all(
        pd.Timestamp(window.end) <= pd.Timestamp(a_share.DEVELOPMENT_END)
        for window in model_read_windows
    )
    prediction_root = (
        StorageLayout(project_root=tmp_path).semantic_artifact_root(
            "a_share", "pv_all_persisted_predictions"
        )
        / "instances"
        / targets["pv_all_persisted_predictions"]["instance_identity"]
        / "outputs"
        / "pv_all_persisted_predictions"
    )
    prediction_manifest = read_manifest(prediction_root / "manifest.json")
    import pyarrow.parquet as pq

    from quant_research_platform.contracts.predictions import (
        A_SHARE_PREDICTION_PRIMARY_KEY,
        RAW_PREDICTION_COLUMN,
        prediction_output_columns,
    )
    from quant_research_platform.storage.identity import native_filesystem_path
    from quant_research_platform.storage.semantic_artifacts import (
        read_semantic_artifact,
    )

    declared = read_semantic_artifact(prediction_root.parent.parent)[
        "semantic_metadata"
    ]["target_contract"]["prediction_schema"]
    assert declared["columns"] == list(
        prediction_output_columns("a_share_oos", RAW_PREDICTION_COLUMN)
    )
    assert (
        declared["columns"]
        == pq.read_schema(
            native_filesystem_path(prediction_root / "prediction.parquet")
        ).names
    )
    assert declared["primary_key"] == list(A_SHARE_PREDICTION_PRIMARY_KEY)
    assert declared["primary_key"] == prediction_manifest["metadata"]["primary_key"]
    assert prediction_manifest["metadata"]["prediction_mode"] == "a_share_oos"
    assert prediction_manifest["metadata"]["prediction_column"] == "raw_prediction"
    keys = dataset_frame.loc[
        dataset_frame.timestamp.between("2019-01-01", "2019-06-30"),
        ["timestamp", "asset_id"],
    ]
    for selector in ("champion", "pv_all_equal_weight_rank_ensemble"):
        with read_prediction_output_bundle(
            prediction_root,
            expected_output_id="pv_all_persisted_predictions",
            family_selection=selector,
        ) as (_manifest, lookup):
            with lookup.values_for_frame(keys, mark_matched=True) as (values, found):
                assert all(found)
                assert len(values) == len(keys)
            assert lookup.coverage_counts() == (0, 0)
    assert prediction_manifest["metadata"]["primary_key"] == [
        "timestamp",
        "asset_id",
        "family_id",
    ]
    assert prediction_manifest["metadata"]["prediction_metadata"][
        "prediction_rows"
    ] == 8 * len(keys)
    layout = StorageLayout(tmp_path)
    first_record = read_research_run_record(layout.research_run_path(result["run_id"]))
    assert first_record.trial_ledgers
    model_before = (model_root / "manifest.json").read_bytes()
    reused = research.research_run(
        profile=profile,
        layout=layout,
        scenario_id="pv_all",
        start_date="2019-01-01",
        end_date="2019-06-30",
    )
    reused_record = read_research_run_record(layout.research_run_path(reused["run_id"]))
    assert not reused_record.trial_ledgers
    assert (
        next(
            item
            for item in reused["target_results"]
            if item["target_id"] == "pv_all_model_selection_refit"
        )["status"]
        == "REUSED"
    )
    rebuilt = research.research_run(
        profile=profile,
        layout=layout,
        scenario_id="pv_all",
        start_date="2019-01-01",
        end_date="2019-06-30",
        rebuild=True,
    )
    rebuild_record = read_research_run_record(
        layout.research_run_path(rebuilt["run_id"])
    )
    assert rebuild_record.trial_ledgers
    assert {item["run_id"] for item in rebuild_record.trial_ledgers} == {
        rebuilt["run_id"]
    }
    assert (model_root / "manifest.json").read_bytes() == model_before
    comparisons = {
        item["target_id"]: item["comparison_status"]
        for item in rebuilt["rebuild_evidence"]
    }
    assert comparisons["pv_all_model_selection_refit"] == "MATCH"
    assert comparisons["pv_all_persisted_predictions"] == "MATCH"
    assert (
        next(
            item
            for item in rebuilt["target_results"]
            if item["target_id"] == "pv_all_statistics"
        )["reason"]
        == "statistics_build_failed:statistics_fdr_family_factors_missing"
    )


def test_model_failure_blocks_real_semantic_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = research_profile()
    layout = StorageLayout(project_root=tmp_path)
    dataset_spec = make_demo_research_dataset_spec(
        data_id="pv_all",
        factor_columns=("x",),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=("label_available_at",),
    )
    write_dataset(
        layout.research_dataset_root(dataset_spec.market_id, dataset_spec.data_id),
        pd.DataFrame(
            {
                "market_id": ["a_share"],
                "asset_id": ["a"],
                "timestamp": pd.to_datetime(["2018-01-02"]),
                "label_available_at": pd.to_datetime(["2018-02-01"]),
                "x": [1.0],
                "label": [0.1],
                "eligible": [True],
            }
        ),
        spec=dataset_spec,
        resource_policy=None,
        borrowed_input=True,
    )
    base = replace(base, research_datasets=ResearchDatasetCatalog((dataset_spec,)))
    model_calls = 0

    def fail_model(**_kwargs: object) -> object:
        nonlocal model_calls
        model_calls += 1
        raise ValueError("mandatory_trial_failed")

    profile = replace(
        base,
        producer_bindings={
            **base.producer_bindings,
            "model": ProducerBinding(ProducerKind.MODEL, fail_model),
        },
    )
    model_target = "pv_all_model_selection_refit"
    terminals = base.protocol(scenario_id="pv_all").terminal_targets("research run")
    target_specs = (
        TargetSpec(model_target, producer_id="model"),
        *(
            TargetSpec(target_id, dependencies=(model_target,), producer_id=target_id)
            for target_id in terminals
        ),
    )
    monkeypatch.setattr(
        research,
        "_target_specs",
        lambda *_args, **_kwargs: target_specs,
    )

    result = research._execute_action(  # noqa: SLF001
        profile=profile,
        run_id="run_failure",
        action="research run",
        terminal_targets=terminals,
        layout=StorageLayout(project_root=tmp_path),
        time_window=TimeWindow("2019-01-01", "2019-03-31"),
        raw_root=tmp_path / "raw",
        start_date="2019-01-01",
        end_date="2019-03-31",
        research_window={
            "start": "2019-01-01",
            "end": "2019-03-31",
            "access": "development",
        },
    )
    results = {item["target_id"]: item for item in result["target_results"]}
    assert model_calls == 1
    assert results[model_target]["status"] == "FAILED"
    assert "mandatory_trial_failed" in results[model_target]["reason"]
    for target_id in terminals:
        assert results[target_id] == {
            "target_id": target_id,
            "status": "BLOCKED",
            "reason": f"dependency_blocked: {model_target}",
            "instance_identity": None,
        }
    assert result["status"] == "FAILED"


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


@pytest.mark.parametrize(
    ("target_id", "artifact_root"),
    (
        (
            r"x\..\..\..\..\escaped",
            r"data/artifacts/a_share/x\..\..\..\..\escaped",
        ),
        ("x", r"C:\data\artifacts\a_share\x"),
        ("x", r"\\server\share\x"),
    ),
)
def test_research_run_reference_rejects_windows_path_escape(
    target_id: str,
    artifact_root: str,
) -> None:
    with pytest.raises(ValueError, match="research_run_artifact_root_invalid"):
        ResearchRunTargetResult(
            target_id=target_id,
            producer_kind=ProducerKind.MODEL,
            status=TargetStatus.BUILT,
            instance_identity="a" * 64,
            schema_identity="b" * 64,
            artifact_root=artifact_root,
        )


def test_crypto_dataset_plan_reports_missing_raw_after_data_producers_are_bound(
    tmp_path: Path,
) -> None:
    profile = research_profile(CRYPTO_PROFILE_ID)
    dataset_id = profile.research_datasets.list()[0].data_id

    plan = research.plan_targets(
        profile=profile,
        action="dataset build",
        terminal_targets=(dataset_id,),
        layout=StorageLayout(project_root=tmp_path),
        research_window={
            "start": "1970-01-01T00:00:00",
            "end": "1970-01-01T00:00:06",
            "access": "development",
        },
    )

    assert plan["status"] == "BLOCKED"
    assert any(
        item["reason"] == "raw_content_identity_unavailable"
        for item in plan["blockers"]
    )


def test_protocol_owns_formal_research_and_report_graphs() -> None:
    a_share = research_profile().protocol("pv_all")
    a_share_specs = {
        spec.target_id: spec for spec in a_share.target_specs("research run")
    }
    assert a_share.terminal_targets("research run") == (
        "pv_all_evaluation",
        "pv_all_statistics",
        "pv_all_portfolio_statistics",
        "pv_all_ew_rank_top20_equal_weight_ledger",
        "pv_all_champion_top20_equal_weight_ledger",
        "pv_all_champion_top20_float_cap_ledger",
        "pv_all_champion_csi500_enhanced_ledger",
        "pv_all_champion_csi1000_enhanced_ledger",
    )
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
    assert "pv_all_ew_rank_model" not in a_share_specs
    assert "pv_all_ew_rank_predictions" not in a_share_specs
    assert a_share_specs["pv_all_ew_rank_top20_equal_weight_ledger"].dependencies == (
        "pv_all_persisted_predictions",
        "pv_all",
        "execution_mask",
        "trading_calendar",
        "a_share_fee_schedule",
        "market_panel_1d",
    )
    assert (
        "benchmark_weights"
        in a_share_specs["pv_all_champion_csi500_enhanced_ledger"].dependencies
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
        "pv_all_model_selection_refit",
        "pv_all",
    )
    assert a_share_specs["pv_all_statistics"].producer_id == "pv_all_statistics"
    assert (
        research_profile().producer_bindings["pv_all_statistics"].kind
        is ProducerKind.STATISTIC
    )
    ledger_ids = tuple(
        target_id for target_id in a_share_specs if target_id.endswith("_ledger")
    )
    assert len(ledger_ids) == 5
    assert a_share.target_specs("report build") == ()
    assert (
        research_profile().producer_bindings["pv_all_report_artifact"].kind
        is ProducerKind.REPORT
    )
    candidates = a_share.portfolio_candidates()
    assert [candidate.construction for candidate in candidates] == [
        "top20_equal_weight",
        "top20_equal_weight",
        "top20_float_cap",
        "benchmark_enhanced",
        "benchmark_enhanced",
    ]
    for candidate in candidates:
        dependencies = a_share_specs[
            f"pv_all_{candidate.candidate_id}_ledger"
        ].dependencies
        assert ("fundamental_factors" in dependencies) == (
            candidate.construction == "benchmark_enhanced"
        )
        if candidate.construction == "benchmark_enhanced":
            assert "price_volume_facts" in dependencies
    assert [
        a_share_specs[f"pv_all_{candidate.candidate_id}_ledger"].producer_id
        for candidate in candidates
    ] == [
        "pv_all_ew_rank_top20_equal_weight_ledger",
        "pv_all_champion_top20_equal_weight_ledger",
        "pv_all_champion_top20_float_cap_ledger",
        "pv_all_champion_csi500_enhanced_ledger",
        "pv_all_champion_csi1000_enhanced_ledger",
    ]

    crypto_profile = research_profile(CRYPTO_PROFILE_ID)
    crypto = crypto_profile.protocol(crypto_profile.research_datasets.list()[0].data_id)
    crypto_specs = {
        spec.target_id: spec for spec in crypto.target_specs("research run")
    }
    ledger_ids = tuple(
        candidate.target_id for candidate in crypto.portfolio_candidates()
    )
    assert len(ledger_ids) == 14
    assert (
        crypto_specs[
            "binance_futures_hft_research_100ms_portfolio_statistics"
        ].dependencies
        == ledger_ids
    )
    assert all(
        crypto_profile.producer_bindings[target].kind is ProducerKind.PORTFOLIO
        for target in ledger_ids
    )
    assert crypto.target_specs("report build") == ()
    assert (
        crypto_profile.producer_bindings[
            "binance_futures_hft_research_100ms_report_artifact"
        ].kind
        is ProducerKind.REPORT
    )


def test_generic_planner_rejects_retired_implicit_report_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="report_run_id_required"):
        research.plan_targets(
            profile=research_profile(),
            action="report build",
            terminal_targets=("pv_all_report_artifact",),
            layout=StorageLayout(project_root=tmp_path),
            scenario_id="pv_all",
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
    marker_root = raw_root / "csmar"
    marker_root.mkdir(parents=True)
    marker = marker_root / "source.raw"
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
    marker_root = raw_root / "csmar"
    marker_root.mkdir(parents=True)
    marker = marker_root / "source.raw"
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
    marker_root = raw_root / "csmar"
    marker_root.mkdir(parents=True)
    marker = marker_root / "source.raw"
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
    marker_root = raw_root / "csmar"
    marker_root.mkdir(parents=True)
    marker = marker_root / "source.raw"
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
    marker_root = raw_root / "csmar"
    marker_root.mkdir(parents=True)
    marker = marker_root / "source.raw"
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
    marker_root = raw_root / "csmar"
    marker_root.mkdir(parents=True)
    marker = marker_root / "source.raw"
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


def test_rebuild_validation_failure_reports_failed_without_empty_candidate(
    tmp_path: Path,
) -> None:
    primary = ValueError("producer validation failure")
    raw_root = tmp_path / "raw"
    marker_root = raw_root / "csmar"
    marker_root.mkdir(parents=True)
    marker = marker_root / "source.raw"
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
    assert failed["target_results"][0]["reason"] == "interim_build_failed:valueerror"
    assert read_dataset_manifest_evidence(old_root)["data_id"] == data_id
    assert calls["count"] == 2
    assert not tuple((tmp_path / ".run").glob(".r-*"))


@pytest.mark.parametrize("exception_type", (KeyboardInterrupt, SystemExit))
def test_rebuild_producer_control_base_exception_preserves_candidate(
    tmp_path: Path,
    exception_type: type[BaseException],
) -> None:
    primary = exception_type("producer control failure")
    raw_root = tmp_path / "raw"
    marker_root = raw_root / "csmar"
    marker_root.mkdir(parents=True)
    marker = marker_root / "source.raw"
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
    marker_root = raw_root / "csmar"
    marker_root.mkdir(parents=True)
    marker = marker_root / "source.raw"
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
    marker_root = raw_root / "csmar"
    marker_root.mkdir(parents=True)
    marker = marker_root / "source.raw"
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
    marker_root = raw_root / "csmar"
    marker_root.mkdir(parents=True)
    marker = marker_root / "source.raw"
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
    marker_root = raw_root / "csmar"
    marker_root.mkdir(parents=True)
    marker = marker_root / "source.raw"
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
    raw_root = tmp_path / "data" / "raw"
    nested_root = raw_root / "csmar" / "calendar" / "history"
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
    raw_root = tmp_path / "data" / "raw"
    result = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
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
    assert research_plan["status"] == "BLOCKED"
    assert not (tmp_path / "data" / "runs").exists()
    assert all(target["decision"] != "REUSE" for target in research_plan["targets"])

    research_result = research.research_run(
        profile=profile,
        layout=layout,
        scenario_id="pv_all",
        start_date="2019-01-01",
        end_date="2019-12-31",
    )
    run_id = research_result["run_id"]
    report_plan = research.report_build(
        profile=profile,
        layout=layout,
        run_id=run_id,
        plan=True,
    )
    assert research_result["status"] == "BLOCKED"
    assert report_plan["status"] == "BLOCKED"
    assert report_plan["target"]["decision"] == "BUILD"
    report_result = research.report_build(
        profile=profile,
        layout=layout,
        run_id=run_id,
    )
    assert report_result["status"] == "BLOCKED"
    assert report_result["source_run_status"] == "BLOCKED"
    assert report_result["target_result"]["status"] == "BUILT"
    assert report_result["report"]["report_status"] == "BLOCKED"

    record = read_research_run_record(layout.research_run_path(run_id))
    assert record.status.value == "BLOCKED"
    assert record.run_id == run_id
    with pytest.raises(ValueError, match="research_run_profile_mismatch"):
        research.report_build(
            profile=replace(profile, profile_id="different_profile"),
            layout=layout,
            run_id=run_id,
        )
    with pytest.raises(ValueError, match="invalid_run_id"):
        research.report_build(
            profile=profile,
            layout=layout,
            run_id="../outside",
        )

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


def test_research_run_records_unexpected_failure_without_masking_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = research_profile()
    layout = StorageLayout(project_root=tmp_path)
    import quant_research_platform.runtime.execution_metrics as observations

    actual_clock = observations.perf_counter
    times = iter((0.0, 0.1))
    monkeypatch.setattr(observations, "perf_counter", lambda: next(times))

    primary = RuntimeError("SYNTHETIC_ROW_VALUE_X")

    def interrupt(**_kwargs: object) -> dict[str, object]:
        raise primary

    monkeypatch.setattr(research, "_plan_targets", interrupt)
    with pytest.raises(RuntimeError) as raised:
        research.research_run(
            profile=profile,
            layout=layout,
            scenario_id="pv_all",
            start_date="2019-01-01",
            end_date="2019-12-31",
        )
    assert raised.value is primary
    records = tuple((tmp_path / "data/runs/research").glob("*.json"))
    assert len(records) == 1
    record = read_research_run_record(records[0])
    assert record.status.value == "FAILED"
    assert record.failure_reason == "research_run_failed:runtimeerror"
    assert record.operation_evidence["reason"] == "operation_incomplete"
    assert record.operation_evidence["sample_count"] == 1
    assert "SYNTHETIC_ROW_VALUE_X" not in records[0].read_text(encoding="utf-8")
    monkeypatch.setattr(observations, "perf_counter", actual_clock)

    monkeypatch.setattr(
        research,
        "write_research_run_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("secondary")),
    )
    with pytest.raises(RuntimeError) as secondary_raised:
        research.research_run(
            profile=profile,
            layout=StorageLayout(project_root=tmp_path / "secondary"),
            scenario_id="pv_all",
            start_date="2019-01-01",
            end_date="2019-12-31",
        )
    assert secondary_raised.value is primary


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
    assert status == {"profile_id": profile.profile_id, "runs": []}
    assert not (root / "spec.json").exists()
    assert not (root / "_manifest").exists()


def _crypto_plan_payload():
    return {
        "folds": [
            {
                "fold_id": "fold_1",
                "train": {"start": "1970-01-01T00:00:00", "end": "1970-01-01T00:00:01"},
                "validation": {
                    "start": "1970-01-01T00:00:01.100000",
                    "end": "1970-01-01T00:00:03",
                },
                "development_test": {
                    "start": "1970-01-01T00:00:04",
                    "end": "1970-01-01T00:00:06",
                },
            }
        ],
        "final_holdout": {"start": "1970-01-01T00:00:07", "end": "1970-01-01T00:00:08"},
        "initial_capital": 1000.0,
        "valuation_currency": "USDT",
        "exposure_limits": {
            asset: {"max_order_quantity": 1.0, "max_position_quantity": 2.0}
            for asset in ("btc_usdt", "eth_usdt")
        },
    }


def test_crypto_research_plan_is_explicit_immutable_and_holdout_locked(tmp_path):
    from typer.testing import CliRunner

    from quant_research_platform.interfaces.cli.commands import app
    from quant_research_platform.profiles.research import (
        CRYPTO_PROFILE_ID,
        research_profile,
    )
    from quant_research_platform.research.protocols.crypto_hft import (
        HFT_RESEARCH_DATASET_ID,
    )

    profile = research_profile(CRYPTO_PROFILE_ID)
    request = dict(
        profile=profile,
        layout=StorageLayout(tmp_path),
        scenario_id=HFT_RESEARCH_DATASET_ID,
        start_date="1970-01-01T00:00:04",
        end_date="1970-01-01T00:00:06",
        plan=True,
    )
    with pytest.raises(ValueError, match="crypto_research_plan_required"):
        research.research_run(**request)
    payload = _crypto_plan_payload()
    bound = profile.protocol(research_plan=payload)
    frozen = bound.manifest()
    payload["initial_capital"] = 9999.0
    payload["exposure_limits"]["btc_usdt"]["max_position_quantity"] = 999.0
    assert bound.manifest() == frozen
    assert frozen["holdout_status"] == "HOLDOUT_LOCKED"
    for field in (
        "folds",
        "final_holdout",
        "initial_capital",
        "valuation_currency",
        "exposure_limits",
    ):
        bad = _crypto_plan_payload()
        del bad[field]
        with pytest.raises(ValueError, match="crypto_research_plan_fields_invalid"):
            research.research_run(**request, research_plan=bad)
    bad = _crypto_plan_payload()
    bad["raw_root"] = "private/provider"
    with pytest.raises(ValueError, match="crypto_research_plan_fields_invalid"):
        research.research_run(**request, research_plan=bad)
    bad = _crypto_plan_payload()
    bad["folds"][0]["validation"]["end"] = "1970-01-01T00:00:01.200000"
    with pytest.raises(ValueError, match="crypto_research_plan_label_window_too_short"):
        research.research_run(**request, research_plan=bad)
    with pytest.raises(PermissionError, match="holdout_frozen"):
        research.research_run(
            **request, research_plan=_crypto_plan_payload(), holdout=True
        )
    valid = _crypto_plan_payload()
    for fold in valid["folds"]:
        for role in ("train", "validation", "development_test"):
            fold[role] = {key: value + "Z" for key, value in fold[role].items()}
    valid["final_holdout"] = {
        key: value + "Z" for key, value in valid["final_holdout"].items()
    }
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(valid), encoding="utf-8")
    args = [
        "research",
        "run",
        "--profile",
        CRYPTO_PROFILE_ID,
        "--start-date",
        request["start_date"] + "Z",
        "--end-date",
        request["end_date"] + "Z",
        "--project-root",
        str(tmp_path),
        "--plan",
    ]
    missing = CliRunner().invoke(app, args)
    assert missing.exit_code == 2 and "crypto_research_plan_required" in missing.output
    result = CliRunner().invoke(app, [*args, "--research-plan", str(plan_file)])
    assert result.exit_code == 0, result.output
    published = json.loads(result.output)
    assert published["protocol"]["research_plan"] == _crypto_plan_payload()
    assert published["protocol"]["holdout_status"] == "HOLDOUT_LOCKED"
    assert published["status"] == "BLOCKED"
    assert "research_plan" not in profile.protocol().manifest()


@pytest.mark.parametrize(
    "kind", (ProducerKind.INTERIM, ProducerKind.PROCESSED, ProducerKind.DATASET)
)
@pytest.mark.parametrize("failure", ("duplicate", "interrupt", "memory", "io"))
def test_target_local_validation_preserves_independent_production(
    tmp_path, monkeypatch, kind, failure
):
    from contextlib import contextmanager

    from quant_research_platform.catalog.processed import ProcessedDataCatalog
    from quant_research_platform.contracts.data import (
        ProcessedDataKind,
        ProcessedDataSpec,
    )
    from quant_research_platform.contracts.datasets import ResearchDatasetSpec

    a, b, c = (
        ("fusion_nf_core", "fusion_nf_expectations", "pv_all")
        if kind is ProducerKind.DATASET
        else ("a", "b", "c")
    )
    asset_column = "source_asset_id" if kind is ProducerKind.INTERIM else "asset_id"
    roles = {
        asset_column: ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
        "value": ColumnRole.FACTOR,
        "label": ColumnRole.LABEL,
    }
    if kind is not ProducerKind.INTERIM:
        roles = {"market_id": ColumnRole.KEY, **roles}
    specs = []
    for name in (a, b, c):
        fields = dict(
            data_id=name,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles=roles,
            primary_key=(asset_column, "timestamp"),
            partition_columns=("year",),
        )
        if kind is ProducerKind.INTERIM:
            spec = InterimDataSpec(**fields, source_id=SourceId("csmar"))
        elif kind is ProducerKind.PROCESSED:
            spec = ProcessedDataSpec(
                **fields, market_id=MarketId("a_share"), kind=ProcessedDataKind.PANEL
            )
        else:
            spec = ResearchDatasetSpec(
                **fields,
                market_id=MarketId("a_share"),
                input_data_ids=("base",),
                base_input_data_id="base",
                factor_columns=("value",),
                label_columns=("label",),
            )
        specs.append(spec)
    by_id = {spec.data_id: spec for spec in specs}
    calls = []
    primary = {
        "interrupt": KeyboardInterrupt("target_primary"),
        "memory": MemoryError("target_primary"),
        "io": OSError("target_primary"),
    }.get(failure)

    def frame(name):
        calls.append(name)
        if name == a and primary is not None:
            raise primary
        values = {
            asset_column: ["asset", "asset"],
            "timestamp": pd.to_datetime(
                ["2020-01-02", "2020-01-02" if name == a else "2020-01-03"]
            ),
            "value": [1.0, 2.0],
            "label": [0.1, 0.2],
        }
        if kind is not ProducerKind.INTERIM:
            values = {"market_id": ["a_share", "a_share"], **values}
        return pd.DataFrame(values)

    def persisted(name, kwargs):
        spec = by_id[name]
        layout = kwargs["layout"]
        root = layout.target_root(name, tmp_path / "unused")
        write_dataset(
            root,
            frame(name),
            spec=spec,
            instance_metadata=kwargs["instance_metadata"],
            resource_policy=kwargs["resource_policy"],
            borrowed_input=True,
        )
        return root

    def interim(**kwargs):
        name = kwargs["data_ids"][0]
        return InterimImportResult(
            artifact_roots={name: persisted(name, kwargs)},
            skipped=(),
            pruning_report_text="synthetic",
        )

    @contextmanager
    def processed(spec, **kwargs):
        from quant_research_platform.runtime.resources import OwnedValue

        def frames():
            lease = yield 65536
            yield OwnedValue(frame(spec.data_id), lease)

        yield frames(), {}, None

    def dataset(**kwargs):
        persisted(kwargs["dataset_id"], kwargs)

    producer = {
        ProducerKind.INTERIM: interim,
        ProducerKind.PROCESSED: processed,
        ProducerKind.DATASET: dataset,
    }[kind]
    catalogs = (
        {"interim": InterimDataCatalog(tuple(specs))}
        if kind is ProducerKind.INTERIM
        else {"processed": ProcessedDataCatalog(tuple(specs))}
        if kind is ProducerKind.PROCESSED
        else {"research_datasets": ResearchDatasetCatalog(tuple(specs))}
    )
    profile = replace(
        research_profile(),
        **catalogs,
        producer_bindings={
            name: ProducerBinding(
                kind,
                producer,
                source_identity=(lambda *_: "a" * 64)
                if kind is ProducerKind.INTERIM
                else None,
            )
            for name in by_id
        },
    )
    monkeypatch.setattr(
        research,
        "_target_specs",
        lambda *_args, **_kwargs: (
            TargetSpec(a, producer_id=a),
            TargetSpec(b, dependencies=(a,), producer_id=b),
            TargetSpec(c, producer_id=c),
        ),
    )
    raw = tmp_path / "raw"
    (raw / "csmar").mkdir(parents=True)
    kwargs = dict(
        profile=profile,
        layout=StorageLayout(tmp_path),
        raw_root=raw,
        start_date="2020-01-01",
        end_date="2020-01-31",
        data_ids=(b, c),
    )
    if primary is not None:
        with pytest.raises(type(primary)) as raised:
            research.data_materialize(**kwargs)
        assert raised.value is primary
        assert calls == [a]
        return
    result = research.data_materialize(**kwargs)
    results = {item["target_id"]: item for item in result["target_results"]}
    assert result["status"] == "FAILED"
    assert calls == [a, c]
    assert results[a]["status"] == "FAILED"
    assert "duplicate_dataset_primary_key" in results[a]["reason"]
    assert results[b]["status"] == "BLOCKED"
    assert results[b]["reason"] == f"dependency_blocked: {a}"
    assert results[c]["status"] == "BUILT"
