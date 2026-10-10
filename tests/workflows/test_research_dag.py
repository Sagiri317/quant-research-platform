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
    TargetPlan,
    TargetSpec,
    TargetStatus,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.interfaces import queries
from quant_research_platform.profiles.research import (
    CRYPTO_PROFILE_ID,
    research_profile,
)
from quant_research_platform.research.protocols import a_share
from quant_research_platform.sources.csmar.interim import InterimImportResult
from quant_research_platform.storage.datasets.types import DatasetComparison
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import (
    content_hash_for_files,
    filesystem_exists,
    filesystem_glob,
    filesystem_is_file,
    filesystem_iterdir,
    filesystem_mkdir,
    filesystem_read_bytes,
    filesystem_read_text,
    filesystem_stat,
    filesystem_unlink,
    filesystem_write_bytes,
    filesystem_write_text,
    native_filesystem_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import (
    read_dataset_manifest_evidence,
    read_manifest,
)
from quant_research_platform.workflows import research
from tests.helpers.research_configuration import (
    compact_a_share_configuration,
    compact_a_share_windows,
    research_configuration,
)
from tests.helpers.research_specs import make_demo_research_dataset_spec


def _old_good_snapshot(root: Path) -> tuple[tuple[str, bytes, int, int, int], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            filesystem_read_bytes(path),
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
        )
        for path in filesystem_glob(root, "**/*")
        if filesystem_is_file(path)
        for info in (filesystem_stat(path),)
    )


@compact_a_share_windows()
def test_a_share_model_prediction_fixture_reports_formal_statistics_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.research.models import (
        production as model_production,
    )

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
    algorithms = ("equal_weight_rank_ensemble", "ridge_regression")
    small_configuration = compact_a_share_configuration()
    small_configuration["models"] = [
        model
        for model in small_configuration["models"]
        if model["model_algorithm"] in algorithms
    ]
    model_catalog = ModelCatalog(
        tuple(
            ModelSpec(
                model_id=f"pv_all_{algorithm}",
                model_algorithm=algorithm,
                input_factor_columns=("factor_a",),
                label_column="label",
                random_seed=42,
                hyperparameters=next(
                    item.get("hyperparameters", {})
                    for item in small_configuration["models"]
                    if item["model_algorithm"] == algorithm
                ),
            )
            for algorithm in algorithms
        )
    )
    sessions = pd.date_range("2017-10-01", "2019-06-30", freq="B")
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
    # Development fixtures cannot expose labels requiring holdout facts.
    unavailable = dataset_frame.label_available_at > pd.Timestamp(
        a_share.DEVELOPMENT_END
    )
    dataset_frame.loc[unavailable, "label"] = float("nan")
    dataset_frame.loc[unavailable, "label_available_at"] = pd.NaT
    dataset_frame = pd.concat(
        [
            dataset_frame,
            pd.DataFrame(
                {
                    "market_id": ["a_share"],
                    "asset_id": ["holdout"],
                    "timestamp": pd.to_datetime(["2019-07-02"]),
                    "label_available_at": pd.to_datetime(["2019-07-31"]),
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
        )
        return InterimImportResult(
            artifact_roots={source_id: output_root},
            skipped=(),
        )

    def dataset_producer(**kwargs: object) -> object:
        output_layout = kwargs["layout"]
        output_root = output_layout.target_root(
            "pv_all",
            output_layout.research_dataset_root(dataset_spec.market_id, "pv_all"),
        )
        return write_dataset(
            output_root,
            dataset_frame.loc[
                dataset_frame.timestamp.between(
                    pd.Timestamp(kwargs["time_window"].start),
                    pd.Timestamp(kwargs["time_window"].end),
                )
            ],
            spec=dataset_spec,
            instance_metadata=kwargs["instance_metadata"],
        )

    raw_root = StorageLayout(project_root=tmp_path).raw_source_root(
        source_spec.source_id
    )
    filesystem_mkdir(raw_root, parents=True)
    filesystem_write_bytes(raw_root / "formal_source.raw", b"formal")
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
    from functools import partial

    from quant_research_platform.profiles import research as profile_module

    configuration = small_configuration
    configuration["portfolios"] = []
    configuration["primary_portfolio_id"] = None
    # Keep registered labels; the synthetic dataset deliberately lacks them.
    for model in configuration["models"]:
        spec = model_catalog.get(model["model_id"])
        model.update(
            input_factor_columns=list(spec.input_factor_columns),
            input_feature_columns=list(spec.input_feature_columns),
            label_column=spec.label_column,
            hyperparameters=dict(spec.hyperparameters),
            random_seed=spec.random_seed,
        )
        model.pop("parameter_candidates", None)
    profile = replace(
        profile,
        protocol=partial(
            a_share.AShareProtocol,
            configuration_json=json.dumps(configuration, sort_keys=True),
        ),
    )
    # Isolate DAG persistence over the deliberately incomplete synthetic catalog;
    # configuration binding is covered separately with the real market catalog.
    monkeypatch.setattr(
        profile_module,
        "bind_research_configuration",
        lambda selected, configuration, **kwargs: selected,
    )
    result = research.research_run(
        profile=profile,
        layout=StorageLayout(project_root=tmp_path),
        research_plan=configuration,
    )
    targets = {item["target_id"]: item for item in result["target_results"]}
    assert result["status"] == "FAILED"
    assert targets["pv_all_global_statistics"] == {
        "target_id": "pv_all_global_statistics",
        "status": "FAILED",
        "reason": "statistics_build_failed:dataset_column_not_found",
        "instance_identity": None,
    }
    assert targets["pv_all_quarterly_statistics"]["status"] == "BLOCKED"
    assert targets["pv_all_evaluation"]["status"] == "BUILT"
    assert targets["pv_all_model_selection_refit"]["status"] == "BUILT"
    assert "pv_all_ew_rank_model" not in targets
    assert all(
        targets[f"pv_all_{group}_persisted_predictions"]["status"] == "BUILT"
        for group in ("global", "quarterly")
    )
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

    model_payload = json.loads(
        filesystem_read_text(model_root / "model.json", encoding="utf-8")
    )
    assert set(model_payload) == {
        "schema_id",
        "model_id",
        "validation_quarters",
        "selection_decisions",
        "final_groups",
    }
    quarters = [str(period) for period in pd.period_range("2018Q1", "2019Q2", freq="Q")]
    assert model_payload["validation_quarters"] == quarters
    artifact = read_quarterly_model_artifact(
        model_root, trial_layout=StorageLayout(tmp_path)
    )
    assert len(artifact.candidate_models) == len(quarters) * len(model_catalog.list())
    assert len(artifact.final_models) == 2 * len(model_catalog.list())
    protocol_family_ids = tuple(f"pv_all_{algorithm}" for algorithm in algorithms)
    assert {item["family_id"] for item in artifact.candidate_models} == set(
        protocol_family_ids
    )
    assert {item["selection_group"] for item in artifact.final_models} == {
        "global",
        "quarterly",
    }
    for item in artifact.candidate_models:
        metadata = item["model_metadata"]
        assert type(metadata["random_seed"]) is int
        assert "model_selection" not in metadata
        assert not {"test_start", "test_end", "oos_start", "oos_end"}.intersection(
            metadata
        )
    assert set(artifact.selections["global"]) == {"final"}
    assert set(artifact.selections["quarterly"]) == {
        *(str(period) for period in pd.period_range("2019Q1", "2019Q2", freq="Q")),
        "final",
    }
    assert model_read_windows
    assert any(
        pd.Timestamp(window.start) == pd.Timestamp("2017-10-01")
        for window in model_read_windows
    )
    observed_windows = {
        (pd.Timestamp(window.start), pd.Timestamp(window.end))
        for window in model_read_windows
    }
    assert {
        (
            pd.Timestamp(item["validation_window"]["start"]),
            pd.Timestamp(item["validation_window"]["end"]),
        )
        for item in artifact.candidate_models
    }.issubset(observed_windows)
    assert all(
        pd.Timestamp(window.end) <= pd.Timestamp(a_share.DEVELOPMENT_END)
        for window in model_read_windows
    )
    for group in ("global", "quarterly"):
        prediction_id = f"pv_all_{group}_persisted_predictions"
        prediction_root = (
            StorageLayout(project_root=tmp_path).semantic_artifact_root(
                "a_share", prediction_id
            )
            / "instances"
            / targets[prediction_id]["instance_identity"]
            / "outputs"
            / prediction_id
        )
        prediction_manifest = read_manifest(prediction_root / "manifest.json")
        import pyarrow.parquet as pq

        from quant_research_platform.contracts.predictions import (
            A_SHARE_PREDICTION_PRIMARY_KEY,
            RAW_PREDICTION_COLUMN,
            prediction_output_columns,
        )
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
            (_manifest, lookup) = read_prediction_output_bundle(
                prediction_root,
                expected_output_id=prediction_id,
                family_selection=selector,
            )
            values, found = lookup.values_for_frame(keys, mark_matched=True)
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
        ] == len(model_catalog.list()) * len(keys)
    layout = StorageLayout(tmp_path)
    first_record = read_research_run_record(layout.research_run_path(result["run_id"]))
    assert first_record.trial_ledgers
    model_before = filesystem_read_bytes(model_root / "manifest.json")
    reused = research.research_run(
        profile=profile,
        layout=layout,
        research_plan=configuration,
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
        rebuild=True,
        research_plan=configuration,
    )
    rebuild_record = read_research_run_record(
        layout.research_run_path(rebuilt["run_id"])
    )
    assert not rebuild_record.trial_ledgers
    assert filesystem_read_bytes(model_root / "manifest.json") == model_before
    comparisons = {
        item["target_id"]: item["comparison_status"]
        for item in rebuilt["rebuild_evidence"]
    }
    assert comparisons == {"pv_all_evaluation": "MATCH"}
    rebuilt_targets = {item["target_id"]: item for item in rebuilt["target_results"]}
    assert rebuilt_targets["pv_all_model_selection_refit"]["status"] == "REUSED"
    assert all(
        rebuilt_targets[f"pv_all_{group}_persisted_predictions"]["status"] == "REUSED"
        for group in ("global", "quarterly")
    )
    assert (
        next(
            item
            for item in rebuilt["target_results"]
            if item["target_id"] == "pv_all_global_statistics"
        )["reason"]
        == "statistics_build_failed:dataset_column_not_found"
    )


def test_model_failure_blocks_real_semantic_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.profiles.research import bind_research_configuration

    base = bind_research_configuration(research_profile(), research_configuration())
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
            "pv_all_model_selection_refit": ProducerBinding(
                ProducerKind.MODEL, fail_model
            ),
        },
    )
    model_target = "pv_all_model_selection_refit"
    terminals = base.protocol(scenario_id="pv_all").terminal_targets("research run")
    target_specs = (
        TargetSpec(
            model_target,
        ),
        *(
            TargetSpec(
                target_id,
                dependencies=(model_target,),
            )
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
        protocol=profile.protocol(
            scenario_id=research._default_scenario_id(profile),
            access_window="development",
        ),
        run_id="run_failure",
        action="research run",
        terminal_targets=terminals,
        layout=StorageLayout(project_root=tmp_path),
        raw_root=tmp_path / "raw",
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
        TargetSpec(
            "root",
            ("b", "a"),
        ),
        TargetSpec(
            "a",
        ),
        TargetSpec(
            "b",
            ("leaf",),
        ),
        TargetSpec(
            "leaf",
        ),
    )
    second = (
        TargetSpec(
            "root",
            ("a", "b"),
        ),
        TargetSpec(
            "b",
            ("leaf",),
        ),
        TargetSpec(
            "leaf",
        ),
        TargetSpec(
            "a",
        ),
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
    from quant_research_platform.profiles.research import bind_research_configuration
    from tests.helpers.research_configuration import research_configuration

    a_profile = bind_research_configuration(
        research_profile(), research_configuration(name="core6")
    )
    a_share = a_profile.protocol("pv_all")
    a_share_specs = {
        spec.target_id: spec for spec in a_share.target_specs("research run")
    }
    base_candidates = (
        "ew_rank_top20_equal_weight",
        "champion_top20_equal_weight",
        "champion_top20_float_cap",
        "champion_csi500_enhanced",
        "champion_csi1000_enhanced",
    )
    ledger_ids = tuple(
        f"pv_all_{group}_{candidate}_ledger"
        for candidate in base_candidates
        for group in ("global", "quarterly")
    )
    assert a_share.terminal_targets("research run") == (
        "pv_all_evaluation",
        "pv_all_factor_statistics",
        "pv_all_global_statistics",
        "pv_all_quarterly_statistics",
        *ledger_ids,
        "pv_all_portfolio_statistics",
    )
    assert a_share_specs["pv_all_evaluation"].dependencies == ("pv_all",)
    assert a_share_specs["pv_all_evaluation"].target_id == "pv_all_evaluation"
    model_id = "pv_all_model_selection_refit"
    assert a_share_specs[model_id].dependencies == ("pv_all",)
    assert a_share_specs[model_id].target_id == model_id
    assert a_profile.producer_bindings[model_id].kind is ProducerKind.MODEL
    for group in ("global", "quarterly"):
        prediction = f"pv_all_{group}_persisted_predictions"
        statistics = f"pv_all_{group}_statistics"
        assert a_share_specs[prediction].dependencies == ("pv_all", model_id)
        assert a_share_specs[prediction].target_id == prediction
        assert a_profile.producer_bindings[prediction].kind is ProducerKind.PREDICTION
        assert a_share_specs[statistics].dependencies == (
            prediction,
            model_id,
            "pv_all",
            *(("pv_all_global_statistics",) if group == "quarterly" else ()),
        )
        assert a_share_specs[statistics].target_id == statistics
        assert a_profile.producer_bindings[statistics].kind is ProducerKind.STATISTIC
    assert "pv_all_ew_rank_model" not in a_share_specs
    assert "pv_all_ew_rank_predictions" not in a_share_specs
    assert "pv_all_persisted_predictions" not in a_share_specs
    assert "pv_all_statistics" not in a_share_specs
    candidates = a_share.portfolio_candidates()
    assert len(candidates) == len(ledger_ids) == 10
    for group in ("global", "quarterly"):
        selected = tuple(item for item in candidates if item.selection_group == group)
        assert [item.construction for item in selected] == [
            "top20_equal_weight",
            "top20_equal_weight",
            "top20_float_cap",
            "benchmark_enhanced",
            "benchmark_enhanced",
        ]
        first = selected[0]
        assert a_share_specs[f"pv_all_{first.candidate_id}_ledger"].dependencies == (
            f"pv_all_{group}_persisted_predictions",
            "pv_all",
            "execution_mask",
            "trading_calendar",
            "a_share_fee_schedule",
            "market_panel_1d",
            "market_bars_1d",
            "stock_master",
            *first.risk_context_target_ids,
        )
        for candidate in selected:
            target = f"pv_all_{candidate.candidate_id}_ledger"
            assert a_share_specs[target].target_id == target
            assert (
                candidate.prediction_target_id
                == f"pv_all_{group}_persisted_predictions"
            )
            dependencies = a_share_specs[target].dependencies
            assert ("fundamental_factors" in dependencies) == (
                candidate.construction == "benchmark_enhanced"
            )
            assert ("benchmark_weights" in dependencies) == (
                candidate.construction == "benchmark_enhanced"
            )
            assert "price_volume_facts" in dependencies
    assert {target for target in a_share_specs if target.endswith("_ledger")} == set(
        ledger_ids
    )
    assert a_share.target_specs("report build") == ()
    assert (
        a_profile.producer_bindings["pv_all_report_artifact"].kind
        is ProducerKind.REPORT
    )
    assert (
        a_profile.evaluations.get("pv_all_evaluation").evaluation_id
        == "pv_all_evaluation"
    )
    assert (
        a_profile.producer_bindings["pv_all_evaluation"].kind is ProducerKind.EVALUATION
    )

    crypto_profile = bind_research_configuration(
        research_profile(CRYPTO_PROFILE_ID),
        research_configuration("crypto", "baseline"),
    )
    scenario = crypto_profile.research_datasets.list()[0].data_id
    development = crypto_profile.protocol(scenario)
    development_specs = {
        spec.target_id: spec
        for spec in research.resolve_target_graph(
            research._target_specs(
                crypto_profile, action="research run", protocol=development
            ),
            development.terminal_targets("research run"),
        )
    }
    assert scenario + "_models" in development_specs
    assert scenario + "_predictions" not in development_specs
    assert not any(target.endswith("_ledger") for target in development_specs)
    crypto = crypto_profile.protocol(scenario, access_window="holdout")
    crypto_specs = {
        spec.target_id: spec for spec in crypto.target_specs("research run")
    }
    execution_targets = crypto.execution_targets()
    ledger_ids = tuple(target.target_id for target in execution_targets)
    assert len(ledger_ids) == 35
    assert sorted(ledger_ids) == sorted(
        target_id
        for target_id in crypto_specs
        if crypto_profile.producer_bindings[target_id].kind is ProducerKind.PORTFOLIO
    )
    assert all(
        crypto_specs[target.statistics_target_id].dependencies == (target.target_id,)
        for target in execution_targets
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
    filesystem_mkdir(panel_root, parents=True)
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
    optional = set(panel_spec.optional_source_input_data_ids)
    required = set(panel_spec.source_input_data_ids) - optional
    assert required.issubset(panel["upstream_target_ids"])
    assert optional.isdisjoint(panel["upstream_target_ids"])
    assert all(
        panel["upstream_identities"][name]
        == research.canonical_json_hash({"source_id": name, "presence": "ABSENT"})
        for name in optional
    )


@pytest.mark.parametrize("deep_root", (False, True))
def test_interim_binding_receives_one_requested_target(
    tmp_path: Path, deep_root: bool
) -> None:
    if deep_root:
        while len(str(tmp_path)) <= 270:
            tmp_path /= "segment_" + "a" * 32
        assert len(str(tmp_path)) > 260
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
            output_root, frame, spec=spec, instance_metadata=kwargs["instance_metadata"]
        )
        return InterimImportResult(
            artifact_roots={data_id: output_root},
            skipped=(),
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

    reused = research.data_materialize(
        profile=profile,
        layout=StorageLayout(project_root=tmp_path),
        raw_root=tmp_path / "raw",
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
    )
    assert result["status"] == reused["status"] == "COMPLETE"
    assert result["target_results"][0]["status"] == "BUILT"
    assert reused["target_results"][0]["status"] == "REUSED"
    assert (
        result["target_results"][0]["instance_identity"]
        == reused["target_results"][0]["instance_identity"]
    )
    assert calls == [(data_id,)]


def test_source_identity_mutation_builds_a_new_immutable_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_id = "market_bars_1d"
    raw_root = tmp_path / "raw"
    marker_root = raw_root / "csmar"
    filesystem_mkdir(marker_root, parents=True)
    marker = marker_root / "source.raw"
    filesystem_write_bytes(marker, b"before")
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
            output_root, frame, spec=spec, instance_metadata=kwargs["instance_metadata"]
        )
        return InterimImportResult(
            artifact_roots={data_id: output_root},
            skipped=(),
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
    filesystem_write_bytes(marker, b"after")
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
        filesystem_iterdir(layout.interim_data_root("csmar", data_id) / "instances")
    )
    assert {path.name for path in instances} == {first_identity, second_identity}

    plan = research._plan_targets

    def mutate_after_plan(**kwargs: object) -> dict[str, object]:
        result = plan(**kwargs)
        filesystem_write_bytes(marker, b"between-plan-and-reuse")
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
    from quant_research_platform.profiles.research import bind_research_configuration
    from tests.helpers.research_configuration import research_configuration

    full_profile = replace(
        bind_research_configuration(base, research_configuration(name="pv25")),
        producer_bindings={data_id: binding},
    )
    core_profile = replace(
        bind_research_configuration(base, research_configuration(name="core6")),
        producer_bindings={data_id: binding},
    )

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
    filesystem_mkdir(marker_root, parents=True)
    marker = marker_root / "source.raw"
    filesystem_write_bytes(marker, b"stable")
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
            output_root, frame, spec=spec, instance_metadata=kwargs["instance_metadata"]
        )
        return InterimImportResult(
            artifact_roots={data_id: output_root},
            skipped=(),
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
    before = _old_good_snapshot(old_root)

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

    after = _old_good_snapshot(old_root)
    assert rebuilt["status"] == "COMPLETE"
    assert rebuilt["target_results"][0]["status"] == "BUILT"
    assert rebuilt["target_results"][0]["instance_identity"] == old_root.name
    assert rebuilt["rebuild_evidence"][0]["comparison_status"] == "MATCH"
    assert before == after
    assert calls == 2
    assert not tuple(filesystem_glob(tmp_path / ".run", ".r-*"))

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
    assert tuple(filesystem_glob(tmp_path / ".run", ".r-*"))
    failed_snapshot = _old_good_snapshot(old_root)
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
        metadata = kwargs["instance_metadata"]
        if variant == "metadata":
            frame["source_market_type"] = 4
            frame["source_state"] = "O"
            metadata = {
                **metadata,
                "producer_diagnostics": {
                    "reconstruction": {
                        "BTCUSDT": {
                            "coverage_segments": {
                                "segment": [
                                    pd.Timestamp("2020-01-01", tz="UTC").value,
                                    pd.Timestamp("2020-01-01", tz="UTC").value
                                    + (100 if calls["count"] == 1 else 90),
                                ]
                            }
                        }
                    }
                },
            }
        write_dataset(output_root, frame, spec=spec, instance_metadata=metadata)
        return InterimImportResult(
            artifact_roots={data_id: output_root},
            skipped=(),
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


def test_plan_reuse_requires_native_physical_members(tmp_path):
    raw_root = tmp_path / "raw"
    marker = raw_root / "csmar" / "source.raw"
    filesystem_mkdir(marker.parent, parents=True)
    filesystem_write_bytes(marker, b"stable")
    profile, data_id, calls = _calendar_rebuild_profile("value", marker=marker)
    layout = StorageLayout(project_root=tmp_path)
    request = dict(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
    )
    built = research.data_materialize(**request)
    root = layout.instance_root(
        layout.interim_data_root("csmar", data_id),
        built["target_results"][0]["instance_identity"],
    )
    manifest = root / "_manifest" / "manifest.json"
    before = filesystem_read_bytes(manifest)
    qualified = research.data_materialize(**request, plan=True)
    assert qualified["status"] == "COMPLETE"
    assert qualified["targets"][0]["decision"] == "REUSE"
    (member,) = tuple(filesystem_glob(root / "data", "**/*.parquet"))
    filesystem_unlink(member)
    rejected = research.data_materialize(**request, plan=True)
    assert rejected["status"] == "BLOCKED"
    assert rejected["targets"][0]["decision"] == "BLOCKED"
    assert rejected["targets"][0]["blocker"] == "persisted_instance_invalid"
    executed = research.data_materialize(**request)
    assert executed["status"] == "BLOCKED"
    assert executed["target_results"][0]["reason"] == "persisted_instance_invalid"
    assert filesystem_read_bytes(manifest) == before
    assert calls["count"] == 1


def test_reuse_retains_one_native_qualification_and_rejects_later_change(
    tmp_path, monkeypatch
):
    raw_root = tmp_path / "raw"
    marker = raw_root / "csmar" / "source.raw"
    filesystem_mkdir(marker.parent, parents=True)
    filesystem_write_bytes(marker, b"stable")
    profile, data_id, calls = _calendar_rebuild_profile("value", marker=marker)
    layout = StorageLayout(project_root=tmp_path)
    request = dict(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
    )
    built = research.data_materialize(**request)
    root = layout.instance_root(
        layout.interim_data_root("csmar", data_id),
        built["target_results"][0]["instance_identity"],
    )
    observed = []
    inspect = research.inspect_dataset_session

    def inspect_native(session):
        observed.append(session.root)
        return inspect(session)

    monkeypatch.setattr(research, "inspect_dataset_session", inspect_native)
    reused = research.data_materialize(**request)
    assert reused["status"] == "COMPLETE"
    assert reused["target_results"][0]["status"] == "REUSED"
    assert observed == [root]
    plan = research._plan_targets

    def mutate_after_plan(**kwargs):
        result = plan(**kwargs)
        (member,) = tuple(filesystem_glob(root / "data", "**/*.parquet"))
        filesystem_write_bytes(member, filesystem_read_bytes(member) + b"\0")
        return result

    monkeypatch.setattr(research, "_plan_targets", mutate_after_plan)
    changed = research.data_materialize(**request)
    assert changed["status"] == "FAILED"
    assert changed["target_results"][0]["reason"] == (
        "interim_build_failed:dataset_file_identity_changed"
    )
    assert observed == [root, root]
    assert calls["count"] == 1


@pytest.mark.parametrize(
    ("variant", "comparison_status"),
    (
        ("value", "CONTENT_MISMATCH"),
        ("row", "ROW_COUNT_MISMATCH"),
        ("key", "CONTENT_MISMATCH"),
        ("schema", "SCHEMA_MISMATCH"),
        ("metadata", "METADATA_MISMATCH"),
    ),
)
def test_rebuild_comparison_mismatch_families_preserve_old_good(
    tmp_path: Path,
    variant: str,
    comparison_status: str,
) -> None:
    raw_root = tmp_path / "raw"
    marker_root = raw_root / "csmar"
    filesystem_mkdir(marker_root, parents=True)
    marker = marker_root / "source.raw"
    filesystem_write_bytes(marker, b"stable")
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
    before = _old_good_snapshot(old_root)

    rebuilt = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date="2020-01-01",
        end_date="2020-12-31",
        data_ids=(data_id,),
        rebuild=True,
    )

    after = _old_good_snapshot(old_root)
    assert rebuilt["status"] == "FAILED"
    assert rebuilt["target_results"][0]["status"] == "FAILED"
    assert rebuilt["target_results"][0]["reason"] == (
        f"rebuild_comparison_{comparison_status.lower()}"
    )
    assert rebuilt["rebuild_evidence"][0]["comparison_status"] == comparison_status
    assert before == after
    assert calls["count"] == 2
    assert tuple(filesystem_glob(tmp_path / ".run", ".r-*"))
    if variant == "metadata":
        from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
        from quant_research_platform.markets.implementations.crypto.portfolio import (
            _market_coverage_check,
        )
        from quant_research_platform.sources.binance import replay
        from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
        from tests.sources.binance.test_replay import _write_coverage_history

        coverage_root = tmp_path / "coverage"
        index = replay.ObservationCoverageIndex(
            coverage_root,
            _write_coverage_history(coverage_root, 1, 32),
            markets={FUTURES_MARKET_TYPE},
            decode_interval=replay._decode_recorder_interval,
        )
        trades = replay.ObservationCoverage(
            index, FUTURES_MARKET_TYPE, "agg_trades", "BTCUSDT"
        )
        start = pd.Timestamp("2020-01-01", tz="UTC").value
        bounds = [(start + 5, start + 95)]

        def coverage(root):
            return read_dataset_manifest_evidence(root)["producer_diagnostics"][
                "reconstruction"
            ]["BTCUSDT"]["coverage_segments"]

        _market_coverage_check(coverage(old_root), trades, bounds)
        (candidate,) = filesystem_glob(
            tmp_path / ".run", ".r-*/**/_manifest/manifest.json"
        )
        with pytest.raises(
            PortfolioTargetBlocked, match="crypto_execution_book_coverage_incomplete"
        ):
            _market_coverage_check(coverage(candidate.parent.parent), trades, bounds)


def test_rebuild_corrupt_old_good_returns_inspection_failed(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    marker_root = raw_root / "csmar"
    filesystem_mkdir(marker_root, parents=True)
    marker = marker_root / "source.raw"
    filesystem_write_bytes(marker, b"stable")
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
    filesystem_unlink(old_root / "_manifest" / "manifest.json")

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
    assert tuple(filesystem_glob(tmp_path / ".run", ".r-*"))


def test_rebuild_compare_primary_is_preserved_with_candidate_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    marker_root = raw_root / "csmar"
    filesystem_mkdir(marker_root, parents=True)
    marker = marker_root / "source.raw"
    filesystem_write_bytes(marker, b"stable")
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
    before = _old_good_snapshot(old_root)

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

    after = _old_good_snapshot(old_root)
    assert caught.value is primary
    assert before == after
    assert calls["count"] == 2
    assert tuple(filesystem_glob(tmp_path / ".run", ".r-*"))


def test_rebuild_producer_primary_is_preserved_with_candidate_evidence(
    tmp_path: Path,
) -> None:
    class PrimaryError(Exception):
        pass

    primary = PrimaryError("producer failure")
    raw_root = tmp_path / "raw"
    marker_root = raw_root / "csmar"
    filesystem_mkdir(marker_root, parents=True)
    marker = marker_root / "source.raw"
    filesystem_write_bytes(marker, b"stable")
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
    assert tuple(filesystem_glob(tmp_path / ".run", ".r-*"))


@pytest.mark.parametrize("failure", ["unsafe", "filter", "key", "type"])
def test_rebuild_validation_failure_reports_failed_without_empty_candidate(
    tmp_path: Path,
    failure,
) -> None:
    if failure == "filter":
        from quant_research_platform.research.filtering import apply_filter_mode

        with pytest.raises(ValueError) as caught:
            apply_filter_mode(pd.DataFrame(), mode="eligible", metadata_key="mode")
        primary = caught.value
    else:
        primary = {
            "unsafe": ValueError(
                "token=private_payload C:\\private\\row.csv row=private_row"
            ),
            "key": KeyError("private_required_field"),
            "type": TypeError("private_payload"),
        }[failure]
    raw_root = tmp_path / "raw"
    marker_root = raw_root / "csmar"
    filesystem_mkdir(marker_root, parents=True)
    marker = marker_root / "source.raw"
    filesystem_write_bytes(marker, b"stable")
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
    reason = failed["target_results"][0]["reason"]
    if failure == "filter":
        assert reason == "interim_build_failed:missing_filter_columns:eligible"
    else:
        assert reason.startswith(
            f"interim_build_failed:{type(primary).__name__.lower()}:quant_research_platform."
        )
    assert "private" not in json.dumps(failed)
    assert read_dataset_manifest_evidence(old_root)["data_id"] == data_id
    assert calls["count"] == 2
    assert not tuple(filesystem_glob(tmp_path / ".run", ".r-*"))


@pytest.mark.parametrize("exception_type", (KeyboardInterrupt, SystemExit))
def test_rebuild_producer_control_base_exception_preserves_candidate(
    tmp_path: Path,
    exception_type: type[BaseException],
) -> None:
    primary = exception_type("producer control failure")
    raw_root = tmp_path / "raw"
    marker_root = raw_root / "csmar"
    filesystem_mkdir(marker_root, parents=True)
    marker = marker_root / "source.raw"
    filesystem_write_bytes(marker, b"stable")
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
    assert tuple(filesystem_glob(tmp_path / ".run", ".r-*"))


def test_rebuild_raw_mutation_before_candidate_build_blocks_without_empty_scratch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    marker_root = raw_root / "csmar"
    filesystem_mkdir(marker_root, parents=True)
    marker = marker_root / "source.raw"
    filesystem_write_bytes(marker, b"stable")
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
        filesystem_write_bytes(marker, b"changed")
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
    assert not tuple(filesystem_glob(tmp_path / ".run", ".r-*"))
    assert old_identity is not None


def test_rebuild_without_old_good_is_a_normal_immutable_build(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    marker_root = raw_root / "csmar"
    filesystem_mkdir(marker_root, parents=True)
    marker = marker_root / "source.raw"
    filesystem_write_bytes(marker, b"stable")
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
    assert not tuple(filesystem_glob(tmp_path / ".run", ".r-*"))


def test_rebuild_blocked_before_materialization_cleans_empty_candidate(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    marker_root = raw_root / "csmar"
    filesystem_mkdir(marker_root, parents=True)
    marker = marker_root / "source.raw"
    filesystem_write_bytes(marker, b"stable")
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
    assert not tuple(filesystem_glob(tmp_path / ".run", ".r-*"))


@pytest.mark.parametrize(
    "rebuild_upstream, corrupt_upstream", [(False, False), (False, True), (True, False)]
)
def test_rebuild_reuses_published_upstream_unless_explicitly_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rebuild_upstream: bool,
    corrupt_upstream: bool,
) -> None:
    raw_root = tmp_path / "raw"
    marker_root = raw_root / "csmar"
    filesystem_mkdir(marker_root, parents=True)
    marker = marker_root / "source.raw"
    filesystem_write_bytes(marker, b"stable")
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
            output_root, frame, spec=spec, instance_metadata=kwargs["instance_metadata"]
        )
        return InterimImportResult(
            artifact_roots={descendant_id: output_root},
            skipped=(),
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
        TargetSpec(
            source_id,
        ),
        TargetSpec(
            descendant_id,
            dependencies=(source_id,),
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
        start_date=a_share.DATA_START.isoformat(),
        end_date="2020-12-31",
        data_ids=(descendant_id,),
    )
    assert first["status"] == "COMPLETE"
    assert source_calls["count"] == 1
    assert descendant_calls["count"] == 1

    if corrupt_upstream:
        upstream = next(
            t for t in first["target_results"] if t["target_id"] == source_id
        )
        manifest = (
            layout.instance_root(
                layout.interim_data_root("csmar", source_id),
                upstream["instance_identity"],
            )
            / "_manifest"
            / "manifest.json"
        )
        filesystem_write_bytes(manifest, b"{}")

    failed = research.data_materialize(
        profile=profile,
        layout=layout,
        raw_root=raw_root,
        start_date=a_share.DATA_START.isoformat(),
        end_date="2020-12-31",
        data_ids=(source_id, descendant_id) if rebuild_upstream else (descendant_id,),
        rebuild=True,
    )

    results = {item["target_id"]: item for item in failed["target_results"]}
    if corrupt_upstream:
        assert failed["status"] == "BLOCKED"
        assert results[source_id]["reason"] == "persisted_instance_invalid"
        assert results[descendant_id]["status"] == "BLOCKED"
        assert source_calls["count"] == descendant_calls["count"] == 1
        assert filesystem_read_bytes(manifest) == b"{}"
        return
    if not rebuild_upstream:
        assert failed["status"] == "COMPLETE"
        assert results[source_id]["status"] == "REUSED"
        assert results[descendant_id]["status"] == "BUILT"
        assert source_calls["count"] == 1
        assert descendant_calls["count"] == 2
        assert {item["target_id"] for item in failed["rebuild_evidence"]} == {
            descendant_id
        }
        return
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
    assert tuple(filesystem_glob(tmp_path / ".run", ".r-*"))


def test_csmar_dataset_action_discovers_nested_single_root(
    tmp_path: Path,
) -> None:
    profile = research_profile()
    layout = StorageLayout(project_root=tmp_path)
    raw_root = tmp_path / "data" / "raw"
    nested_root = raw_root / "csmar" / "calendar" / "history"
    filesystem_mkdir(nested_root, parents=True)
    filesystem_write_text(
        nested_root / "native-calendar.csv",
        "Markettype,Clddt,Daywk,State\n4,2020-01-02,4,O\n",
        encoding="utf-8",
    )
    filesystem_write_text(
        nested_root / "csmar__special_treatment_changes__all_market_types__full__"
        "export-20260605.csv",
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
        plan=True,
        research_plan=research_configuration(development=("2019-01-01", "2019-12-31")),
    )
    assert research_plan["status"] == "BLOCKED"
    assert not filesystem_exists(tmp_path / "data" / "runs")
    assert all(target["decision"] != "REUSE" for target in research_plan["targets"])

    research_result = research.research_run(
        profile=profile,
        layout=layout,
        research_plan=research_configuration(development=("2019-01-01", "2019-12-31")),
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

    selected = queries.inspect(
        profile=profile,
        layout=layout,
        dataset_id="pv_all_report_artifact",
        run_id=run_id,
    )
    other_run = research.research_run(
        profile=profile,
        layout=layout,
        research_plan=research_configuration(development=("2019-01-01", "2019-12-31")),
    )
    other_report = research.report_build(
        profile=profile, layout=layout, run_id=other_run["run_id"]
    )
    other_root = layout.instance_root(
        layout.semantic_artifact_root(profile.market_id, "pv_all_report_artifact"),
        other_report["target_result"]["instance_identity"],
    )
    markdown = other_root / "report.md"
    markdown.write_bytes(markdown.read_bytes() + b"\n")
    assert (
        queries.inspect(
            profile=profile,
            layout=layout,
            dataset_id="pv_all_report_artifact",
            run_id=run_id,
        )
        == selected
    )
    with pytest.raises(ValueError, match="semantic_artifact_member_integrity_invalid"):
        queries.inspect(
            profile=profile,
            layout=layout,
            dataset_id="pv_all_report_artifact",
            run_id=other_run["run_id"],
        )

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
            research_plan=research_configuration(
                development=("2019-01-01", "2019-12-31")
            ),
        )
    assert raised.value is primary
    records = tuple(filesystem_glob(tmp_path / "data/runs/research", "*.json"))
    assert len(records) == 1
    record = read_research_run_record(records[0])
    assert record.status.value == "FAILED"
    assert record.failure_reason == "research_run_failed:runtimeerror"
    assert record.operation_evidence["reason"] == "operation_incomplete"
    assert record.operation_evidence["sample_count"] == 1
    assert "SYNTHETIC_ROW_VALUE_X" not in filesystem_read_text(
        records[0], encoding="utf-8"
    )
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
            research_plan=research_configuration(
                development=("2019-01-01", "2019-12-31")
            ),
        )
    assert secondary_raised.value is primary


def test_inspect_and_status_never_treat_a_directory_as_a_complete_target(
    tmp_path: Path,
) -> None:
    profile = research_profile()
    layout = StorageLayout(project_root=tmp_path)
    root = layout.research_dataset_root("a_share", "pv_all")
    filesystem_mkdir(root, parents=True)

    inspected = queries.inspect(
        profile=profile,
        layout=layout,
        dataset_id="pv_all",
    )
    status = queries.status(profile=profile, layout=layout)

    assert inspected["status"] == "BLOCKED"
    assert inspected["blocker"] == "dataset_instances_missing"
    assert status == {"profile_id": profile.profile_id, "runs": []}
    assert not filesystem_exists(root / "spec.json")
    assert not filesystem_exists(root / "_manifest")


def _crypto_plan_payload():
    return {
        "folds": [
            {
                "fold_id": "fold_1",
                "train": {"start": "1970-01-01T00:00:00", "end": "1970-01-01T00:00:01"},
                "validation": {
                    "start": "1970-01-01T00:00:04",
                    "end": "1970-01-01T00:00:06",
                },
            }
        ],
        "test": {"start": "1970-01-01T00:00:07", "end": "1970-01-01T00:00:08"},
        "initial_capital": 1000.0,
        "valuation_currency": "USDT",
        "execution": research_configuration("crypto", "baseline")["research_plan"][
            "execution"
        ],
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
        bind_research_configuration,
        research_profile,
    )

    profile = research_profile(CRYPTO_PROFILE_ID)
    request = dict(profile=profile, layout=StorageLayout(tmp_path), plan=True)
    with pytest.raises(TypeError, match="research_plan"):
        research.research_run(**request)

    configuration = research_configuration(
        "crypto",
        "baseline",
        development=("1970-01-01T00:00:04", "1970-01-01T00:00:06"),
        plan=_crypto_plan_payload(),
    )
    bound = bind_research_configuration(profile, configuration)
    frozen = bound.protocol(configuration["scenario_id"]).manifest()
    configuration["research_plan"]["initial_capital"] = 9999.0
    configuration["research_plan"]["exposure_limits"]["btc_usdt"][
        "max_position_quantity"
    ] = 999.0
    assert bound.protocol(configuration["scenario_id"]).manifest() == frozen
    assert "holdout_status" not in frozen
    for field in (
        "folds",
        "test",
        "initial_capital",
        "valuation_currency",
        "exposure_limits",
    ):
        bad = _crypto_plan_payload()
        del bad[field]
        configuration["research_plan"] = bad
        with pytest.raises(ValueError, match="crypto_research_plan_fields_invalid"):
            research.research_run(**request, research_plan=configuration)
    bad = _crypto_plan_payload()
    bad["folds"][0]["development_test"] = dict(bad["folds"][0]["validation"])
    configuration["research_plan"] = bad
    with pytest.raises(ValueError, match="crypto_research_plan_fold_invalid"):
        research.research_run(**request, research_plan=configuration)
    bad = _crypto_plan_payload()
    bad["raw_root"] = "private/provider"
    configuration["research_plan"] = bad
    with pytest.raises(ValueError, match="crypto_research_plan_fields_invalid"):
        research.research_run(**request, research_plan=configuration)
    bad = _crypto_plan_payload()
    bad["folds"][0]["validation"]["end"] = "1970-01-01T00:00:04.200000"
    configuration["research_plan"] = bad
    configuration["windows"]["development"] = dict(bad["folds"][0]["validation"])
    with pytest.raises(ValueError, match="crypto_research_plan_label_window_too_short"):
        research.research_run(**request, research_plan=configuration)
    configuration["research_plan"] = _crypto_plan_payload()
    configuration["windows"]["development"] = dict(frozen["development_window"])
    with pytest.raises(PermissionError, match="holdout_frozen"):
        research.research_run(**request, research_plan=configuration, holdout=True)
    valid = _crypto_plan_payload()
    for fold in valid["folds"]:
        for role in ("train", "validation"):
            fold[role] = {key: value + "Z" for key, value in fold[role].items()}
    valid["test"] = {key: value + "Z" for key, value in valid["test"].items()}
    configuration["research_plan"] = valid
    plan_file = tmp_path / "plan.json"
    filesystem_write_text(plan_file, json.dumps(configuration), encoding="utf-8")
    args = ["research", "run", "--project-root", str(tmp_path), "--plan"]
    missing = CliRunner().invoke(app, args)
    assert missing.exit_code == 2 and "research-plan" in missing.output
    result = CliRunner().invoke(app, [*args, "--research-plan", str(plan_file)])
    assert result.exit_code == 3, result.output
    published = json.loads(result.output)
    assert published["protocol"]["research_plan"] == _crypto_plan_payload()
    assert "holdout_status" not in published["protocol"]
    assert published["status"] == "BLOCKED"
    assert "research_plan" not in profile.protocol().manifest()


@pytest.mark.parametrize(
    "kind", (ProducerKind.INTERIM, ProducerKind.PROCESSED, ProducerKind.DATASET)
)
@pytest.mark.parametrize("failure", ("duplicate", "interrupt", "memory", "io"))
@pytest.mark.parametrize("phase", ("build", "reuse"))
def test_target_local_validation_preserves_independent_production(
    tmp_path, monkeypatch, kind, failure, phase
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

    priming = False

    def frame(name):
        calls.append(name)
        if name == a and primary is not None and not priming:
            raise primary
        values = {
            asset_column: ["asset", "asset"],
            "timestamp": pd.to_datetime(
                [
                    "2020-01-02",
                    "2020-01-02" if name == a and not priming else "2020-01-03",
                ]
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
            root, frame(name), spec=spec, instance_metadata=kwargs["instance_metadata"]
        )
        return root

    def interim(**kwargs):
        name = kwargs["data_ids"][0]
        return InterimImportResult(
            artifact_roots={name: persisted(name, kwargs)},
            skipped=(),
        )

    @contextmanager
    def processed(spec, **kwargs):

        def frames():
            yield frame(spec.data_id)

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
            TargetSpec(
                a,
            ),
            TargetSpec(
                b,
                dependencies=(a,),
            ),
            TargetSpec(
                c,
            ),
        ),
    )
    raw = tmp_path / "raw"
    filesystem_mkdir(raw / "csmar", parents=True)
    kwargs = dict(
        profile=profile,
        layout=StorageLayout(tmp_path),
        raw_root=raw,
        start_date="2020-01-01",
        end_date="2020-01-31",
        data_ids=(b, c),
    )
    if phase == "reuse":
        priming = True
        initial = research.data_materialize(**{**kwargs, "data_ids": (a,)})
        assert initial["status"] == "COMPLETE"
        priming = False
        calls.clear()
        identity = initial["target_results"][0]["instance_identity"]
        root = research._target_instance_root(profile, kwargs["layout"], a, identity)
        if primary is None:
            part = next(filesystem_glob(root, "**/*.parquet"))
            content = bytearray(filesystem_read_bytes(part))
            content[4] ^= 1
            filesystem_write_bytes(part, bytes(content))
        planned = research.data_materialize(**kwargs, plan=True)
        planned_target = next(
            item for item in planned["targets"] if item["target_id"] == a
        )
        assert planned_target["decision"] == ("BLOCKED" if primary is None else "REUSE")
        if primary is None:
            assert planned_target["blocker"] == "persisted_instance_invalid"
        actual_inspect = research.inspect_dataset_session

        def inspect(*args, **inner):
            calls.append(a)
            if primary is not None:
                raise primary
            return actual_inspect(*args, **inner)

        monkeypatch.setattr(research, "inspect_dataset_session", inspect)
    if primary is not None:
        with pytest.raises(type(primary)) as raised:
            research.data_materialize(**kwargs)
        assert raised.value is primary
        assert calls == [a]
        return
    result = research.data_materialize(**kwargs)
    results = {item["target_id"]: item for item in result["target_results"]}
    assert result["status"] == ("FAILED" if phase == "build" else "BLOCKED")
    assert calls == ([a, c] if phase == "build" else [c])
    assert results[a]["status"] == ("FAILED" if phase == "build" else "BLOCKED")
    if phase == "build":
        assert "duplicate_dataset_primary_key" in results[a]["reason"]
    else:
        assert results[a]["reason"] == "persisted_instance_invalid"
    assert results[b]["status"] == "BLOCKED"
    assert results[b]["reason"] == f"dependency_blocked: {a}"
    assert results[c]["status"] == "BUILT"


@pytest.mark.parametrize("failure", ("missing", "memory", "io", "interrupt"))
@pytest.mark.parametrize("fail_at", (1, 2))
def test_source_hash_failure_keeps_run_failure_semantics(
    tmp_path, monkeypatch, failure, fail_at
):
    import errno

    from quant_research_platform.storage import identity

    raw = tmp_path / "raw"
    marker = raw / "csmar" / "source.raw"
    filesystem_mkdir(marker.parent, parents=True)
    filesystem_write_bytes(marker, b"synthetic")
    profile, _, _ = _calendar_rebuild_profile("value", marker=marker)
    profile = replace(
        profile,
        producer_bindings={
            **research_profile().producer_bindings,
            **profile.producer_bindings,
        },
    )
    primary = {
        "missing": FileNotFoundError(errno.ENOENT, "missing"),
        "memory": MemoryError("primary"),
        "io": OSError(errno.EIO, "primary"),
        "interrupt": KeyboardInterrupt("primary"),
    }[failure]
    original = identity._hash_file_content
    calls = 0

    def fail_hash(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls >= fail_at:
            raise primary
        return original(*args, **kwargs)

    monkeypatch.setattr(identity, "_hash_file_content", fail_hash)
    kwargs = dict(
        profile=profile,
        layout=StorageLayout(tmp_path),
        raw_root=raw,
        research_plan=research_configuration(development=("2019-01-01", "2019-12-31")),
    )
    if failure == "missing":
        result = research.research_run(**kwargs)
        calendar = next(
            item
            for item in result["target_results"]
            if item["target_id"] == "trading_calendar"
        )
        assert calendar["status"] == "BLOCKED"
        assert calendar["reason"] == "source_identity_invalid"
    else:
        with pytest.raises(type(primary)) as caught:
            research.research_run(**kwargs)
        assert caught.value is primary
    records = tuple(filesystem_glob(tmp_path / "data/runs/research", "*.json"))
    assert len(records) == 1
    record = read_research_run_record(records[0])
    if failure != "missing":
        assert record.status.value == "FAILED"
        assert (
            record.failure_reason
            == f"research_run_failed:{type(primary).__name__.lower()}"
        )


def test_partial_development_blocks_global_selection_without_blocking_shared_model(
    tmp_path: Path,
) -> None:
    from quant_research_platform.profiles.research import bind_research_configuration

    profile = bind_research_configuration(
        research_profile(), research_configuration(name="core6")
    )
    profile = replace(
        profile,
        producer_bindings={
            key: replace(binding, source_identity=lambda *_args: "a" * 64)
            if binding.source_identity is not None
            else binding
            for key, binding in profile.producer_bindings.items()
        },
    )
    protocol = profile.protocol("pv_all")
    plan = research.plan_targets(
        profile=profile,
        action="research run",
        terminal_targets=protocol.terminal_targets("research run"),
        scenario_id="pv_all",
        layout=StorageLayout(project_root=tmp_path),
        research_window={
            "start": "2019-01-01",
            "end": "2019-06-30",
            "access": "development",
        },
    )
    targets = {item["target_id"]: item for item in plan["targets"]}
    assert targets["pv_all_model_selection_refit"]["decision"] == "BUILD"
    assert targets["pv_all_quarterly_persisted_predictions"]["decision"] == "BUILD"
    assert targets["pv_all_global_persisted_predictions"]["decision"] == "BLOCKED"
    assert targets["pv_all_global_persisted_predictions"]["blocker"] == (
        "global_selection_requires_complete_development"
    )


@pytest.mark.parametrize(
    ("start", "prediction_start"),
    (("2019-01-01", "2019-01-01"), ("2022-01-01", "2021-01-01")),
)
@pytest.mark.parametrize("reuse", (False, True), ids=("build", "reuse"))
def test_formal_prediction_dispatch_preserves_planned_window_and_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start: str,
    prediction_start: str,
    reuse: bool,
) -> None:
    from quant_research_platform.contracts.targets import TargetResult
    from quant_research_platform.profiles.research import bind_research_configuration

    class ArtifactReadBoundary(BaseException):
        pass

    base = research_profile()
    base = replace(
        base,
        producer_bindings={
            key: replace(binding, source_identity=lambda *_args: "a" * 64)
            if binding.source_identity is not None
            else binding
            for key, binding in base.producer_bindings.items()
        },
    )
    config = research_configuration(development=(start, "2023-12-31"))
    layout = StorageLayout(project_root=tmp_path)
    public_plan = research.research_run(
        profile=base,
        layout=layout,
        research_plan=config,
        raw_root=tmp_path / "raw",
        plan=True,
    )
    assert public_plan["status"] == "COMPLETE"
    public_targets = {item["target_id"]: item for item in public_plan["targets"]}
    bound = bind_research_configuration(base, config)
    protocol = bound.protocol("pv_all")
    prediction_ids = set(protocol.prediction_target_ids)
    expected_window = TimeWindow(prediction_start, "2023-12-31")
    active_dispatch: list[str] = []
    reached: set[str] = set()
    original_build = research._build_target
    original_reuse = research._reuse_target
    original_semantic_build = research._build_semantic_target
    original_window = research._semantic_target_time_window
    original_matching = research._matching_instance

    def planned_window_only(*args, **kwargs):
        assert not active_dispatch, "execution transformed an already planned window"
        return original_window(*args, **kwargs)

    def matching_prediction(**kwargs):
        if reuse and kwargs["spec"].target_id in prediction_ids:
            return kwargs["requested_identity"], None
        return original_matching(**kwargs)

    def stop_at_model_reader(*_args, **_kwargs):
        reached.add(active_dispatch[0])
        raise ArtifactReadBoundary

    def stop_at_reuse_reader(_profile, _protocol, spec, root, identity, **_kwargs):
        assert spec.target_id == active_dispatch[0]
        assert (
            root.name
            == identity
            == public_targets[spec.target_id]["requested_identity"]
        )
        reached.add(spec.target_id)
        raise ArtifactReadBoundary

    def observe_semantic_build(**kwargs):
        assert kwargs["time_window"] == expected_window
        assert kwargs["research_window"] == {
            "start": prediction_start,
            "end": "2023-12-31",
            "access": "development",
        }
        return original_semantic_build(**kwargs)

    def dispatch_prediction(original, status, kwargs):
        target_id = kwargs["spec"].target_id
        assert kwargs["time_window"] == expected_window
        assert (
            kwargs["planned"].requested_identity
            == public_targets[target_id]["requested_identity"]
        )
        active_dispatch.append(target_id)
        try:
            with pytest.raises(ArtifactReadBoundary):
                original(**kwargs)
        finally:
            active_dispatch.pop()
        return TargetResult(
            target_id,
            status,
            instance_identity=kwargs["planned"].requested_identity,
        )

    def build_prerequisite_or_prediction(**kwargs):
        target_id = kwargs["spec"].target_id
        if target_id in prediction_ids:
            assert not reuse
            assert kwargs["access"] == "development"
            return dispatch_prediction(original_build, TargetStatus.BUILT, kwargs)
        # This control-plane fixture stops before actual model or dataset IO.
        return TargetResult(
            target_id,
            TargetStatus.BUILT,
            instance_identity=kwargs["planned"].requested_identity,
        )

    def reuse_prediction(**kwargs):
        assert reuse and kwargs["spec"].target_id in prediction_ids
        return dispatch_prediction(original_reuse, TargetStatus.REUSED, kwargs)

    monkeypatch.setattr(research, "_semantic_target_time_window", planned_window_only)
    monkeypatch.setattr(research, "_matching_instance", matching_prediction)
    monkeypatch.setattr(research, "_build_target", build_prerequisite_or_prediction)
    monkeypatch.setattr(research, "_reuse_target", reuse_prediction)
    monkeypatch.setattr(research, "_build_semantic_target", observe_semantic_build)
    monkeypatch.setattr(research, "read_quarterly_model_artifact", stop_at_model_reader)
    monkeypatch.setattr(
        research, "_validate_semantic_target_root", stop_at_reuse_reader
    )
    result = research._execute_action(
        profile=bound,
        protocol=protocol,
        action="research run",
        terminal_targets=tuple(protocol.terminal_targets("research run")),
        layout=layout,
        raw_root=tmp_path / "raw",
        research_window={"start": start, "end": "2023-12-31", "access": "development"},
        run_id="run_formal_prediction_window",
    )
    assert result["status"] == "COMPLETE"
    assert reached == prediction_ids
    assert not tuple(filesystem_glob(tmp_path, "**/manifest.json"))


def test_prediction_lineage_requires_exact_model_dependency_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from quant_research_platform.contracts.models import (
        A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
    )

    model_identity = "a" * 64
    profile = SimpleNamespace(
        producer_bindings={
            "prediction": ProducerBinding(ProducerKind.PREDICTION, lambda: None),
            "model": ProducerBinding(ProducerKind.MODEL, lambda: None),
        }
    )
    target = TargetSpec(
        target_id="prediction",
        dependencies=("model",),
    )
    protocol = SimpleNamespace(
        protocol_id="a_share_low_frequency",
        selection_group_for_target=lambda _target: "synthetic",
        access_window=TimeWindow("2024-01-01", "2024-01-02"),
    )
    layout = StorageLayout(project_root=tmp_path)
    envelope = {
        "logical_artifact_id": "prediction",
        "upstream_identities": {"model": model_identity},
    }
    manifest = {
        "metadata": {
            "test_start": "2024-01-01",
            "test_end": "2024-01-02",
            "model_metadata": {
                "model_artifact_id": "model",
                "model_artifact_identity": "b" * 64,
                "model_artifact_schema": A_SHARE_QUARTERLY_MODEL_ARTIFACT_SCHEMA_ID,
            },
        }
    }

    monkeypatch.setattr(
        research,
        "read_semantic_artifact_record",
        lambda *_args, **_kwargs: (envelope, {}),
    )

    def bundle(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return manifest, None

    monkeypatch.setattr(research, "_validate_prediction_output_bundle", bundle)
    monkeypatch.setattr(
        research, "_target_instance_root", lambda *_args: tmp_path / "model"
    )
    monkeypatch.setattr(
        research,
        "read_quarterly_model_artifact",
        lambda *_args, **_kwargs: SimpleNamespace(
            prediction_lineage=lambda *_args, **_kwargs: manifest["metadata"][
                "model_metadata"
            ]
        ),
    )
    with pytest.raises(
        ValueError,
        match="^prediction_inner_artifact_invalid:prediction_model_lineage_mismatch$",
    ):
        research._validate_semantic_target_root(
            profile,
            protocol,
            target,
            Path("prediction-root"),
            "c" * 64,
            trial_layout=layout,
        )
    manifest["metadata"]["model_metadata"]["model_artifact_identity"] = model_identity
    research._validate_semantic_target_root(
        profile,
        protocol,
        target,
        Path("prediction-root"),
        "c" * 64,
        trial_layout=layout,
    )


def test_model_selection_rejects_protocol_family_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    model_target = "model-target"
    expected_model_id = "scenario_linear_regression"
    profile = SimpleNamespace(
        producer_bindings={
            model_target: ProducerBinding(ProducerKind.MODEL, lambda: None),
        },
        models=SimpleNamespace(
            get=lambda _model_id: SimpleNamespace(input_columns=()),
        ),
    )
    protocol = SimpleNamespace(
        protocol_id="a_share_low_frequency",
        scenario_id="scenario",
        manifest=lambda: {"selection_tie_break": ["linear_regression"]},
        model_ids_for_target=lambda _target_id: (expected_model_id,),
    )
    target = TargetSpec(
        model_target,
    )
    selection = {
        "candidate_family_ids": [expected_model_id],
        "tie_break_order": [expected_model_id],
        "selected_family_id": "different_family",
    }
    persisted = SimpleNamespace(
        envelope={"logical_artifact_id": model_target},
        selections={"global": {"final": selection}},
        candidate_models=(
            {
                "quarter_id": "2019Q1",
                "family_id": "different_family",
                "model_metadata": {
                    "model_algorithm": "linear_regression",
                    "input_columns": [],
                },
            },
        ),
        final_models=(),
    )
    monkeypatch.setattr(
        research,
        "read_quarterly_model_artifact",
        lambda *_args, **_kwargs: persisted,
    )
    with pytest.raises(ValueError, match="model_selection_family_mismatch"):
        research._validate_semantic_target_root(
            profile, protocol, target, Path("model-root"), "a" * 64
        )


def test_evaluation_result_is_owned_before_atomic_publication_callback(
    tmp_path, monkeypatch
):
    from quant_research_platform.profiles.research import bind_research_configuration
    from quant_research_platform.research.evaluations import runner
    from quant_research_platform.storage.errors import OutputAlreadyExistsError

    config = research_configuration()
    config.update(
        stage="evaluation",
        models=[],
        portfolios=[],
        comparison_model_id=None,
        primary_portfolio_id=None,
    )
    profile = bind_research_configuration(research_profile(), config)
    protocol = profile.protocol(config["scenario_id"])
    target = next(
        item
        for item in protocol.target_specs("research run")
        if item.target_id.endswith("_evaluation")
    )
    layout = StorageLayout(tmp_path)
    window = TimeWindow(**config["windows"]["development"])
    evaluation_spec = profile.evaluations.get(target.target_id)
    dataset_spec = profile.research_datasets.get(evaluation_spec.research_dataset_id)
    input_frame = pd.DataFrame(
        {
            column: [
                "a_share"
                if column == "market_id"
                else "asset"
                if column == "asset_id"
                else pd.Timestamp(window.start)
                if role is ColumnRole.TIMESTAMP
                else True
                if role in (ColumnRole.MASK, ColumnRole.UNIVERSE)
                else 1.0
            ]
            for column, role in dataset_spec.column_roles.items()
        }
    )
    write_dataset(
        layout.research_dataset_root("a_share", dataset_spec.data_id),
        input_frame,
        spec=dataset_spec,
    )
    window_payload = {**config["windows"]["development"], "access": "development"}
    software = research.production_software_identity()
    envelope = research._target_envelope_for_spec(
        profile=profile,
        protocol=protocol,
        target_spec=target,
        data_spec=None,
        research_window=window_payload,
        upstream_identities={},
        software_identity=software,
        raw_content_identity=None,
    )
    existing = layout.instance_root(
        research._target_logical_root(profile, layout, target.target_id),
        envelope.instance_identity,
    )
    filesystem_mkdir(existing, parents=True)
    marker = existing / "existing.txt"
    filesystem_write_bytes(marker, b"unchanged")
    actual_write = research.write_semantic_artifact
    observed = []
    created, consumed, closed, callbacks = [], [], [], []
    actual_chunks = runner.iter_research_dataset_chunks
    actual_outputs = research.write_evaluation_outputs_from_chunks

    class ObservedIterator:
        def __init__(self, source):
            self.source = source

        def __iter__(self):
            return self

        def __next__(self):
            consumed.append(True)
            return next(self.source)

        def close(self):
            self.source.close()
            closed.append(True)

    def chunks(*args, **kwargs):
        created.append(True)
        return ObservedIterator(actual_chunks(*args, **kwargs))

    def outputs(*args, **kwargs):
        callbacks.append(True)
        return actual_outputs(*args, **kwargs)

    def publish(*args, **kwargs):
        try:
            return actual_write(*args, **kwargs)
        except ValueError as error:
            observed.append(error)
            raise

    monkeypatch.setattr(research, "write_semantic_artifact", publish)
    monkeypatch.setattr(runner, "iter_research_dataset_chunks", chunks)
    monkeypatch.setattr(research, "write_evaluation_outputs_from_chunks", outputs)
    with pytest.raises(
        ValueError, match="^evaluation_write_failed:output_already_exists$"
    ) as raised:
        research._build_semantic_target(
            profile=profile,
            protocol=protocol,
            layout=layout,
            spec=target,
            planned=TargetPlan(
                target_id=target.target_id,
                upstream_target_ids=tuple(target.dependencies),
                decision="BUILD",
                **{
                    "requested_identity": envelope.instance_identity,
                    "upstream_identities": {},
                },
            ),
            target_layout=layout,
            time_window=window,
            research_window=window_payload,
            software_identity=software,
            raw_root=tmp_path / "raw",
        )
    assert len(observed) == 1 and raised.value.__cause__ is observed[0]
    assert isinstance(observed[0], OutputAlreadyExistsError)
    assert created == closed == [True]
    assert callbacks == consumed == []
    assert filesystem_read_bytes(marker) == b"unchanged"


@pytest.mark.parametrize(
    "start,end,expected_start,expected_end",
    [
        ("2020-01-01", None, "2020-01-01", "2023-12-31"),
        (None, "2020-12-31", "2019-01-01", "2020-12-31"),
        ("2020-01-01", "2020-12-31", "2020-01-01", "2020-12-31"),
    ],
)
def test_materialize_single_endpoint_matches_public_plan_and_execute(
    tmp_path, start, end, expected_start, expected_end
):
    from typer.testing import CliRunner

    from quant_research_platform.interfaces.cli.commands import app
    from quant_research_platform.profiles.research import bind_research_configuration
    from quant_research_platform.storage.datasets.reading import read_dataset
    from tests.sources.reference.test_interim import _write_reference_root

    profile = bind_research_configuration(research_profile(), research_configuration())
    target = "a_share_fee_schedule"
    raw = _write_reference_root(tmp_path / "raw")
    expected = {"start": expected_start, "end": expected_end, "access": "development"}
    api_layout = StorageLayout(tmp_path / "api")
    cli_layout = StorageLayout(tmp_path / "cli")
    arguments = dict(
        profile=profile,
        layout=api_layout,
        raw_root=raw,
        start_date=start,
        end_date=end,
        data_ids=(target,),
    )
    native_frames = []
    for plan in (True, False):
        result = research.data_materialize(**arguments, plan=plan)
        assert result["status"] == "COMPLETE"
        if plan:
            assert result["research_window"] == expected
        else:
            (built,) = result["target_results"]
            assert built["status"] == "BUILT"
            native = api_layout.instance_root(
                api_layout.interim_data_root("reference", target),
                built["instance_identity"],
            )
            window = read_dataset_manifest_evidence(native)["common_manifest"][
                "research_window"
            ]
            assert window == {**expected, "access": "not_applicable"}
            native_frames.append(read_dataset(native))
        command = [
            "data",
            "materialize",
            "--project-root",
            str(cli_layout.project_root),
            "--raw-root",
            str(raw),
            "--data-id",
            target,
        ]
        for option, value in (("--start-date", start), ("--end-date", end)):
            if value is not None:
                command.extend((option, value))
        if plan:
            command.append("--plan")
        response = CliRunner().invoke(app, command)
        assert response.exit_code == 0, response.output
        payload = json.loads(response.output)
        assert payload["status"] == "COMPLETE"
        if plan:
            assert payload["research_window"] == expected
        else:
            (built,) = payload["target_results"]
            assert built["status"] == "BUILT"
            native = cli_layout.instance_root(
                cli_layout.interim_data_root("reference", target),
                built["instance_identity"],
            )
            window = read_dataset_manifest_evidence(native)["common_manifest"][
                "research_window"
            ]
            assert window == {**expected, "access": "not_applicable"}
            native_frames.append(read_dataset(native))
    pd.testing.assert_frame_equal(*native_frames)


@pytest.mark.parametrize("start,end", [("invalid", None), (None, "invalid")])
def test_materialize_rejects_invalid_single_endpoint_before_sources(
    tmp_path, start, end
):
    for plan in (True, False):
        with pytest.raises(ValueError):
            research.data_materialize(
                profile=research_profile(),
                layout=StorageLayout(tmp_path),
                raw_root=tmp_path / "missing-raw",
                start_date=start,
                end_date=end,
                data_ids=(research_profile().interim.list()[0].data_id,),
                plan=plan,
            )
    assert not (tmp_path / "data").exists()


def test_stage_choice_validates_published_source_and_never_uses_holdout(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace

    from quant_research_platform.artifacts.evidence import write_json_evidence
    from quant_research_platform.artifacts.research import write_research_run_record
    from quant_research_platform.contracts.business_manifest import (
        build_business_manifest_envelope,
    )
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.contracts.targets import (
        ResearchRunRecord,
        ResearchRunTargetResult,
        RunStatus,
        unmeasured_operation_evidence,
    )
    from quant_research_platform.storage.semantic_artifacts import (
        write_semantic_artifact,
    )

    layout = StorageLayout(project_root=tmp_path)
    windows = {
        "development": {"start": "2019-01-01", "end": "2023-12-31"},
        "holdout": {"start": "2024-01-01", "end": "2025-12-31"},
    }
    target_id = "pv_all_factor_statistics"
    envelope = build_business_manifest_envelope(
        logical_artifact_id=target_id,
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={**windows["development"], "access": "development"},
        upstream_identities={},
        semantic_metadata={},
    )
    logical = layout.semantic_artifact_root(MarketId("a_share"), target_id)
    root = write_semantic_artifact(
        logical,
        envelope,
        lambda staging: write_json_evidence(
            staging / "value.json", {"schema_id": "stage-test", "status": "DEFINED"}
        ),
    )
    reference = ResearchRunTargetResult(
        target_id,
        ProducerKind.STATISTIC,
        TargetStatus.BUILT,
        instance_identity=envelope.instance_identity,
        schema_identity=envelope.schema_identity,
        artifact_root=logical.relative_to(tmp_path).as_posix(),
    )
    source_protocol = {
        "scenario_id": "pv_all",
        "access_window": "development",
        "research_configuration": {"stage": "evaluation", "windows": windows},
    }
    source = ResearchRunRecord(
        run_id="run_source",
        profile_id="a_share_csmar_choice",
        scenario_id="pv_all",
        protocol=source_protocol,
        protocol_identity=canonical_json_hash(source_protocol),
        research_window={**windows["development"], "access": "development"},
        software_identity="c" * 64,
        terminal_targets=(target_id,),
        target_results=(reference,),
        status=RunStatus.COMPLETE,
        operation_evidence=unmeasured_operation_evidence("fixture"),
    )
    write_research_run_record(layout.research_run_path(source.run_id), source)
    choice = {
        "run_id": source.run_id,
        "target_id": target_id,
        "instance_identity": envelope.instance_identity,
    }
    config = {
        "stage": "models",
        "windows": windows,
        "stage_selection": {
            "selected_by": "agent",
            "reason": "Compare development evidence",
            "sources": [choice],
        },
    }
    profile = SimpleNamespace(profile_id=source.profile_id)
    protocol = SimpleNamespace(scenario_id="pv_all", configuration=lambda: config)
    research._validate_stage_selection(profile, layout, protocol)
    choice["instance_identity"] = "d" * 64
    with pytest.raises(ValueError, match="research_stage_source_identity_mismatch"):
        research._validate_stage_selection(profile, layout, protocol)
    choice["instance_identity"] = envelope.instance_identity
    config["stage"] = "evaluation"
    with pytest.raises(ValueError, match="research_stage_source_ineligible"):
        research._validate_stage_selection(profile, layout, protocol)
    config["stage"] = "models"
    holdout_protocol = {**source_protocol, "access_window": "holdout"}
    holdout = replace(
        source,
        run_id="run_holdout_choice",
        protocol=holdout_protocol,
        protocol_identity=canonical_json_hash(holdout_protocol),
        research_window={**windows["holdout"], "access": "holdout"},
    )
    write_research_run_record(layout.research_run_path(holdout.run_id), holdout)
    choice["run_id"] = holdout.run_id
    with pytest.raises(ValueError, match="research_stage_source_ineligible"):
        research._validate_stage_selection(profile, layout, protocol)
    choice["run_id"] = source.run_id
    # A valid run whose artifact was modified cannot be chosen by its old identity.
    (root / "value.json").write_text('{"status":"CHANGED"}')
    with pytest.raises(ValueError):
        research._validate_stage_selection(profile, layout, protocol)


def test_stage_selection_prose_does_not_change_computation_contracts():
    from copy import deepcopy

    from quant_research_platform.profiles.research import bind_research_configuration

    config = research_configuration()
    config["stage_selection"] = {
        "selected_by": "agent",
        "reason": "first explanation",
        "sources": [
            {
                "run_id": "run_source",
                "target_id": "pv_all_factor_statistics",
                "instance_identity": "a" * 64,
            }
        ],
    }
    changed = deepcopy(config)
    changed["stage_selection"].update(selected_by="user", reason="reworded explanation")

    def projection(value):
        profile = bind_research_configuration(research_profile(), value)
        protocol = profile.protocol(value["scenario_id"])
        return protocol, {
            target.target_id: research._semantic_target_payload(
                profile=profile,
                protocol=protocol,
                target_spec=target,
                binding=profile.producer_bindings[target.target_id],
            )
            for target in protocol.target_specs("research run")
            if profile.producer_bindings[target.target_id].kind
            is not ProducerKind.REPORT
        }

    original_protocol, original = projection(config)
    changed_protocol, reworded = projection(changed)
    assert original == reworded
    assert original_protocol.manifest() != changed_protocol.manifest()
    assert (
        a_share._inference_blocker(next(iter(original.values()))["protocol"])
        == "development_stage_selection_exploratory"
    )
    changed.pop("stage_selection")
    _, unselected = projection(changed)
    assert original != unselected
    changed = deepcopy(config)
    changed["models"][0]["input_factor_columns"] = list(
        reversed(changed["models"][0]["input_factor_columns"])
    )
    if (
        changed["models"][0]["input_factor_columns"]
        != config["models"][0]["input_factor_columns"]
    ):
        assert projection(changed)[1] != original


def test_planner_reads_frozen_manifest_once_for_all_targets(tmp_path, monkeypatch):
    profile = research_profile()
    protocol_type = type(profile.protocol("pv_all"))
    original = protocol_type.manifest
    calls = []

    def observed(self):
        calls.append(True)
        return original(self)

    arguments = dict(
        profile=profile,
        action="dataset build",
        terminal_targets=("pv_all",),
        layout=StorageLayout(tmp_path),
    )
    expected = research.plan_targets(**arguments)
    monkeypatch.setattr(protocol_type, "manifest", observed)
    actual = research.plan_targets(**arguments)
    assert actual == expected
    assert len(actual["targets"]) > 2
    # One frozen-state read and one final public manifest.
    assert len(calls) == 2
