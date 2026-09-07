from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs import validate_output_bundle
from quant_research_platform.artifacts.research import read_research_run_record
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
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
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.portfolios import CONTINUOUS_EXECUTION_COLUMNS
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.targets import (
    ProducerBinding,
    ProducerKind,
    research_run_record_from_payload,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.profiles.research import (
    A_SHARE_PROFILE_ID,
    ResearchProfile,
    research_profile,
)
from quant_research_platform.reports.a_share import REQUIRED_REPORT_SECTIONS
from quant_research_platform.research.models.dependencies import (
    preflight_model_dependency,
)
from quant_research_platform.research.models.persisted import (
    write_quarterly_model_artifact,
)
from quant_research_platform.research.models.production import (
    _fit_selected_model_result,
)
from quant_research_platform.research.models.rank_ensemble import RankEnsemblePredictor
from quant_research_platform.research.protocols import a_share
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.sources.csmar.interim import InterimImportResult
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import (
    content_hash_for_files,
    native_filesystem_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research
from tests.helpers.portfolio_ledger import session_frame
from tests.research.workflow_test_support import _quarterly_result, _trial_reference


def _interim_specs() -> InterimDataCatalog:
    calendar = InterimDataSpec(
        data_id="trading_calendar",
        source_id="csmar",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.SOURCE_MARKET_CALENDAR_DATE,
        column_roles={
            "timestamp": ColumnRole.TIMESTAMP,
            "calendar_date": ColumnRole.SOURCE,
            "source_market_type": ColumnRole.KEY,
            "is_trading_day": ColumnRole.CONTEXT,
        },
        primary_key=("timestamp", "source_market_type"),
        partition_columns=("year",),
    )
    fee = InterimDataSpec(
        data_id="a_share_fee_schedule",
        source_id="reference",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.MARKET_TIMESTAMP,
        column_roles={
            "fee_code": ColumnRole.KEY,
            "effective_date": ColumnRole.TIMESTAMP,
            "end_date": ColumnRole.KEY,
            "market": ColumnRole.KEY,
            "side": ColumnRole.KEY,
            "basis": ColumnRole.KEY,
            "rate": ColumnRole.CONTEXT,
            "included_in_ledger": ColumnRole.CONTEXT,
            "included_in": ColumnRole.CONTEXT,
        },
        primary_key=(
            "fee_code",
            "effective_date",
            "end_date",
            "market",
            "side",
            "basis",
        ),
        partition_columns=("year",),
    )
    return InterimDataCatalog(
        (calendar, fee, research_profile().interim.get("index_market"))
    )


def _processed_specs() -> ProcessedDataCatalog:
    common = {
        "market_id": ColumnRole.KEY,
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
    }
    panel = ProcessedDataSpec(
        data_id="market_panel_1d",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            **common,
            "adj_open": ColumnRole.CONTEXT,
            "adj_close": ColumnRole.CONTEXT,
            "amount": ColumnRole.CONTEXT,
            "market_type": ColumnRole.CONTEXT,
            "period_return": ColumnRole.CONTEXT,
            "float_market_cap": ColumnRole.CONTEXT,
            "industry_level1_code": ColumnRole.KEY,
            "is_nonfinancial_industry": ColumnRole.CONTEXT,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
        market_id="a_share",
        kind=ProcessedDataKind.PANEL,
    )
    execution = ProcessedDataSpec(
        data_id="execution_mask",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={
            **common,
            "can_enter": ColumnRole.MASK,
            "can_exit": ColumnRole.MASK,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
        market_id="a_share",
        kind=ProcessedDataKind.ANNOTATION,
    )
    benchmark = ProcessedDataSpec(
        data_id="benchmark_weights",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.MARKET_TIMESTAMP,
        column_roles={
            "market_id": ColumnRole.KEY,
            "benchmark_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "weight": ColumnRole.CONTEXT,
        },
        primary_key=("market_id", "benchmark_id", "asset_id", "timestamp"),
        partition_columns=("year",),
        market_id="a_share",
        kind=ProcessedDataKind.ANNOTATION,
    )
    risk_context = tuple(
        ProcessedDataSpec(
            data_id=data_id,
            frequency=DataFrequency.DAILY,
            grain=DataGrain.ASSET_TIMESTAMP,
            column_roles={
                **common,
                **{column: ColumnRole.FACTOR for column in columns},
            },
            primary_key=("market_id", "asset_id", "timestamp"),
            partition_columns=("year",),
            market_id="a_share",
            kind=ProcessedDataKind.FACTOR,
        )
        for data_id, columns in (
            (
                "price_volume_facts",
                ("log_float_market_cap", "momentum_252_20", "amihud_60d"),
            ),
            ("fundamental_factors", ("earnings_yield",)),
        )
    )
    return ProcessedDataCatalog((panel, execution, benchmark, *risk_context))


def _dataset_spec() -> ResearchDatasetSpec:
    statistical_factors = tuple(factor for factor in a_share.PV_FULL if factor != "x")
    return ResearchDatasetSpec(
        data_id="pv_all",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.RESEARCH_ROW,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "x": ColumnRole.FACTOR,
            **{factor: ColumnRole.FACTOR for factor in statistical_factors},
            "label": ColumnRole.LABEL,
            "forward_return_20d": ColumnRole.LABEL,
            "forward_return_20d_available_at": ColumnRole.LABEL,
            "eligible": ColumnRole.MASK,
            "float_market_cap": ColumnRole.CONTEXT,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
        market_id="a_share",
        input_data_ids=("market_panel_1d",),
        base_input_data_id="market_panel_1d",
        factor_columns=("x", *statistical_factors),
        label_columns=(
            "label",
            "forward_return_20d",
            "forward_return_20d_available_at",
        ),
        filter_columns=("eligible",),
        context_columns=("float_market_cap",),
    )


def _frames() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2018-12-03", "2019-01-15")
    assets = ("a", "b")
    panel = pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "asset_id": asset,
                "timestamp": timestamp,
                "adj_open": 100.0,
                "adj_close": 100.0,
                "amount": 10_000.0,
                "market_type": 1,
            }
            for asset in assets
            for timestamp in dates
        ]
    )
    execution = pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "asset_id": asset,
                "timestamp": timestamp,
                "can_enter": True,
                "can_exit": True,
            }
            for asset in assets
            for timestamp in dates
        ]
    )
    dataset = pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "asset_id": asset,
                "timestamp": timestamp,
                "x": float(asset == "b"),
                **{
                    factor: float(asset == "b")
                    for factor in a_share.PV_FULL
                    if factor != "x"
                },
                "label": float(asset == "b") * 0.01,
                "forward_return_20d": float(asset == "b") * 0.02,
                "forward_return_20d_available_at": timestamp,
                "eligible": True,
                "float_market_cap": 1_000.0 if asset == "a" else 2_000.0,
            }
            for asset in assets
            for timestamp in dates
        ]
    )
    panel["period_return"] = 0.001
    panel["float_market_cap"] = panel["asset_id"].map({"a": 1_000.0, "b": 2_000.0})
    panel["industry_level1_code"] = "A01"
    panel["is_nonfinancial_industry"] = True
    calendar = pd.DataFrame(
        {
            "timestamp": dates,
            "calendar_date": dates,
            "source_market_type": 1,
            "is_trading_day": True,
        }
    )
    fees = pd.DataFrame(
        [
            {
                "fee_code": fee_code,
                "effective_date": pd.Timestamp("2018-01-01"),
                "end_date": pd.Timestamp("2025-12-31"),
                "market": "a_share",
                "side": side,
                "basis": "amount",
                "rate": rate,
                "included_in_ledger": True,
                "included_in": None,
            }
            for fee_code, rate in (
                ("BROKER_COMMISSION", 0.0003),
                ("TRANSFER_FEE", 0.0001),
            )
            for side in ("buy", "sell")
        ]
        + [
            {
                "fee_code": "STAMP_DUTY",
                "effective_date": pd.Timestamp("2018-01-01"),
                "end_date": pd.Timestamp("2025-12-31"),
                "market": "a_share",
                "side": "sell",
                "basis": "amount",
                "rate": 0.001,
                "included_in_ledger": True,
                "included_in": None,
            }
        ]
    )
    fees["included_in"] = pd.Series([pd.NA] * len(fees), dtype="string")
    return {
        "market_panel_1d": panel,
        "price_volume_facts": dataset.loc[
            :,
            [
                "market_id",
                "asset_id",
                "timestamp",
                "log_float_market_cap",
                "momentum_252_20",
                "amihud_60d",
            ],
        ].copy(),
        "fundamental_factors": dataset.loc[
            :, ["market_id", "asset_id", "timestamp"]
        ].assign(earnings_yield=0.01),
        "index_market": pd.DataFrame(
            [
                {
                    "source_asset_id": code,
                    "timestamp": timestamp,
                    "trade_date": timestamp,
                    "index_name": code,
                    "open": float("nan"),
                    "close": 100.0,
                    "period_return": 0.0,
                }
                for code in ("H00905", "H00852")
                for timestamp in dates
            ]
        ),
        "execution_mask": execution,
        "pv_all": dataset,
        "trading_calendar": calendar,
        "a_share_fee_schedule": fees,
        "benchmark_weights": pd.DataFrame(
            [
                {
                    "market_id": "a_share",
                    "benchmark_id": benchmark_id,
                    "asset_id": asset,
                    "timestamp": timestamp,
                    "weight": 0.5,
                }
                for benchmark_id in ("csi500", "csi1000")
                for timestamp in dates
                for asset in assets
            ]
        ),
    }


def _profile(frames: dict[str, pd.DataFrame]) -> ResearchProfile:
    base = research_profile()
    interim = _interim_specs()
    processed = _processed_specs()
    dataset = _dataset_spec()
    protocol = a_share.protocol("pv_all")

    def source_identity(raw_root: Path, data_id: str) -> str:
        marker = Path(raw_root) / f"{data_id}.synthetic"
        return content_hash_for_files(
            ((data_id, marker),),
            error="synthetic_source_identity_invalid",
        )

    def interim_producer(
        *,
        layout: StorageLayout,
        raw_root: Path,
        data_ids: tuple[str, ...],
        start_date: str,
        end_date: str,
        resource_policy: object,
        instance_metadata: dict[str, object],
    ) -> InterimImportResult:
        del start_date, end_date
        data_id = data_ids[0]
        spec = interim.get(data_id)
        root = layout.target_root(
            data_id,
            layout.interim_data_root(spec.source_id, data_id),
        )
        write_dataset(
            root,
            frames[data_id].copy(),
            spec=spec,
            instance_metadata=instance_metadata,
            resource_policy=resource_policy,
            borrowed_input=True,
        )
        return InterimImportResult(
            artifact_roots={data_id: root},
            skipped=(),
            pruning_report_text="synthetic persisted source",
        )

    def processed_producer(
        spec: ProcessedDataSpec,
        *,
        layout: StorageLayout,
        time_window: TimeWindow,
        processed_catalog: ProcessedDataCatalog,
        source_catalog: InterimDataCatalog,
        resource_policy: object,
    ) -> pd.DataFrame:
        del layout, time_window, processed_catalog, source_catalog, resource_policy
        return frames[spec.data_id].copy()

    def dataset_producer(
        *,
        layout: StorageLayout,
        dataset_catalog: ResearchDatasetCatalog,
        processed_catalog: ProcessedDataCatalog,
        dataset_id: str,
        time_window: TimeWindow,
        resource_policy: object,
        instance_metadata: dict[str, object],
    ) -> Path:
        del dataset_catalog, processed_catalog, time_window
        root = layout.target_root(
            dataset_id,
            layout.research_dataset_root(dataset.market_id, dataset_id),
        )
        write_dataset(
            root,
            frames[dataset_id].copy(),
            spec=dataset,
            instance_metadata=instance_metadata,
            resource_policy=resource_policy,
            borrowed_input=True,
        )
        return root

    bindings: dict[str, ProducerBinding] = {
        spec.data_id: ProducerBinding(
            ProducerKind.INTERIM,
            interim_producer,
            source_identity=source_identity,
        )
        for spec in interim.list()
    }
    bindings.update(
        {
            spec.data_id: ProducerBinding(ProducerKind.PROCESSED, processed_producer)
            for spec in processed.list()
        }
    )
    bindings.update(
        {
            "pv_all": ProducerBinding(ProducerKind.DATASET, dataset_producer),
            "pv_all_evaluation": base.producer_bindings["pv_all_evaluation"],
            "pv_all_report_artifact": base.producer_bindings["pv_all_report_artifact"],
            **{
                target.producer_id: base.producer_bindings[target.producer_id]
                for target in protocol.target_specs("research run")
                if target.producer_id in base.producer_bindings
                and base.producer_bindings[target.producer_id].kind
                is not ProducerKind.EVALUATION
            },
        }
    )
    return ResearchProfile(
        profile_id=A_SHARE_PROFILE_ID,
        market_id=base.market_id,
        frequency=base.frequency,
        protocol=a_share.protocol,
        factors=FactorCatalog(),
        labels=LabelCatalog(),
        interim=interim,
        processed=processed,
        research_datasets=ResearchDatasetCatalog((dataset,)),
        evaluations=EvaluationCatalog(
            (
                EvaluationSpec(
                    evaluation_id="pv_all_evaluation",
                    market_id=base.market_id,
                    research_dataset_id="pv_all",
                    signals=(
                        SignalSpec(
                            signal_id="x",
                            source_kind="factor",
                            source_column="x",
                        ),
                    ),
                    label_columns=("label",),
                ),
            )
        ),
        models=ModelCatalog(
            tuple(
                ModelSpec(
                    model_id=f"pv_all_{algorithm}",
                    model_algorithm=algorithm,
                    input_factor_columns=("x",),
                    label_column="label",
                )
                for algorithm in a_share.MODEL_FAMILY_ORDER
            )
        ),
        producer_bindings=bindings,
        schema_inventory_important_columns=(),
    )


def _prepare_model_artifacts(
    profile: ResearchProfile,
    layout: StorageLayout,
    plan: dict[str, object],
) -> None:
    protocol = profile.protocol(scenario_id="pv_all", access_window="development")
    plans = {item["target_id"]: item for item in plan["targets"]}
    requested_window = plan["research_window"]
    champion_ids = tuple(
        f"pv_all_{algorithm}" for algorithm in a_share.MODEL_FAMILY_ORDER
    )
    target_id, model_ids = protocol.model_selection_target_id, champion_ids
    target = next(
        item
        for item in protocol.target_specs("research run")
        if item.target_id == target_id
    )
    target_window = research._semantic_target_time_window(
        "research run",
        protocol,
        target,
        None,
        requested_window,
    )
    target_research_window = research._window_payload(
        target_window,
        access=str(requested_window["access"]),
    )
    envelope = research._target_envelope_for_spec(
        profile=profile,
        protocol=protocol,
        target_spec=target,
        data_spec=None,
        research_window=target_research_window,
        upstream_identities=plans[target_id]["upstream_identities"],
        software_identity=research.production_software_identity(),
        raw_content_identity=None,
    )
    selected_model_id = "pv_all_equal_weight_rank_ensemble"
    quarterly = _quarterly_result(
        1,
        trial_layout=layout,
        model_id=selected_model_id,
        columns=("x",),
        predictor=RankEnsemblePredictor(columns=("x",), directions=(1.0,)),
    )
    selection = quarterly.selection_metadata
    selection["candidate_family_ids"] = list(model_ids)
    selection["tie_break_order"] = list(model_ids)
    selection["selected_family_id"] = selected_model_id
    selection["selected_model_id"] = selected_model_id
    selection["families"] = []
    references = []
    for model_id in model_ids:
        reference = _trial_reference(
            layout,
            quarterly.quarter,
            family_id=model_id,
            model_id=model_id,
            target_id=target_id,
            dataset_id=protocol.scenario_id,
            data_identity=plans[target_id]["upstream_identities"][protocol.scenario_id],
            protocol=protocol.manifest(),
            rankic=0.3 if model_id == selected_model_id else 0.2,
        )
        references.append(reference)
        selection["families"].append(
            {
                "family_id": model_id,
                "config_tie_break_order": [model_id],
                "selected_config_id": model_id,
                "trial_ledger_semantic_identity": reference["semantic_identity"],
            }
        )
    training = pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "asset_id": str(asset),
                "timestamp": timestamp,
                "x": float(asset),
                "label": float(asset) / 10,
                "eligible": True,
            }
            for timestamp in pd.bdate_range("2018-11-01", periods=30)
            for asset in range(6)
        ]
    )

    def chunks(columns, *, input_columns):
        yield training.loc[:, list(columns)]

    def fitted_families():
        for model_id, reference in zip(model_ids, references, strict=True):
            fitted, lease = _fit_selected_model_result(
                dataset_spec=profile.research_datasets.get("pv_all"),
                selected_spec=profile.models.get(model_id),
                chunks=chunks,
                fitting_cutoff="2018-12-31",
                quarter_id="2019Q1",
                root_seed=42,
                config_index=0,
            )
            try:
                yield replace(
                    quarterly,
                    model_id=model_id,
                    family_id=model_id,
                    model_result=fitted,
                    trial_ledgers=(reference,),
                )
            finally:
                fitted = None
                if lease is not None:
                    lease.close()

    for algorithm in a_share.MODEL_FAMILY_ORDER:
        preflight_model_dependency(algorithm)
    with runtime_resource_policy_scope(None):
        write_quarterly_model_artifact(
            layout.semantic_artifact_root("a_share", target_id),
            model_id=target_id,
            envelope=envelope,
            quarterly_results=fitted_families(),
            trial_layout=layout,
        )


def test_five_portfolio_candidates_build_and_reuse_one_real_persisted_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile(_frames())
    layout = StorageLayout(project_root=tmp_path)
    for spec in profile.interim.list():
        marker = (
            layout.project_root
            / "data/raw"
            / str(spec.source_id)
            / f"{spec.data_id}.synthetic"
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(spec.data_id.encode("utf-8"))
    request = {
        "start": "2019-01-02",
        "end": "2019-01-15",
        "access": "development",
    }
    planned = research.plan_targets(
        profile=profile,
        action="research run",
        terminal_targets=profile.protocol("pv_all").terminal_targets("research run"),
        layout=layout,
        scenario_id="pv_all",
        research_window=request,
    )
    _prepare_model_artifacts(profile, layout, planned)
    result = research.research_run(
        profile=profile,
        layout=layout,
        scenario_id="pv_all",
        start_date=request["start"],
        end_date=request["end"],
    )
    results = {item["target_id"]: item for item in result["target_results"]}
    assert result["status"] == "COMPLETE", result["target_results"]
    record = read_research_run_record(layout.research_run_path(result["run_id"]))
    assert record.status.value == "COMPLETE"
    forged = record.payload()
    next(target for target in forged["target_results"] if target["path"] is not None)[
        "path"
    ] = "../outside"
    with pytest.raises(ValueError, match="research_run_record_invalid"):
        research_run_record_from_payload(forged)
    protocol = profile.protocol(scenario_id="pv_all", access_window="development")
    portfolio_ids = tuple(
        f"pv_all_{candidate.candidate_id}_ledger"
        for candidate in protocol.portfolio_candidates()
    )
    failed = {
        target_id: (
            results[target_id]["status"],
            results[target_id].get("reason"),
        )
        for target_id in (
            protocol.model_selection_target_id,
            protocol.persisted_prediction_target_id,
            "pv_all_statistics",
            *portfolio_ids,
        )
        if results[target_id]["status"] not in {"BUILT", "REUSED"}
    }
    assert not failed, failed
    statistics_identity = results[
        protocol.persisted_prediction_target_id.replace(
            "persisted_predictions",
            "statistics",
        )
    ]["instance_identity"]
    assert statistics_identity is not None
    assert results[protocol.persisted_prediction_target_id]["status"] == "BUILT"
    champion_identity = results[protocol.persisted_prediction_target_id][
        "instance_identity"
    ]
    champion_root = layout.instance_root(
        layout.semantic_artifact_root(
            "a_share",
            protocol.persisted_prediction_target_id,
        ),
        champion_identity,
    )
    prediction_manifest, _prediction_members = validate_output_bundle(
        champion_root / "outputs" / protocol.persisted_prediction_target_id,
        expected_output_id=protocol.persisted_prediction_target_id,
        expected_run_id=f"run_{protocol.persisted_prediction_target_id}",
    )
    assert (
        prediction_manifest["metadata"]["prediction_metadata"]["test_start"]
        == "2019-01-01"
    )
    statistics_root = layout.instance_root(
        layout.semantic_artifact_root("a_share", "pv_all_statistics"),
        statistics_identity,
    )
    statistics_artifact = research.read_statistics_artifact(
        statistics_root,
        expected_identity=statistics_identity,
    )
    with statistics_artifact.table("prediction_diagnostics.parquet") as table:
        prediction_diagnostics = table.iloc[0].copy()
    assert prediction_diagnostics["status"] == "UNDEFINED"
    assert prediction_diagnostics["reason"] == "hac_lag_exceeds_observations"
    with statistics_artifact.table("robustness.parquet") as table:
        robustness = table.copy()
    assert robustness["status"].eq("UNDEFINED").all()
    assert set(robustness["reason"]) == {
        "bootstrap_block_exceeds_common_observations",
    }
    assert (
        robustness.loc[
            robustness.test_id.isin(
                ("spa_reality_check", "stepm_superior_models", "model_confidence_set")
            ),
            "observation_count",
        ]
        .eq(10)
        .all()
    )
    assert (
        robustness.loc[
            ~robustness.test_id.isin(
                ("spa_reality_check", "stepm_superior_models", "model_confidence_set")
            ),
            "observation_count",
        ]
        .eq(0)
        .all()
    )
    from quant_research_platform.artifacts.outputs.bundle_validation import (
        read_prediction_output_bundle,
    )
    from quant_research_platform.contracts.business_manifest import (
        BusinessManifestEnvelope,
    )
    from quant_research_platform.research.statistics_production import (
        build_a_share_statistics_tables,
    )
    from quant_research_platform.storage.semantic_artifacts import (
        read_semantic_artifact,
        write_semantic_artifact,
    )
    from tests.helpers.runtime_resources import capture_test_policy

    bundle = champion_root / "outputs" / protocol.persisted_prediction_target_id
    damaged = tmp_path / "missing_family"
    shutil.copytree(native_filesystem_path(bundle), native_filesystem_path(damaged))
    values = pd.read_parquet(damaged / "prediction.parquet")
    missing_family = "pv_all_ridge_regression"
    values = values.loc[~values.family_id.eq(missing_family)]
    values.to_parquet(damaged / "prediction.parquet", index=False)
    manifest = json.loads((damaged / "manifest.json").read_text(encoding="utf-8"))
    manifest["metadata"]["prediction_metadata"]["prediction_rows"] = len(values)
    (damaged / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    model = research.read_quarterly_model_artifact(
        layout.instance_root(
            layout.semantic_artifact_root(
                "a_share", protocol.model_selection_target_id
            ),
            results[protocol.model_selection_target_id]["instance_identity"],
        ),
        trial_layout=layout,
    )
    data_layout = layout.with_target_roots(
        {
            "pv_all": layout.instance_root(
                layout.research_dataset_root("a_share", "pv_all"),
                results["pv_all"]["instance_identity"],
            )
        }
    )
    failed_policy = capture_test_policy(available_memory_bytes=128 * 1024**2)
    with runtime_resource_policy_scope(failed_policy):
        with read_prediction_output_bundle(
            damaged,
            expected_output_id=protocol.persisted_prediction_target_id,
            time_window=TimeWindow(request["start"], request["end"]),
        ) as (_manifest, lookup):
            with pytest.raises(ValueError, match="missing_required_prediction_keys"):
                build_a_share_statistics_tables(
                    layout=data_layout,
                    dataset_catalog=profile.research_datasets,
                    dataset_id="pv_all",
                    prediction_id=protocol.persisted_prediction_target_id,
                    persisted_model=model,
                    protocol_manifest=protocol.manifest(),
                    time_window=TimeWindow(request["start"], request["end"]),
                    prediction_lookup=lookup,
                    resource_policy=failed_policy,
                )
        assert failed_policy.used_memory_bytes == 0

    def forged_writer(staging):
        shutil.copytree(
            native_filesystem_path(bundle),
            native_filesystem_path(
                staging / "outputs" / protocol.persisted_prediction_target_id
            ),
        )
        path = (
            staging
            / "outputs"
            / protocol.persisted_prediction_target_id
            / "manifest.json"
        )
        forged = json.loads(
            Path(native_filesystem_path(path)).read_text(encoding="utf-8")
        )
        forged["metadata"]["model_metadata"]["champion_by_quarter"]["2019Q1"] = (
            missing_family
        )
        Path(native_filesystem_path(path)).write_text(
            json.dumps(forged), encoding="utf-8"
        )

    forged_root = write_semantic_artifact(
        tmp_path / "forged",
        BusinessManifestEnvelope(**read_semantic_artifact(champion_root)),
        forged_writer,
    )
    prediction_target = next(
        target
        for target in protocol.target_specs("research run")
        if target.target_id == protocol.persisted_prediction_target_id
    )
    with pytest.raises(ValueError, match="prediction_model_lineage_mismatch"):
        research._validate_semantic_target_root(
            profile,
            protocol,
            prediction_target,
            forged_root,
            champion_identity,
            trial_layout=layout,
        )
    from quant_research_platform.artifacts.outputs import (
        prediction as prediction_writer,
    )
    from quant_research_platform.contracts.predictions import PredictionSpec
    from quant_research_platform.policy.lifecycle import closing_iterator
    from quant_research_platform.research.predictions.runner import (
        build_prediction_result_from_persisted_model,
    )

    original_bytes = Path(
        native_filesystem_path(bundle / "prediction.parquet")
    ).read_bytes()
    native_write = prediction_writer.write_parquet_chunks
    primary = KeyboardInterrupt("merged_writer_interrupted")
    observed = []

    def fail_merged_write(frames, path, **kwargs):
        if path.name != "prediction.parquet":
            return native_write(frames, path, **kwargs)

        def interrupted_frames():
            first = None
            with closing_iterator(iter(frames)) as stream:
                try:
                    first = next(stream)
                    observed.append(len(first))
                    yield first
                    first = None
                    raise primary
                finally:
                    first = None

        return native_write(interrupted_frames(), path, **kwargs)

    interrupt_policy = capture_test_policy(available_memory_bytes=128 * 1024**2)
    with (
        monkeypatch.context() as patch,
        runtime_resource_policy_scope(interrupt_policy),
    ):
        patch.setattr(prediction_writer, "write_parquet_chunks", fail_merged_write)
        pspec = PredictionSpec(
            prediction_id=protocol.persisted_prediction_target_id,
            model_id=protocol.model_selection_target_id,
            research_dataset_id="pv_all",
            train_end=None,
            test_start="2019-01-01",
            test_end=request["end"],
            prediction_mode="a_share_oos",
            prediction_column="raw_prediction",
        )
        output = build_prediction_result_from_persisted_model(
            layout=data_layout,
            dataset_catalog=profile.research_datasets,
            prediction_spec=pspec,
            persisted_model=model,
            resource_policy=interrupt_policy,
        )
        with pytest.raises(KeyboardInterrupt) as caught:
            with output:
                research.write_prediction_outputs(
                    spec=pspec,
                    output_spec=research._output_spec_for_target(
                        pspec.prediction_id, "a_share", "prediction"
                    ),
                    layout=StorageLayout(tmp_path / "interrupted_prediction"),
                    output_chunks=output.output_chunks,
                    model_metadata=output.model_metadata,
                    prediction_metadata_factory=output.prediction_metadata_factory,
                    resource_policy=interrupt_policy,
                )
        assert caught.value is primary
        assert observed
        assert interrupt_policy.used_memory_bytes == 0
    assert (
        Path(native_filesystem_path(bundle / "prediction.parquet")).read_bytes()
        == original_bytes
    )
    for target_id in portfolio_ids:
        identity = results[target_id]["instance_identity"]
        root = layout.instance_root(
            layout.semantic_artifact_root("a_share", target_id),
            identity,
        )
        artifact = research.read_portfolio_ledger_artifact(
            root,
            expected_identity=identity,
        )
        chunks = [session_frame(batch) for batch in artifact.iter_chunks()]
        assert chunks
        assert all(
            tuple(chunk.columns) == CONTINUOUS_EXECUTION_COLUMNS for chunk in chunks
        )
        assert all(chunk["asset_id"].iloc[-1] == "__cash__" for chunk in chunks)
        if "enhanced" in target_id:
            assert all(
                chunk.loc[chunk["asset_id"].eq("__cash__"), "decision_reason"]
                .eq("enhanced_benchmark_only")
                .all()
                for chunk in chunks
            )

    reused = research.research_run(
        profile=profile,
        layout=layout,
        scenario_id="pv_all",
        start_date=request["start"],
        end_date=request["end"],
    )
    reused_results = {item["target_id"]: item for item in reused["target_results"]}
    assert result["run_id"] != reused["run_id"]
    assert all(
        reused_results[target_id]["status"] == "REUSED"
        for target_id in (
            *portfolio_ids,
            protocol.persisted_prediction_target_id,
            protocol.persisted_prediction_target_id,
        )
    )

    assert {
        item["target_id"]: item["instance_identity"]
        for item in result["target_results"]
    } == {
        item["target_id"]: item["instance_identity"]
        for item in reused["target_results"]
    }
    with monkeypatch.context() as report_only:
        report_only.setattr(
            research,
            "_plan_targets",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError("report replanned")),
        )
        report_only.setattr(
            research,
            "preflight_model_specs",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("report trained")
            ),
        )
        report_only.setattr(
            research,
            "_default_raw_root",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("report read raw")
            ),
        )
        report_result = research.report_build(
            profile=profile,
            layout=layout,
            run_id=result["run_id"],
        )
    report_identity = report_result["target_result"]["instance_identity"]
    assert report_result["source_run_status"] == "COMPLETE"
    assert report_result["status"] == "BLOCKED"
    assert report_result["target_result"]["status"] == "BUILT"
    assert report_identity is not None
    report_root = layout.instance_root(
        layout.semantic_artifact_root("a_share", "pv_all_report_artifact"),
        report_identity,
    )
    report = research.read_research_report_artifact(
        report_root,
        expected_identity=report_identity,
    )
    assert set(report["sections"]) == set(REQUIRED_REPORT_SECTIONS)
    capacity = report["sections"]["capacity"]
    assert capacity["status"] == "UNDEFINED"
    defined_capacity = [
        row for row in capacity["results"] if row["status"] == "DEFINED"
    ]
    undefined_capacity = [
        row for row in capacity["results"] if row["status"] == "UNDEFINED"
    ]
    assert len(capacity["results"]) == 96
    assert len(defined_capacity) == 72 and len(undefined_capacity) == 24
    assert all(
        row["value"] is not None and row["reason"] == "" for row in defined_capacity
    )
    assert all(
        row["value"] is None and row["reason"] == "no_rebalance_requests"
        for row in undefined_capacity
    )
    assert report["sections"]["risk"]["status"] == "BLOCKED"
    assert report["sections"]["attribution"]["status"] == "BLOCKED"

    from tests.interfaces.published_research_test_support import (
        check_published_evidence_views,
    )

    check_published_evidence_views(
        layout=layout,
        profile=profile,
        run_id=result["run_id"],
        other_run_id=reused["run_id"],
        report_identity=report_identity,
        monkeypatch=monkeypatch,
    )

    statistics_manifest = statistics_root / "manifest.json"
    saved_statistics_manifest = statistics_root / "manifest.saved"
    statistics_manifest.rename(saved_statistics_manifest)
    try:
        damaged_report = research.report_build(
            profile=profile,
            layout=layout,
            run_id=reused["run_id"],
        )
    finally:
        saved_statistics_manifest.rename(statistics_manifest)
    assert damaged_report["source_run_status"] == "COMPLETE"
    assert damaged_report["status"] == "FAILED"
    assert damaged_report["report"]["sections"]["factor diagnostics"] == {
        "status": "FAILED",
        "reason": "target_reference_invalid",
    }

    changed = research.plan_targets(
        profile=profile,
        action="research run",
        terminal_targets=protocol.terminal_targets("research run"),
        layout=layout,
        scenario_id="pv_all",
        research_window={
            "start": "2019-01-03",
            "end": "2019-01-15",
            "access": "development",
        },
    )
    first = {item["target_id"]: item for item in planned["targets"]}
    second = {item["target_id"]: item for item in changed["targets"]}
    for target_id in (
        "trading_calendar",
        "market_panel_1d",
        "execution_mask",
        "a_share_fee_schedule",
        "pv_all",
    ):
        assert (
            first[target_id]["requested_identity"]
            == second[target_id]["requested_identity"]
        )
    for target_id in portfolio_ids:
        assert (
            first[target_id]["requested_identity"]
            != second[target_id]["requested_identity"]
        )
    assert (
        first[protocol.persisted_prediction_target_id]["requested_identity"]
        == second[protocol.persisted_prediction_target_id]["requested_identity"]
    )
