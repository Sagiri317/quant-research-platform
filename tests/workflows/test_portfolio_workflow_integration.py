from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
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
from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.csmar.interim import InterimImportResult
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import (
    content_hash_for_files,
    native_filesystem_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research
from tests.helpers.portfolio_ledger import session_frame
from tests.helpers.research_configuration import (
    A_SHARE_MODEL_ALGORITHMS,
    PV25_FACTORS,
    research_configuration,
)
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
    statistical_factors = tuple(factor for factor in PV25_FACTORS if factor != "x")
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
            "label_available_at": ColumnRole.CONTEXT,
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
        context_columns=("float_market_cap", "label_available_at"),
    )


def _frames(dates: pd.DatetimeIndex | None = None) -> dict[str, pd.DataFrame]:
    if dates is None:
        dates = pd.bdate_range("2018-12-03", "2019-01-15")
    calendar_dates = dates.append(
        pd.bdate_range(dates[-1] + pd.offsets.BDay(1), periods=21)
    )
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
                    for factor in PV25_FACTORS
                    if factor != "x"
                },
                "label": float(asset == "b") * 0.01,
                "label_available_at": calendar_dates[date_index + 1],
                "forward_return_20d": float(asset == "b") * 0.02,
                "forward_return_20d_available_at": calendar_dates[date_index + 21],
                "eligible": True,
                "float_market_cap": 1_000.0 if asset == "a" else 2_000.0,
            }
            for asset in assets
            for date_index, timestamp in enumerate(dates)
        ]
    )
    panel["period_return"] = 0.001
    panel["float_market_cap"] = panel["asset_id"].map({"a": 1_000.0, "b": 2_000.0})
    panel["industry_level1_code"] = "A01"
    panel["is_nonfinancial_industry"] = True
    calendar = pd.DataFrame(
        {
            "timestamp": calendar_dates,
            "calendar_date": calendar_dates,
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

    @contextmanager
    def processed_producer(
        spec: ProcessedDataSpec,
        *,
        layout: StorageLayout,
        time_window: TimeWindow,
        processed_catalog: ProcessedDataCatalog,
        source_catalog: InterimDataCatalog,
        resource_policy: object,
    ):
        from quant_research_platform.policy.lifecycle import closing_iterator
        from quant_research_platform.runtime.resources import (
            OwnedValue,
            pandas_frame_payload_bytes,
        )

        del layout, time_window, processed_catalog, source_catalog, resource_policy

        def output():
            lease = yield pandas_frame_payload_bytes(frames[spec.data_id])
            yield OwnedValue(frames[spec.data_id].copy(), lease)

        with closing_iterator(output()) as data:
            yield data, {}, None

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
        del processed_catalog, time_window
        selected = dataset_catalog.get(dataset_id)
        root = layout.target_root(
            dataset_id,
            layout.research_dataset_root(selected.market_id, dataset_id),
        )
        write_dataset(
            root,
            frames[dataset_id].loc[:, list(selected.required_columns)].copy(),
            spec=selected,
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
                for algorithm in A_SHARE_MODEL_ALGORITHMS
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
        f"pv_all_{algorithm}" for algorithm in A_SHARE_MODEL_ALGORITHMS
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
            model_spec=profile.models.get(model_id),
            root_seed=42,
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
                "label_available_at": timestamp + pd.offsets.BDay(1),
                "eligible": True,
            }
            for timestamp in pd.bdate_range("2018-11-01", periods=30)
            for asset in range(6)
        ]
    )

    def chunks(columns, *, input_columns, start, end):
        history_start = protocol.manifest()["formal_dataset_window"]["start"]
        selected = training["timestamp"].between(
            pd.Timestamp(history_start if start is None else start), pd.Timestamp(end)
        )
        yield training.loc[selected, list(columns)]

    def fitted_families():
        for model_id, reference in zip(model_ids, references, strict=True):
            fitted, lease = _fit_selected_model_result(
                dataset_spec=profile.research_datasets.get("pv_all"),
                selected_spec=profile.models.get(model_id),
                candidate_parameters=dict(profile.models.get(model_id).hyperparameters),
                chunks=chunks,
                fitting_cutoff="2018-12-31",
                quarter_id="2019Q1",
                root_seed=42,
            )
            owned = OwnedValue(
                replace(
                    quarterly,
                    model_id=model_id,
                    family_id=model_id,
                    model_result=fitted,
                    trial_ledgers=(reference,),
                ),
                lease,
            )
            fitted = None
            try:
                yield owned
            finally:
                if owned.lease.active:
                    owned.release(owner="test.model")
                owned = None

    for algorithm in A_SHARE_MODEL_ALGORITHMS:
        preflight_model_dependency(algorithm)
    with runtime_resource_policy_scope(None):
        write_quarterly_model_artifact(
            layout.semantic_artifact_root("a_share", target_id),
            model_id=target_id,
            envelope=envelope,
            quarterly_results=fitted_families(),
            trial_layout=layout,
        )


@pytest.mark.parametrize(
    "start,end,access,expected_start",
    (
        ("2019-01-01", "2023-12-31", "development", "2014-01-01"),
        ("2023-12-01", "2023-12-31", "development", "2023-12-01"),
        ("2024-01-01", "2025-12-31", "holdout", "2024-01-01"),
    ),
)
def test_factor_and_model_windows_keep_separate_formal_boundaries(
    start, end, access, expected_start
):
    protocol = a_share.protocol("pv_all", access_window=access)
    requested = TimeWindow(start, end)
    factor_window, model_window = a_share.analysis_windows(
        protocol.manifest(), requested
    )
    assert factor_window == TimeWindow(expected_start, end)
    assert model_window == requested
    for target in ("pv_all_evaluation", "pv_all_statistics"):
        assert (
            research._a_share_target_window(protocol, target, requested)
            == factor_window
        )


def test_holdout_evidence_uses_full_registered_window_and_actual_identities(tmp_path):
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.contracts.targets import (
        ResearchRunRecord,
        ResearchRunTargetResult,
        unmeasured_operation_evidence,
    )

    layout = StorageLayout(tmp_path)
    profile = _profile(_frames())
    protocol = replace(
        profile.protocol("pv_all"),
        access_window="holdout",
        frozen_development={
            "run_id": "run_development",
            "alpha_sources": [
                {
                    "reference": {
                        "target_id": "alpha_ledger",
                        "instance_identity": "a" * 64,
                    },
                    "observed_at": "2023-12-29",
                }
            ],
        },
    )
    window = {"start": "2024-01-01", "end": "2024-03-31", "access": "holdout"}
    facts = (
        ("pv_all", ProducerKind.DATASET, "b" * 64),
        (protocol.model_selection_target_id, ProducerKind.MODEL, "c" * 64),
        (protocol.persisted_prediction_target_id, ProducerKind.PREDICTION, "d" * 64),
    )
    results = tuple(
        ResearchRunTargetResult(
            target_id=target,
            producer_kind=kind,
            status="BUILT",
            instance_identity=identity,
            schema_identity="e" * 64,
            artifact_root=f"data/artifacts/a_share/{target}",
        )
        for target, kind, identity in facts
    )
    record = ResearchRunRecord(
        run_id="run_first",
        profile_id=profile.profile_id,
        scenario_id="pv_all",
        protocol=protocol.manifest(),
        protocol_identity=canonical_json_hash(protocol.manifest()),
        research_window=window,
        software_identity="f" * 64,
        terminal_targets=(protocol.persisted_prediction_target_id,),
        target_results=results,
        status="COMPLETE",
        operation_evidence=unmeasured_operation_evidence("fixture"),
        holdout_evidence={"qualification": "HOLDOUT_FIRST_EVIDENCE", "prior_runs": []},
    )
    research.write_research_run_record(layout.research_run_path(record.run_id), record)
    forged = record.payload()
    forged["holdout_evidence"]["qualification"] = "HOLDOUT_NOT_ACCESSED"
    import json

    forged_path = layout.research_run_path("run_forged_not_accessed")
    forged["run_id"] = "run_forged_not_accessed"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="research_run_record_invalid"):
        read_research_run_record(forged_path)
    forged_path.unlink()
    with pytest.raises(ValueError, match="research_run_holdout_evidence_invalid"):
        replace(
            record,
            status="BLOCKED",
            holdout_evidence=forged["holdout_evidence"],
            target_results=(
                replace(
                    results[0],
                    status="BLOCKED",
                    reason="dependency_blocked",
                    instance_identity=None,
                    schema_identity=None,
                    artifact_root=None,
                ),
                *results[1:],
            ),
        )
    arguments = dict(
        layout=layout,
        profile=profile,
        protocol=protocol,
        research_window=window,
        software_identity="f" * 64,
        target_results=results,
    )
    assert research._holdout_evidence(**arguments)["qualification"] == "REUSE"
    for status in ("FAILED", "BLOCKED"):
        partial_results = (
            *results[:-1],
            replace(
                results[-1],
                status=status,
                reason="prediction_unavailable",
                instance_identity=None,
                schema_identity=None,
                artifact_root=None,
            ),
        )
        assert (
            research._holdout_evidence(
                **{**arguments, "target_results": partial_results}
            )["qualification"]
            != "REUSE"
        )
        partial = replace(record, status=status, target_results=partial_results)
        partial_layout = StorageLayout(tmp_path / status.lower())
        research.write_research_run_record(
            partial_layout.research_run_path(partial.run_id), partial
        )
        prior = research._holdout_evidence(**{**arguments, "layout": partial_layout})
        assert prior["qualification"] == "POST_HOLDOUT_CORRECTED"
        assert [row["run_id"] for row in prior["prior_runs"]] == [partial.run_id]
    other_window = {"start": "2025-01-01", "end": "2025-03-31", "access": "holdout"}
    assert (
        research._holdout_evidence(**{**arguments, "research_window": other_window})[
            "qualification"
        ]
        == "POST_HOLDOUT_ADDITIONAL_EVIDENCE"
    )
    new_protocol = replace(
        protocol,
        frozen_development={
            **protocol.frozen_development,
            "run_id": "run_other_development",
        },
    )
    changed_output = (*results[:-1], replace(results[-1], instance_identity="1" * 64))
    assert (
        research._holdout_evidence(
            **{
                **arguments,
                "protocol": new_protocol,
                "target_results": changed_output,
            }
        )["qualification"]
        == "POST_HOLDOUT_ADDITIONAL_EVIDENCE"
    )
    assert (
        research._holdout_evidence(
            **{**arguments, "protocol": new_protocol, "software_identity": "2" * 64}
        )["qualification"]
        == "POST_HOLDOUT_CORRECTED"
    )


def test_holdout_freezes_models_and_alpha_across_observation_quarters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.research.models import fitting, production
    from quant_research_platform.research.portfolios import production as portfolios
    from quant_research_platform.runtime import resources

    captured = []
    original_capture = resources._capture_runtime_resource_policy

    def capture(**kwargs):
        policy = original_capture(**kwargs)
        captured.append(policy)
        return policy

    def assert_one_capture(previous):
        assert len(captured) == previous + 1
        assert captured[-1].state == "CLOSED"
        assert captured[-1].used_memory_bytes == 0
        assert resources.current_runtime_resource_policy() is None

    monkeypatch.setattr(resources, "_capture_runtime_resource_policy", capture)

    dates = (
        pd.bdate_range("2014-01-01", "2025-12-31", freq="BME")
        .union(pd.bdate_range("2023-12-01", "2024-01-15"))
        .union(pd.bdate_range("2025-12-01", "2025-12-31"))
    )
    frames = _frames(dates)
    profile = _profile(frames)
    base = research_profile()
    configuration = research_configuration(development=("2023-12-01", "2023-12-31"))
    for model in configuration["models"]:
        model["input_factor_columns"] = ["reversal_1d"]
        model["input_feature_columns"] = []
        model["label_column"] = "forward_return_1d"
        model["hyperparameters"] = {}
        model.pop("parameter_candidates", None)
        if model["model_algorithm"] == "ridge_regression":
            model["hyperparameters"] = {"alpha": 0.1}
            model["parameter_candidates"] = [{"regularization_strength": 0.1}]
    facts = {
        "is_a_share_scope": True,
        "is_trading": True,
        "is_st": False,
        "is_pt": False,
        "listing_age_trading_days": 252,
        "close": 100.0,
        "adj_close": 100.0,
        "volume": 100.0,
        "amount": 10_000.0,
        "turnover": 0.1,
        "industry_standard_id": "CSRC2012",
        "industry_level1_code": "A01",
        "industry_level1_name": "industry",
        "total_market_cap": 4_000.0,
    }
    frames["pv_all"] = frames["pv_all"].assign(**facts)
    sessions = pd.DatetimeIndex(frames["trading_calendar"]["timestamp"])
    positions = sessions.get_indexer(frames["pv_all"]["timestamp"])
    frames["pv_all"]["forward_return_1d"] = frames["pv_all"]["label"]
    frames["pv_all"]["forward_return_1d_available_at"] = sessions.take(positions + 2)
    from quant_research_platform.research.features.transforms import (
        canonical_feature_pipeline,
    )

    frames["pv_all"] = canonical_feature_pipeline(
        frames["pv_all"], factor_columns=PV25_FACTORS
    )
    key_roles = {
        "market_id": ColumnRole.KEY,
        "asset_id": ColumnRole.KEY,
        "timestamp": ColumnRole.TIMESTAMP,
    }
    price = replace(
        profile.processed.get("price_volume_facts"),
        column_roles={
            **key_roles,
            **{name: ColumnRole.FACTOR for name in PV25_FACTORS},
        },
    )
    forward = replace(
        base.processed.get("forward_returns"),
        column_roles={
            **key_roles,
            "forward_return_1d": ColumnRole.LABEL,
            "forward_return_1d_available_at": ColumnRole.LABEL,
            "forward_return_20d": ColumnRole.LABEL,
            "forward_return_20d_available_at": ColumnRole.LABEL,
        },
        processed_input_data_ids=(),
    )
    frames["price_volume_facts"] = (
        frames["pv_all"].loc[:, list(price.required_columns)].copy()
    )
    frames["forward_returns"] = (
        frames["pv_all"].loc[:, list(forward.required_columns)].copy()
    )
    profile = replace(
        profile,
        factors=base.factors,
        labels=base.labels,
        processed=ProcessedDataCatalog(
            (
                *(
                    item
                    for item in profile.processed.list()
                    if item.data_id != price.data_id
                ),
                price,
                forward,
            )
        ),
        producer_bindings={
            **base.producer_bindings,
            **profile.producer_bindings,
            "forward_returns": profile.producer_bindings["price_volume_facts"],
        },
    )
    from quant_research_platform.profiles.research import bind_research_configuration

    profile = bind_research_configuration(profile, configuration)
    layout = StorageLayout(tmp_path)
    for spec in profile.interim.list():
        marker = (
            tmp_path / "data/raw" / str(spec.source_id) / f"{spec.data_id}.synthetic"
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(spec.data_id.encode())
    previous = len(captured)
    development = research.research_run(
        profile=profile,
        layout=layout,
        research_plan=configuration,
    )
    assert development["status"] == "COMPLETE", development
    assert_one_capture(previous)
    development_record = read_research_run_record(
        layout.research_run_path(development["run_id"])
    )
    assert development_record.trial_ledgers
    from quant_research_platform.research.datasets.io import (
        iter_research_dataset_chunks,
    )

    dataset_reference = next(
        item for item in development_record.target_results if item.target_id == "pv_all"
    )
    dataset_layout = layout.with_target_roots(
        {
            "pv_all": layout.instance_root(
                layout.research_dataset_root("a_share", "pv_all"),
                dataset_reference.instance_identity,
            )
        }
    )
    with runtime_resource_policy_scope() as policy:
        for chunk in iter_research_dataset_chunks(
            dataset_layout,
            market_id="a_share",
            dataset_id="pv_all",
            columns=("timestamp",),
        ):
            assert chunk.timestamp.max() <= pd.Timestamp("2023-12-31")
        with pytest.raises(
            PermissionError, match="research_dataset_access_window_forbidden"
        ):
            iter_research_dataset_chunks(
                dataset_layout,
                market_id="a_share",
                dataset_id="pv_all",
                time_window=TimeWindow("2024-01-01", "2024-01-02"),
            )
        assert policy.used_memory_bytes == 0
    source_model = next(
        item
        for item in development_record.target_results
        if item.producer_kind is ProducerKind.MODEL
    )
    model_root = layout.instance_root(
        layout.semantic_artifact_root("a_share", source_model.target_id),
        source_model.instance_identity,
    )
    model_hash = content_hash_for_files(
        (
            (item.relative_to(model_root).as_posix(), item)
            for item in model_root.rglob("*")
            if item.is_file()
        ),
        error="model_hash_invalid",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("holdout attempted supervised learning")

    monkeypatch.setattr(fitting, "_fit_model_result_owned_from_frames", forbidden)
    monkeypatch.setattr(production, "_select_model_family_winners", forbidden)
    monkeypatch.setattr(portfolios, "calibrate_expected_alpha", forbidden)
    request = dict(
        profile=profile,
        layout=layout,
        research_plan=configuration,
        holdout=True,
        development_run_id=development["run_id"],
    )
    with pytest.raises(
        PermissionError, match="holdout_frozen_development_identity_unavailable"
    ):
        research.research_run(**{**request, "development_run_id": None}, plan=True)
    previous = len(captured)
    with pytest.raises(ValueError, match="dataset_metadata_file_invalid"):
        research.research_run(
            **{**request, "development_run_id": "run_missing"}, plan=True
        )
    assert_one_capture(previous)
    with pytest.raises(ValueError, match="holdout_development_run_invalid"):
        research.research_run(
            **{
                **request,
                "research_plan": research_configuration(
                    name="fusion_nf_core",
                    development=("2023-12-01", "2023-12-31"),
                ),
            },
            plan=True,
        )
    incomplete = replace(
        development_record,
        run_id="run_incomplete",
        trial_ledgers=(),
        terminal_targets=development_record.terminal_targets[:1],
    )
    research.write_research_run_record(
        layout.research_run_path(incomplete.run_id), incomplete
    )
    with pytest.raises(ValueError, match="holdout_development_run_invalid"):
        research.research_run(
            **{**request, "development_run_id": incomplete.run_id}, plan=True
        )
    wrong_reference = replace(
        development_record,
        run_id="run_wrong_reference",
        trial_ledgers=(),
        target_results=tuple(
            replace(item, schema_identity="f" * 64)
            if item.target_id == source_model.target_id
            else item
            for item in development_record.target_results
        ),
    )
    research.write_research_run_record(
        layout.research_run_path(wrong_reference.run_id), wrong_reference
    )
    with pytest.raises(ValueError, match="research_run_target_reference_invalid"):
        research.research_run(
            **{**request, "development_run_id": wrong_reference.run_id}, plan=True
        )
    from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked

    def unavailable_alpha(*args, **kwargs):
        raise PortfolioTargetBlocked("alpha_calibration_insufficient_observations")

    with monkeypatch.context() as preflight:
        preflight.setattr(research, "frozen_alpha_calibration", unavailable_alpha)
        preflight.setattr(research, "_execute_action", forbidden)
        planned_block = research.research_run(**request, plan=True)
        assert planned_block["status"] == "BLOCKED" and "run_id" not in planned_block
        blocked = research.research_run(**request)
    blocked_record = read_research_run_record(
        layout.research_run_path(blocked["run_id"])
    )
    assert blocked_record.status.value == "BLOCKED"
    assert blocked_record.holdout_evidence["qualification"] == "HOLDOUT_NOT_ACCESSED"
    assert all(item.status.value == "BLOCKED" for item in blocked_record.target_results)
    assert (
        blocked_record.protocol["frozen_development"]["run_id"] == development["run_id"]
    )
    previous = len(captured)
    planned = research.research_run(**request, plan=True)
    assert_one_capture(previous)
    model_plan = next(
        item
        for item in planned["targets"]
        if item["target_id"] == source_model.target_id
    )
    assert model_plan["decision"] == "REUSE"
    assert model_plan["upstream_target_ids"] == []
    assert model_plan["requested_identity"] == source_model.instance_identity
    previous = len(captured)
    holdout = research.research_run(**request)
    assert holdout["status"] == "COMPLETE", holdout
    assert_one_capture(previous)
    record = read_research_run_record(layout.research_run_path(holdout["run_id"]))
    assert record.trial_ledgers == ()
    assert record.holdout_evidence["qualification"] == "HOLDOUT_FIRST_EVIDENCE"
    assert len(record.protocol["frozen_development"]["alpha_sources"]) == 2
    results = {item.target_id: item for item in record.target_results}
    assert (
        results[source_model.target_id].instance_identity
        == source_model.instance_identity
    )
    prediction_id = profile.protocol("pv_all").persisted_prediction_target_id
    prediction_root = layout.instance_root(
        layout.semantic_artifact_root("a_share", prediction_id),
        results[prediction_id].instance_identity,
    )
    manifest, _ = validate_output_bundle(
        prediction_root / "outputs" / prediction_id,
        expected_output_id=prediction_id,
        expected_run_id=f"run_{prediction_id}",
    )
    models = manifest["metadata"]["model_metadata"]["quarterly_models"]
    assert {item["source_quarter"] for item in models} == {"2023Q4"}
    assert {item["quarter_id"] for item in models} == {
        str(period) for period in pd.period_range("2024Q1", "2025Q4", freq="Q")
    }
    assert len(models) == 8 * len(A_SHARE_MODEL_ALGORITHMS)
    from quant_research_platform.storage.semantic_artifacts import (
        compare_semantic_artifacts,
    )

    frozen, blocker = research._frozen_holdout_protocol(
        profile,
        layout,
        profile.protocol("pv_all"),
        development["run_id"],
        "2024-01-01",
        "2025-12-31",
    )
    assert blocker is None
    target_layout = layout.with_target_roots(
        {
            item.target_id: layout.instance_root(
                layout.project_root / item.artifact_root, item.instance_identity
            )
            for item in record.target_results
        }
    )
    rebuild_layout = StorageLayout(tmp_path / "prediction_statistics_rebuild")
    with runtime_resource_policy_scope() as policy:
        for target_id in (prediction_id, "pv_all_statistics"):
            spec = next(
                item
                for item in frozen.target_specs("research run")
                if item.target_id == target_id
            )
            target_plan = next(
                item for item in planned["targets"] if item["target_id"] == target_id
            )
            rebuilt = research._build_target(
                profile=profile,
                protocol=frozen,
                layout=rebuild_layout,
                spec=spec,
                planned=target_plan,
                action="research run",
                research_window=record.research_window,
                software_identity=research.production_software_identity(),
                target_layout=target_layout,
                policy=policy,
                raw_root=tmp_path / "data/raw",
            )
            assert rebuilt.status.value == "BUILT", rebuilt
            rebuilt_root = research._target_instance_root(
                profile, rebuild_layout, target_id, rebuilt.instance_identity
            )
            assert (
                compare_semantic_artifacts(
                    target_layout.target_root(target_id, Path("missing")),
                    rebuilt_root,
                    trial_layout=layout,
                ).status
                == "MATCH"
            )
        assert policy.used_memory_bytes == 0
    assert not rebuild_layout.semantic_artifact_root(
        "a_share", source_model.target_id
    ).exists()
    sources = []
    for candidate in profile.protocol("pv_all").portfolio_candidates():
        target_id = f"pv_all_{candidate.candidate_id}_ledger"
        artifact = research.read_portfolio_ledger_artifact(
            layout.instance_root(
                layout.semantic_artifact_root("a_share", target_id),
                results[target_id].instance_identity,
            ),
            expected_identity=results[target_id].instance_identity,
        )
        if candidate.construction == "benchmark_enhanced":
            values = {
                row.number
                for batch in artifact.iter_chunks()
                for row in batch.diagnostics
                if row.name == "alpha_slope"
            }
            assert len(values) == 1
            source = next(
                item
                for item in development_record.target_results
                if item.target_id == target_id
            )
            original = research.read_portfolio_ledger_artifact(
                layout.instance_root(
                    layout.semantic_artifact_root("a_share", target_id),
                    source.instance_identity,
                ),
                expected_identity=source.instance_identity,
            )
            calibration, _ = portfolios.frozen_alpha_calibration(
                original, cutoff=pd.Timestamp("2023-12-31")
            )
            assert (
                calibration.valid_rebalance_dates >= 6 and calibration.coverage >= 0.8
            )
            assert values == {calibration.slope}
            sources.append(values)
    assert sources[0] == sources[1]
    report = research.report_build(profile=profile, layout=layout, run_id=record.run_id)
    assert (
        report["report"]["sections"]["development vs holdout"]["evidence"][
            "qualification"
        ]
        == "HOLDOUT_FIRST_EVIDENCE"
    )
    previous = len(captured)
    reused = research.research_run(**request)
    assert reused["status"] == "COMPLETE", reused
    assert_one_capture(previous)
    reused_record = read_research_run_record(layout.research_run_path(reused["run_id"]))
    assert reused_record.holdout_evidence["qualification"] == "REUSE"
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.storage.software_identity import (
        ProductionSoftwareIdentity,
    )

    previous = len(captured)
    new_development = research.research_run(
        profile=profile,
        layout=layout,
        research_plan=configuration,
    )
    assert new_development["status"] == "COMPLETE", new_development
    assert_one_capture(previous)
    assert new_development["run_id"] != development["run_id"]
    future = frames["pv_all"].timestamp.ge("2024-01-01")
    frames["pv_all"].loc[
        future, ["label", "forward_return_1d", "forward_return_20d"]
    ] *= -100.0
    software_payload = research.production_software_identity().payload
    software_payload["package_content_hash"] = "f" * 64
    corrected_software = ProductionSoftwareIdentity(
        software_payload,
        canonical_json_hash(software_payload),
    )
    monkeypatch.setattr(
        research, "production_software_identity", lambda: corrected_software
    )
    previous = len(captured)
    corrected = research.research_run(
        **{**request, "development_run_id": new_development["run_id"]}
    )
    assert corrected["status"] == "COMPLETE", corrected
    assert_one_capture(previous)
    corrected_record = read_research_run_record(
        layout.research_run_path(corrected["run_id"])
    )
    assert (
        corrected_record.holdout_evidence["qualification"] == "POST_HOLDOUT_CORRECTED"
    )
    assert record.run_id in {
        item["run_id"] for item in corrected_record.holdout_evidence["prior_runs"]
    }
    corrected_prediction = next(
        item
        for item in corrected_record.target_results
        if item.target_id == prediction_id
    )
    corrected_root = layout.instance_root(
        layout.semantic_artifact_root("a_share", prediction_id),
        corrected_prediction.instance_identity,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(
            native_filesystem_path(
                prediction_root / "outputs" / prediction_id / "prediction.parquet"
            )
        ),
        pd.read_parquet(
            native_filesystem_path(
                corrected_root / "outputs" / prediction_id / "prediction.parquet"
            )
        ),
        check_exact=True,
    )
    corrected_report = research.report_build(
        profile=profile, layout=layout, run_id=corrected_record.run_id
    )
    assert (
        corrected_report["report"]["sections"]["development vs holdout"]["evidence"][
            "qualification"
        ]
        == "POST_HOLDOUT_CORRECTED"
    )
    assert (
        content_hash_for_files(
            (
                (item.relative_to(model_root).as_posix(), item)
                for item in model_root.rglob("*")
                if item.is_file()
            ),
            error="model_hash_invalid",
        )
        == model_hash
    )


def test_five_portfolio_candidates_build_and_reuse_one_real_persisted_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _frames()
    profile = _profile(frames)
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
        research_plan=research_configuration(
            development=(request["start"], request["end"])
        ),
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
    observations = frames["pv_all"]
    mature_dates = observations.loc[
        observations.timestamp.between(request["start"], request["end"])
        & observations.forward_return_20d_available_at.le(pd.Timestamp(request["end"])),
        "timestamp",
    ].nunique()
    assert (
        robustness.loc[
            robustness.test_id.isin(
                ("spa_reality_check", "stepm_superior_models", "model_confidence_set")
            ),
            "observation_count",
        ]
        .eq(mature_dates)
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
        research_plan=research_configuration(
            development=(request["start"], request["end"])
        ),
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
    coverage = report["sections"]["Data/Coverage"]
    assert coverage["row_count"] > 0
    assert coverage["evaluation"]["rows_input"] > 0
    dataset_ref = next(
        item for item in record.target_results if item.target_id == "pv_all"
    )
    manifest = research.read_dataset_manifest_evidence(
        layout.instance_root(
            layout.project_root / dataset_ref.artifact_root,
            dataset_ref.instance_identity,
        )
    )
    assert coverage["factor_nan_profile"] == {
        column: manifest["factor_nan_profile"].get(column, 0)
        for column in manifest["columns"]
        if manifest["column_roles"][column] == "factor"
    }
    assert coverage["pit"]["raw_completeness"] == "UNDEFINED"
    assert report["sections"]["factor diagnostics"]["results"]
    assert report["sections"]["model selection"]["quarterly_selections"]
    assert report["sections"]["model selection"]["configurations"]
    blocked_ids = {
        "fundamental_factors",
        "pv_all_champion_csi500_enhanced_ledger",
        "pv_all_champion_csi1000_enhanced_ledger",
        "pv_all_portfolio_statistics",
    }
    partial = replace(
        record,
        status="BLOCKED",
        target_results=tuple(
            replace(
                item,
                status="BLOCKED",
                reason="dependency_blocked",
                instance_identity=None,
                schema_identity=None,
                artifact_root=None,
            )
            if item.target_id in blocked_ids
            else item
            for item in record.target_results
        ),
    )
    from quant_research_platform.artifacts.research import produce_a_share_report

    with (
        runtime_resource_policy_scope() as policy,
        policy.lease("report.partial", 1) as report_lease,
    ):
        partial_report = produce_a_share_report(
            layout=layout,
            record=partial,
            report_lease=report_lease,
            quarterly_selections=model.quarterly_selections,
        )
    assert partial_report["report_status"] == "BLOCKED"
    retained = partial_report["sections"]["portfolio"]["results"]
    assert len(retained) == 3
    assert all(
        row["ledger_identity"] and row["economics"]["nav"] > 0 for row in retained
    )
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
