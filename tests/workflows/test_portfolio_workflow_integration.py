from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.catalog.models import ModelCatalog
from quant_research_platform.catalog.portfolios import PortfolioCatalog
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
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.portfolios import PORTFOLIO_LEDGER_COLUMNS
from quant_research_platform.contracts.targets import (
    ProducerBinding,
    ProducerKind,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.profiles.research import ResearchProfile, research_profile
from quant_research_platform.research.models.persisted import (
    write_quarterly_model_artifact,
)
from quant_research_platform.research.models.rank_ensemble import RankEnsemblePredictor
from quant_research_platform.research.protocols import a_share
from quant_research_platform.sources.csmar.interim import InterimImportResult
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import content_hash_for_files
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research
from tests.research.workflow_test_support import _quarterly_result


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
    return InterimDataCatalog((calendar, fee))


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
    return ProcessedDataCatalog((panel, execution, benchmark))


def _dataset_spec() -> ResearchDatasetSpec:
    return ResearchDatasetSpec(
        data_id="pv_all",
        frequency=DataFrequency.DAILY,
        grain=DataGrain.RESEARCH_ROW,
        column_roles={
            "market_id": ColumnRole.KEY,
            "asset_id": ColumnRole.KEY,
            "timestamp": ColumnRole.TIMESTAMP,
            "x": ColumnRole.FACTOR,
            "label": ColumnRole.LABEL,
            "eligible": ColumnRole.MASK,
            "float_market_cap": ColumnRole.CONTEXT,
        },
        primary_key=("market_id", "asset_id", "timestamp"),
        partition_columns=("year",),
        market_id="a_share",
        input_data_ids=("market_panel_1d",),
        base_input_data_id="market_panel_1d",
        factor_columns=("x",),
        label_columns=("label",),
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
                "label": float(asset == "b") * 0.01,
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
        profile_id="synthetic_a_share_persisted_portfolio",
        market_id=base.market_id,
        frequency=base.frequency,
        protocol=a_share.protocol,
        factors=FactorCatalog(),
        labels=LabelCatalog(),
        interim=interim,
        processed=processed,
        research_datasets=ResearchDatasetCatalog((dataset,)),
        evaluations=EvaluationCatalog(),
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
        build_portfolio_catalog=lambda **_kwargs: PortfolioCatalog(),
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
    champion_ids = tuple(
        f"pv_all_{algorithm}" for algorithm in a_share.MODEL_FAMILY_ORDER
    )
    for target_id, model_ids in (
        (protocol.model_selection_target_id, champion_ids),
        (protocol.ew_rank_model_target_id, ("pv_all_equal_weight_rank_ensemble",)),
    ):
        target = next(
            item
            for item in protocol.target_specs("research run")
            if item.target_id == target_id
        )
        envelope = research._target_envelope_for_spec(
            profile=profile,
            protocol=protocol,
            target_spec=target,
            data_spec=None,
            research_window={
                "start": "2019-01-02",
                "end": "2019-01-15",
                "access": "development",
            },
            upstream_identities=plans[target_id]["upstream_identities"],
            software_identity=research.production_software_identity(),
            raw_content_identity=None,
        )
        selected_model_id = "pv_all_equal_weight_rank_ensemble"
        quarterly = _quarterly_result(
            1,
            model_id=selected_model_id,
            columns=("x",),
            predictor=RankEnsemblePredictor(columns=("x",), directions=(1.0,)),
        )
        selection = quarterly.model_result.metadata["model_selection"]
        base_trial = deepcopy(selection["families"][0]["trials"][0])
        selection["candidate_family_ids"] = list(model_ids)
        selection["tie_break_order"] = list(model_ids)
        selection["selected_family_id"] = selected_model_id
        selection["selected_model_id"] = selected_model_id
        selection["families"] = []
        for model_id in model_ids:
            trial = deepcopy(base_trial)
            trial["model_id"] = model_id
            selection["families"].append(
                {
                    "family_id": model_id,
                    "config_tie_break_order": [model_id],
                    "selected_config_id": model_id,
                    "trials": [trial],
                    "validation_rows": trial["validation_rows"],
                    "selection_metric_value": trial["validation_rankic"],
                    "selection_metric_reason": trial["validation_rankic_reason"],
                }
            )
        write_quarterly_model_artifact(
            layout.semantic_artifact_root("a_share", target_id),
            model_id=target_id,
            envelope=envelope,
            quarterly_results=(quarterly,),
        )


def test_five_portfolio_candidates_build_and_reuse_one_real_persisted_chain(
    tmp_path: Path,
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
        terminal_targets=("pv_all_research",),
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
        for target_id in portfolio_ids
        if results[target_id]["status"] != "BUILT"
    }
    assert not failed, failed
    assert results[protocol.persisted_prediction_target_id]["status"] == "BUILT"
    assert results[protocol.ew_rank_prediction_target_id]["status"] == "BUILT"
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
        chunks = list(artifact.iter_chunks())
        assert chunks
        assert all(tuple(chunk.columns) == PORTFOLIO_LEDGER_COLUMNS for chunk in chunks)
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
    assert all(
        reused_results[target_id]["status"] == "REUSED"
        for target_id in (
            *portfolio_ids,
            protocol.persisted_prediction_target_id,
            protocol.ew_rank_prediction_target_id,
        )
    )

    changed = research.plan_targets(
        profile=profile,
        action="research run",
        terminal_targets=("pv_all_research",),
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
    for target_id in (
        protocol.persisted_prediction_target_id,
        protocol.ew_rank_prediction_target_id,
        *portfolio_ids,
    ):
        assert (
            first[target_id]["requested_identity"]
            != second[target_id]["requested_identity"]
        )
