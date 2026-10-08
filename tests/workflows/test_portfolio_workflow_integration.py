from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

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
from quant_research_platform.research.protocols import a_share
from quant_research_platform.sources.csmar.interim import InterimImportResult
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.identity import (
    content_hash_for_files,
    filesystem_exists,
    filesystem_glob,
    filesystem_is_file,
    filesystem_mkdir,
    filesystem_read_text,
    filesystem_rename,
    filesystem_unlink,
    filesystem_write_bytes,
    filesystem_write_text,
    native_filesystem_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows import research
from tests.helpers.portfolio_ledger import session_frame
from tests.helpers.research_configuration import (
    A_SHARE_MODEL_ALGORITHMS,
    PV25_FACTORS,
    compact_a_share_configuration,
    compact_a_share_windows,
    research_configuration,
)


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
        (
            calendar,
            fee,
            *(
                research_profile().interim.get(target)
                for target in ("index_market", "market_bars_1d", "stock_master")
            ),
        )
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
            "industry_standard_id": ColumnRole.KEY,
            "industry_level1_code": ColumnRole.KEY,
            "is_nonfinancial_industry": ColumnRole.CONTEXT,
            "is_a_share_scope": ColumnRole.CONTEXT,
            "is_trading": ColumnRole.CONTEXT,
            "is_st": ColumnRole.CONTEXT,
            "is_pt": ColumnRole.CONTEXT,
            "listing_age_trading_days": ColumnRole.CONTEXT,
            "has_252_trading_sessions": ColumnRole.CONTEXT,
            "close": ColumnRole.CONTEXT,
            "volume": ColumnRole.CONTEXT,
            "turnover": ColumnRole.CONTEXT,
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
            ("market_model_factors", ("beta_252d", "residual_volatility_60d")),
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
        dates = pd.bdate_range("2014-01-01", "2023-12-31")
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
    panel["industry_standard_id"] = "CSRC2012"
    panel["industry_level1_code"] = "A"
    panel["is_nonfinancial_industry"] = True
    panel = panel.assign(
        is_a_share_scope=True,
        is_trading=True,
        is_st=False,
        is_pt=False,
        listing_age_trading_days=300,
        has_252_trading_sessions=True,
        close=panel.adj_close,
        volume=1000.0,
        turnover=0.01,
    )
    from quant_research_platform.research.factors.engine import (
        calculate_factors_for_panel,
    )

    market_factors = calculate_factors_for_panel(
        panel,
        definitions=tuple(
            definition
            for definition in a_share.factor_catalog().list()
            if definition.family == "market_model"
        ),
    )
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
    bars = (
        panel[["asset_id", "timestamp", "market_type"]]
        .rename(columns={"asset_id": "source_asset_id"})
        .assign(filling_flag=0)
    )
    first = panel.groupby("asset_id", sort=False).timestamp.min()
    stocks = pd.DataFrame(
        {
            "source_asset_id": first.index,
            "timestamp": first.to_numpy(),
            "listing_date": first.to_numpy(),
            "delisting_date": pd.NaT,
            "company_status": "active",
        }
    )
    return {
        "market_bars_1d": bars.reindex(
            columns=research_profile().interim.get("market_bars_1d").required_columns
        ),
        "stock_master": stocks.reindex(
            columns=research_profile().interim.get("stock_master").required_columns
        ),
        "market_panel_1d": panel,
        "market_model_factors": market_factors,
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
        instance_metadata: dict[str, object],
    ) -> InterimImportResult:
        data_id = data_ids[0]
        spec = interim.get(data_id)
        root = layout.target_root(
            data_id,
            layout.interim_data_root(spec.source_id, data_id),
        )
        write_dataset(
            root,
            (
                frames[data_id].loc[
                    frames[data_id].timestamp.between(start_date, end_date)
                ]
                if "timestamp" in frames[data_id]
                else frames[data_id]
            ).copy(),
            spec=spec,
            instance_metadata=instance_metadata,
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
    ):
        from quant_research_platform.policy.lifecycle import closing_iterator

        del layout, processed_catalog, source_catalog
        selected_frame = frames[spec.data_id].loc[
            frames[spec.data_id].timestamp.between(
                pd.Timestamp(time_window.start), pd.Timestamp(time_window.end)
            )
        ]

        def output():
            yield selected_frame.copy()

        with closing_iterator(output()) as data:
            yield data, {}, None

    def dataset_producer(
        *,
        layout: StorageLayout,
        dataset_catalog: ResearchDatasetCatalog,
        processed_catalog: ProcessedDataCatalog,
        dataset_id: str,
        time_window: TimeWindow,
        instance_metadata: dict[str, object],
    ) -> Path:
        del processed_catalog
        selected = dataset_catalog.get(dataset_id)
        root = layout.target_root(
            dataset_id,
            layout.research_dataset_root(selected.market_id, dataset_id),
        )
        output = (
            frames[dataset_id]
            .loc[
                frames[dataset_id].timestamp.between(
                    pd.Timestamp(time_window.start), pd.Timestamp(time_window.end)
                ),
                list(selected.required_columns),
            ]
            .copy()
        )
        for availability in (
            column for column in output if column.endswith("_available_at")
        ):
            label = availability.removesuffix("_available_at")
            invisible = output[availability] > pd.Timestamp(time_window.end)
            output.loc[invisible, label] = float("nan")
            output.loc[invisible, availability] = pd.NaT
        write_dataset(root, output, spec=selected, instance_metadata=instance_metadata)
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
                    hyperparameters=next(
                        item["hyperparameters"]
                        for item in research_configuration(small_models=True)["models"]
                        if item["model_algorithm"] == algorithm
                    ),
                )
                for algorithm in A_SHARE_MODEL_ALGORITHMS
            )
        ),
        producer_bindings=bindings,
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
    for target in ("pv_all_evaluation", *protocol.statistics_target_ids):
        assert (
            protocol.target_window("research run", None, requested, target)
            == factor_window
        )


def test_holdout_evidence_uses_full_registered_window_and_actual_identities(
    tmp_path, monkeypatch
):
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
            "targets": {},
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
        *(
            (target, ProducerKind.PREDICTION, identity * 64)
            for target, identity in zip(
                protocol.prediction_target_ids, ("d", "1"), strict=True
            )
        ),
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
        terminal_targets=protocol.prediction_target_ids,
        target_results=results,
        status="COMPLETE",
        operation_evidence=unmeasured_operation_evidence("fixture"),
        holdout_evidence={
            "qualification": "HOLDOUT_FIRST_EVIDENCE",
            "prior_runs": [],
        },
    )
    research.write_research_run_record(layout.research_run_path(record.run_id), record)
    forged = record.payload()
    forged["holdout_evidence"]["qualification"] = "HOLDOUT_NOT_ACCESSED"
    import json

    forged_path = layout.research_run_path("run_forged_not_accessed")
    forged["run_id"] = "run_forged_not_accessed"
    filesystem_write_text(forged_path, json.dumps(forged), encoding="utf-8")
    with pytest.raises(ValueError, match="research_run_record_invalid"):
        read_research_run_record(forged_path)
    filesystem_unlink(forged_path)
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
        target_results=results,
    )
    emitted = []
    read_paths = []
    original_glob = research.filesystem_glob
    original_read = research.read_research_run_record

    def incremental_glob(*args):
        for path in original_glob(*args):
            emitted.append(path)
            yield path
            assert path in read_paths

    def tracked_read(path, **kwargs):
        read_paths.append(path)
        return original_read(path, **kwargs)

    monkeypatch.setattr(research, "filesystem_glob", incremental_glob)
    monkeypatch.setattr(research, "read_research_run_record", tracked_read)
    assert research._holdout_evidence(**arguments)["qualification"] == "REUSE"
    assert emitted and len(emitted) == len(read_paths)
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
    assert (
        research._holdout_evidence(**{**arguments, "protocol": new_protocol})[
            "qualification"
        ]
        == "REUSE"
    )
    changed_output = (
        *results[:-1],
        replace(results[-1], instance_identity="3" * 64),
    )
    assert (
        research._holdout_evidence(**{**arguments, "target_results": changed_output})[
            "qualification"
        ]
        == "POST_HOLDOUT_ADDITIONAL_EVIDENCE"
    )


def test_portfolio_statistics_target_identity_tracks_its_consumed_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.profiles.research import bind_research_configuration

    configuration = research_configuration()
    protocol = bind_research_configuration(research_profile(), configuration).protocol(
        "pv_all"
    )
    target_id = "pv_all_portfolio_statistics"
    original = protocol.target_manifest(target_id, "STATISTIC")
    model = protocol.target_manifest("pv_all_model_selection_refit", "MODEL")
    ledger = protocol.target_manifest(
        "pv_all_global_ew_rank_top20_equal_weight_ledger", "PORTFOLIO"
    )
    assert original["portfolio_candidates"] == [
        candidate.candidate_id for candidate in protocol.portfolio_candidates()
    ]
    assert original["statistics"] == protocol.manifest()["statistics"]

    monkeypatch.setattr(
        a_share, "STATISTICS_REPETITIONS", a_share.STATISTICS_REPETITIONS + 1
    )
    changed = protocol.target_manifest(target_id, "STATISTIC")
    assert canonical_json_hash(changed) != canonical_json_hash(original)
    assert protocol.target_manifest("pv_all_model_selection_refit", "MODEL") == model
    assert (
        protocol.target_manifest(
            "pv_all_global_ew_rank_top20_equal_weight_ledger", "PORTFOLIO"
        )
        == ledger
    )


@compact_a_share_windows()
@patch.multiple(
    a_share,
    DATA_END=pd.Timestamp("2019-10-31").date(),
    FORMAL_DATASET_END=pd.Timestamp("2019-10-31").date(),
    DEVELOPMENT_END=pd.Timestamp("2019-08-30").date(),
    HOLDOUT_START=pd.Timestamp("2019-09-02").date(),
    HOLDOUT_END=pd.Timestamp("2019-10-31").date(),
)
def test_holdout_freezes_models_and_alpha_across_observation_quarters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from quant_research_platform.research.models import fitting, production
    from quant_research_platform.research.portfolios import production as portfolios

    # Model backends have their own real ladder tests; the raw complete chain
    # retains the 252-by-100 risk gate. This test keeps all ten live portfolios.
    configuration = compact_a_share_configuration()
    for candidate in configuration["portfolios"]:
        if candidate["benchmark_id"] == "csi1000":
            candidate["rebalance_schedule"] = "monthly-last-trading-day"
    configuration["models"] = [
        model
        for model in configuration["models"]
        if model["model_algorithm"]
        in {"equal_weight_rank_ensemble", "ridge_regression"}
    ]
    model_count = len(configuration["models"])
    assert model_count == 2
    assert len(configuration["portfolios"]) == 5
    history = pd.bdate_range("2017-10-02", periods=5)
    for quarter in pd.period_range("2018Q1", "2018Q4", freq="Q"):
        history = history.append(pd.bdate_range(quarter.start_time, periods=10))
    development_dates = pd.bdate_range("2019-01-01", "2019-08-30")
    holdout_dates = pd.bdate_range("2019-09-02", "2019-10-31")
    dates = history.append(development_dates).append(holdout_dates)
    assert dates.is_unique
    assert len(holdout_dates) == 44
    assert set(holdout_dates.to_period("Q")) == {
        pd.Period("2019Q3"),
        pd.Period("2019Q4"),
    }
    frames = _frames(dates)
    # Avoid a perfectly interpolated two-point target with zero Huber residual scale.
    dataset = frames["pv_all"]
    day = (dataset.timestamp - pd.Timestamp("2017-10-01")).dt.days
    dataset["label"] += 0.002 * np.sin(day * 0.173 + dataset.asset_id.eq("b") * 0.71)
    profile = _profile(frames)
    base = research_profile()
    for model in configuration["models"]:
        model["input_factor_columns"] = ["reversal_1d"]
        model["input_feature_columns"] = []
        model["label_column"] = "forward_return_1d"
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
        "has_252_trading_sessions": True,
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
        filesystem_mkdir(marker.parent, parents=True, exist_ok=True)
        filesystem_write_bytes(marker, spec.data_id.encode())
    development = research.research_run(
        profile=profile,
        layout=layout,
        research_plan=configuration,
    )
    assert development["status"] == "COMPLETE", development
    development_record = read_research_run_record(
        layout.research_run_path(development["run_id"])
    )
    assert development_record.trial_ledgers
    requested_ridge = next(
        model
        for model in configuration["models"]
        if model["model_algorithm"] == "ridge_regression"
    )
    submitted_base = dict(requested_ridge["hyperparameters"])
    requested_ridge["hyperparameters"] = {
        "alpha": -1.0,
        "random_state": 999,
        "unused_native_option": True,
    }
    recorded_ridge = next(
        model
        for model in development_record.payload()["protocol"]["research_configuration"][
            "models"
        ]
        if model["model_algorithm"] == "ridge_regression"
    )
    assert recorded_ridge["hyperparameters"] == submitted_base
    assert recorded_ridge["hyperparameters"] != requested_ridge["hyperparameters"]
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
    for chunk in iter_research_dataset_chunks(
        dataset_layout,
        market_id="a_share",
        dataset_id="pv_all",
        columns=("timestamp",),
    ):
        assert chunk.timestamp.max() <= pd.Timestamp("2019-08-30")
    with pytest.raises(
        PermissionError, match="research_dataset_access_window_forbidden"
    ):
        iter_research_dataset_chunks(
            dataset_layout,
            market_id="a_share",
            dataset_id="pv_all",
            time_window=TimeWindow("2019-09-02", "2019-09-03"),
        )
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
            for item in filesystem_glob(model_root, "**/*")
            if filesystem_is_file(item)
        ),
        error="model_hash_invalid",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("holdout attempted supervised learning")

    monkeypatch.setattr(fitting, "_fit_model_result_from_frames", forbidden)
    monkeypatch.setattr(production, "_build_a_share_models", forbidden)
    monkeypatch.setattr(portfolios, "calibrate_expected_alpha", forbidden)
    from quant_research_platform.research.predictions import runner as predictions

    loaded_sources = []
    original_predictor_reader = predictions.read_quarterly_predictor

    def counted_predictor(artifact, selection_group, family, **kwargs):
        loaded_sources.append((selection_group, family))
        return original_predictor_reader(artifact, selection_group, family, **kwargs)

    monkeypatch.setattr(predictions, "read_quarterly_predictor", counted_predictor)
    request = dict(
        profile=profile,
        layout=layout,
        research_plan=configuration,
        holdout=True,
        development_run_id=development["run_id"],
    )
    from copy import deepcopy

    actual_candidate_change = deepcopy(configuration)
    changed_ridge = next(
        model
        for model in actual_candidate_change["models"]
        if model["model_algorithm"] == "ridge_regression"
    )
    changed_ridge["parameter_candidates"][0]["regularization_strength"] = 0.2
    with pytest.raises(ValueError, match="holdout_development_run_invalid"):
        research.research_run(
            **{**request, "research_plan": actual_candidate_change}, plan=True
        )
    with pytest.raises(
        PermissionError, match="holdout_frozen_development_identity_unavailable"
    ):
        research.research_run(**{**request, "development_run_id": None}, plan=True)
    with pytest.raises(ValueError, match="dataset_metadata_file_invalid"):
        research.research_run(
            **{**request, "development_run_id": "run_missing"}, plan=True
        )
    with pytest.raises(ValueError, match="holdout_development_run_invalid"):
        research.research_run(
            **{
                **request,
                "research_plan": compact_a_share_configuration(name="fusion_nf_core"),
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
        from quant_research_platform.research.portfolios import production

        preflight.setattr(production, "frozen_alpha_calibration", unavailable_alpha)
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
    planned = research.research_run(**request, plan=True)
    model_plan = next(
        item
        for item in planned["targets"]
        if item["target_id"] == source_model.target_id
    )
    assert model_plan["decision"] == "REUSE"
    assert model_plan["upstream_target_ids"] == []
    assert model_plan["requested_identity"] == source_model.instance_identity
    holdout = research.research_run(**request)
    assert holdout["status"] == "COMPLETE", holdout
    assert len(loaded_sources) == 2 * model_count
    assert len(set(loaded_sources)) == len(loaded_sources)
    record = read_research_run_record(layout.research_run_path(holdout["run_id"]))
    assert record.trial_ledgers == ()
    assert record.holdout_evidence["qualification"] == "HOLDOUT_FIRST_EVIDENCE"
    expected_sources = {"global": [], "quarterly": []}
    for candidate in profile.protocol("pv_all").portfolio_candidates():
        if candidate.construction != "benchmark_enhanced":
            continue
        reference = next(
            item
            for item in development_record.target_results
            if item.target_id == f"pv_all_{candidate.candidate_id}_ledger"
        )
        expected_sources[candidate.selection_group].append(
            {
                "selection_group": candidate.selection_group,
                "reference": reference.payload(),
                "observed_at": development_record.research_window["end"],
            }
        )
    assert all(len(values) == 2 for values in expected_sources.values())
    assert (
        record.payload()["protocol"]["frozen_development"]["alpha_sources"]
        == expected_sources
    )
    results = {item.target_id: item for item in record.target_results}
    assert (
        results[source_model.target_id].instance_identity
        == source_model.instance_identity
    )
    prediction_ids = profile.protocol("pv_all").prediction_target_ids
    prediction_id = prediction_ids[0]
    prediction_root = layout.instance_root(
        layout.semantic_artifact_root("a_share", prediction_id),
        results[prediction_id].instance_identity,
    )
    for group, current_prediction_id in zip(
        ("global", "quarterly"), prediction_ids, strict=True
    ):
        current_root = layout.instance_root(
            layout.semantic_artifact_root("a_share", current_prediction_id),
            results[current_prediction_id].instance_identity,
        )
        current_manifest, _ = validate_output_bundle(
            current_root / "outputs" / current_prediction_id,
            expected_output_id=current_prediction_id,
            expected_run_id=f"run_{current_prediction_id}",
        )
        metadata = current_manifest["metadata"]["model_metadata"]
        assert metadata["selection_group"] == group
        assert {item["quarter_id"] for item in metadata["quarterly_models"]} == {
            str(period) for period in pd.period_range("2019Q3", "2019Q4", freq="Q")
        }
        assert len(metadata["quarterly_models"]) == 2 * model_count
        assert all(
            item["prediction_source"] == "final"
            for item in metadata["quarterly_models"]
        )
    current_manifest = metadata = _ = None
    from quant_research_platform.storage.semantic_artifacts import (
        compare_semantic_artifacts,
    )

    frozen, blocker = research._frozen_holdout_protocol(
        profile,
        layout,
        profile.protocol("pv_all"),
        development["run_id"],
        "2019-09-02",
        "2019-10-31",
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
    for target_id in (*prediction_ids, *frozen.statistics_target_ids):
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
            time_window=research._semantic_target_time_window(
                "research run", frozen, spec, None, record.research_window
            ),
            access=str(record.research_window["access"]),
            software_identity=research.production_software_identity(),
            target_layout=target_layout,
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
    frozen = spec = None
    assert not filesystem_exists(
        rebuild_layout.semantic_artifact_root("a_share", source_model.target_id)
    )
    sources = {"global": [], "quarterly": []}
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
                original.final_alpha_calibration,
                cutoff=pd.Timestamp("2019-08-30"),
                selection_group=candidate.selection_group,
            )
            assert (
                calibration.valid_rebalance_dates >= 6 and calibration.coverage >= 0.8
            )
            assert values == {calibration.slope}
            sources[candidate.selection_group].append(
                (
                    candidate.rebalance_schedule,
                    calibration.valid_rebalance_dates,
                    values,
                )
            )
    artifact = original = calibration = None
    for values in sources.values():
        assert len(values) == 2
        by_schedule = {schedule: (count, slopes) for schedule, count, slopes in values}
        assert set(by_schedule) == {
            "weekly-last-trading-day",
            "monthly-last-trading-day",
        }
        assert (
            by_schedule["weekly-last-trading-day"][0]
            > by_schedule["monthly-last-trading-day"][0]
            >= 6
        )
    report = research.report_build(profile=profile, layout=layout, run_id=record.run_id)
    assert (
        report["report"]["sections"]["development vs holdout"]["evidence"][
            "qualification"
        ]
        == "HOLDOUT_FIRST_EVIDENCE"
    )
    reused = research.research_run(**request)
    assert reused["status"] == "COMPLETE", reused
    reused_record = read_research_run_record(layout.research_run_path(reused["run_id"]))
    assert reused_record.holdout_evidence["qualification"] == "REUSE"
    new_development = research.research_run(
        profile=profile,
        layout=layout,
        research_plan=configuration,
    )
    assert new_development["status"] == "COMPLETE", new_development
    assert new_development["run_id"] != development["run_id"]
    import shutil

    from quant_research_platform.storage.software_identity import (
        production_software_identity,
    )

    software_root = tmp_path / "software"
    shutil.copytree(
        Path(research.__file__).resolve().parents[1],
        software_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    resource_source = software_root / "runtime" / "resources.py"
    resource_source.write_bytes(
        resource_source.read_bytes() + b"\n# Operational change.\n"
    )
    operational_software = production_software_identity(package_root=software_root)
    monkeypatch.setattr(
        research, "production_software_identity", lambda: operational_software
    )
    operational = research.research_run(**request)
    assert operational["status"] == "COMPLETE", operational
    operational_record = read_research_run_record(
        layout.research_run_path(operational["run_id"])
    )
    assert operational_record.software_identity != record.software_identity
    assert operational_record.holdout_evidence["qualification"] == "REUSE"
    assert {
        item.target_id: item.instance_identity
        for item in operational_record.target_results
    } == {item.target_id: item.instance_identity for item in record.target_results}

    future = frames["pv_all"].timestamp.ge("2019-09-02")
    frames["pv_all"].loc[
        future, ["label", "forward_return_1d", "forward_return_20d"]
    ] *= -100.0
    scientific_source = software_root / "research" / "datasets" / "assembler.py"
    scientific_source.write_bytes(
        scientific_source.read_bytes() + b"\n# Scientific change.\n"
    )
    corrected_software = production_software_identity(package_root=software_root)
    monkeypatch.setattr(
        research, "production_software_identity", lambda: corrected_software
    )
    corrected = research.research_run(
        **{**request, "development_run_id": new_development["run_id"]}
    )
    assert corrected["status"] == "COMPLETE", corrected
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
                for item in filesystem_glob(model_root, "**/*")
                if filesystem_is_file(item)
            ),
            error="model_hash_invalid",
        )
        == model_hash
    )


@compact_a_share_windows()
@patch.multiple(a_share, DEVELOPMENT_END=pd.Timestamp("2019-03-01").date())
def test_both_groups_build_and_reuse_one_real_persisted_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    # This integration proves persisted seams; backend and portfolio matrices
    # are covered by model, execution and five-ledger statistical regressions.
    configuration = compact_a_share_configuration()
    configuration["models"] = [
        model
        for model in configuration["models"]
        if model["model_algorithm"]
        in {"equal_weight_rank_ensemble", "ridge_regression"}
    ]
    configuration["portfolios"] = [
        candidate
        for candidate in configuration["portfolios"]
        if candidate["candidate_id"]
        in {
            "ew_rank_top20_equal_weight",
            "champion_top20_equal_weight",
            "champion_csi500_enhanced",
        }
    ]
    # Every required walk-forward quarter remains represented. The formal
    # quarter has 44 sessions and 23 mature twenty-day labels, including the
    # existing zero-mature and short-HAC nested windows.
    history = pd.bdate_range("2017-10-02", periods=5)
    for quarter in pd.period_range("2018Q1", "2018Q4", freq="Q"):
        history = history.append(pd.bdate_range(quarter.start_time, periods=10))
    dates = history.append(pd.bdate_range("2019-01-01", "2019-03-01"))
    frames = _frames(dates)
    dataset = frames["pv_all"]
    day = (dataset.timestamp - pd.Timestamp("2017-10-01")).dt.days
    dataset["label"] += 0.002 * np.sin(day * 0.173 + dataset.asset_id.eq("b") * 0.71)
    profile = _profile(frames)
    model_ids = {model["model_id"] for model in configuration["models"]}
    profile = replace(
        profile,
        models=ModelCatalog(
            tuple(spec for spec in profile.models.list() if spec.model_id in model_ids)
        ),
    )
    layout = StorageLayout(project_root=tmp_path)
    for spec in profile.interim.list():
        marker = (
            layout.project_root
            / "data/raw"
            / str(spec.source_id)
            / f"{spec.data_id}.synthetic"
        )
        filesystem_mkdir(marker.parent, parents=True, exist_ok=True)
        filesystem_write_bytes(marker, spec.data_id.encode("utf-8"))
    request = {
        "start": "2019-01-01",
        "end": "2019-03-01",
        "access": "development",
    }
    from functools import partial

    from quant_research_platform.profiles import research as profile_module

    bound_library = profile_module.bind_research_configuration(
        research_profile(), configuration
    )
    profile = replace(
        profile,
        producer_bindings={
            **profile.producer_bindings,
            **{
                key: binding
                for key, binding in bound_library.producer_bindings.items()
                if binding.kind is ProducerKind.PORTFOLIO
            },
        },
    )
    for model in configuration["models"]:
        spec = profile.models.get(model["model_id"])
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
    # Keep the real persisted model/ledger chain over this synthetic catalog;
    # the real-source acceptance test covers configuration binding end to end.
    monkeypatch.setattr(
        profile_module,
        "bind_research_configuration",
        lambda selected, configuration, **kwargs: selected,
    )
    planned = research.plan_targets(
        profile=profile,
        action="research run",
        terminal_targets=profile.protocol("pv_all").terminal_targets("research run"),
        layout=layout,
        scenario_id="pv_all",
        research_window=request,
    )
    result = research.research_run(
        profile=profile,
        layout=layout,
        research_plan=configuration,
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
    prediction_id = protocol.prediction_target_ids[0]
    statistics_id = protocol.statistics_target_ids[0]
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
            *protocol.prediction_target_ids,
            *protocol.statistics_target_ids,
            *portfolio_ids,
        )
        if results[target_id]["status"] not in {"BUILT", "REUSED"}
    }
    assert not failed, failed
    assert len(portfolio_ids) == 6
    statistics_identity = results[statistics_id]["instance_identity"]
    assert statistics_identity is not None
    assert results[prediction_id]["status"] == "BUILT"
    champion_identity = results[prediction_id]["instance_identity"]
    champion_root = layout.instance_root(
        layout.semantic_artifact_root(
            "a_share",
            prediction_id,
        ),
        champion_identity,
    )
    prediction_manifest, _prediction_members = validate_output_bundle(
        champion_root / "outputs" / prediction_id,
        expected_output_id=prediction_id,
        expected_run_id=f"run_{prediction_id}",
    )
    assert (
        prediction_manifest["metadata"]["prediction_metadata"]["test_start"]
        == "2019-01-01"
    )
    prediction_manifest = _prediction_members = None
    statistics_root = layout.instance_root(
        layout.semantic_artifact_root("a_share", statistics_id),
        statistics_identity,
    )
    statistics_artifact = research.read_statistics_artifact(
        statistics_root, expected_identity=statistics_identity
    )
    with statistics_artifact.table("prediction_diagnostics.parquet") as table:
        observations = frames["pv_all"]
        mature_dates = observations.loc[
            observations.timestamp.between(request["start"], request["end"])
            & observations.forward_return_20d_available_at.le(
                pd.Timestamp(request["end"])
            ),
            "timestamp",
        ].nunique()
        assert mature_dates == 23
        assert table.observation_count.eq(mature_dates).all()
    statistics_artifact = None
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

    bundle = champion_root / "outputs" / prediction_id
    damaged = tmp_path / "missing_family"
    shutil.copytree(native_filesystem_path(bundle), native_filesystem_path(damaged))
    values = pd.read_parquet(native_filesystem_path(damaged / "prediction.parquet"))
    missing_family = "pv_all_ridge_regression"
    values = values.loc[~values.family_id.eq(missing_family)]
    values.to_parquet(
        native_filesystem_path(damaged / "prediction.parquet"), index=False
    )
    manifest = json.loads(
        filesystem_read_text(damaged / "manifest.json", encoding="utf-8")
    )
    manifest["metadata"]["prediction_metadata"]["prediction_rows"] = len(values)
    filesystem_write_text(
        damaged / "manifest.json", json.dumps(manifest), encoding="utf-8"
    )
    model_root = layout.instance_root(
        layout.semantic_artifact_root("a_share", protocol.model_selection_target_id),
        results[protocol.model_selection_target_id]["instance_identity"],
    )
    model = research.read_quarterly_model_artifact(model_root, trial_layout=layout)
    global_decision = model.selections["global"]["final"]
    quarterly_decision = model.selections["quarterly"]["2019Q1"]
    assert global_decision["validation_start"] == "2019-01-01"
    assert global_decision["validation_end"] == "2019-03-01"
    assert pd.Timestamp(quarterly_decision["validation_end"]) < pd.Timestamp(
        "2019-01-01"
    )
    for decision in (global_decision, quarterly_decision):
        assert set(decision["candidate_family_ids"]) == model_ids
        assert decision["common_validation_coverage"]["row_count"] > 0
        assert len(decision["families"]) == 2
        assert all(family["trial_metrics"] for family in decision["families"])
    assert len(model.candidate_models) == 10
    assert len(model.final_models) == 4
    for entry in (*model.candidate_models, *model.final_models):
        assert entry["model_metadata"]["rows_after_input_label_dropna"] > 0
    for entry in model.candidate_models:
        assert pd.Timestamp(entry["model_metadata"]["train_end"]) < pd.Timestamp(
            entry["validation_window"]["start"]
        )
    model = global_decision = quarterly_decision = None
    data_layout = layout.with_target_roots(
        {
            "pv_all": layout.instance_root(
                layout.research_dataset_root("a_share", "pv_all"),
                results["pv_all"]["instance_identity"],
            )
        }
    )
    from quant_research_platform.artifacts.statistics import write_statistics_artifact
    from quant_research_platform.contracts.business_manifest import (
        build_business_manifest_envelope,
    )

    for end_date in ("2019-01-15", "2019-02-15"):
        narrow_window = TimeWindow("2019-01-02", end_date)
        narrow_global = None
        model = research.read_quarterly_model_artifact(model_root, trial_layout=layout)
        for index, group in enumerate(("global", "quarterly")):
            group_prediction_id = protocol.prediction_target_ids[index]
            group_statistics_id = protocol.statistics_target_ids[index]
            group_root = layout.instance_root(
                layout.semantic_artifact_root("a_share", group_prediction_id),
                results[group_prediction_id]["instance_identity"],
            )
            (_manifest, lookup) = read_prediction_output_bundle(
                group_root / "outputs" / group_prediction_id,
                expected_output_id=group_prediction_id,
                time_window=narrow_window,
            )
            narrow = build_a_share_statistics_tables(
                layout=data_layout,
                dataset_catalog=profile.research_datasets,
                dataset_id="pv_all",
                prediction_id=group_prediction_id,
                persisted_model=model,
                protocol_manifest=protocol.target_manifest(
                    group_statistics_id, "STATISTIC"
                ),
                time_window=narrow_window,
                prediction_lookup=lookup,
                global_statistics=narrow_global,
            )
            prediction_diagnostics = narrow["prediction_diagnostics"].iloc[0]
            assert prediction_diagnostics["status"] == "UNDEFINED"
            if end_date == "2019-01-15":
                assert prediction_diagnostics["observation_count"] == 0
            else:
                assert 0 < prediction_diagnostics["observation_count"] < 20
            robustness = narrow["robustness"]
            assert robustness.status.eq("UNDEFINED").all()
            assert robustness.observation_count.lt(20).all()
            if group == "global":
                reason = "global_development_selection_reuses_validation"
                assert prediction_diagnostics["reason"] == reason
                assert robustness.reason.eq(reason).all()
                narrow_envelope = build_business_manifest_envelope(
                    **{
                        key: value
                        for key, value in read_semantic_artifact(
                            statistics_root
                        ).items()
                        if key
                        not in {
                            "instance_identity",
                            "status",
                            "logical_artifact_id",
                            "research_window",
                        }
                    },
                    logical_artifact_id=f"pv_all_global_narrow_{end_date}_statistics",
                    research_window={
                        "start": "2019-01-02",
                        "end": end_date,
                        "access": "development",
                    },
                )
                narrow_root = write_statistics_artifact(
                    tmp_path / f"narrow_global_statistics_{end_date}",
                    envelope=narrow_envelope,
                    tables=narrow,
                )
                narrow_global = research.read_statistics_artifact(
                    narrow_root, expected_identity=narrow_envelope.instance_identity
                )
            else:
                assert (
                    prediction_diagnostics["reason"] == "hac_lag_exceeds_observations"
                )
                assert robustness.reason.eq(
                    "bootstrap_block_exceeds_common_observations"
                ).all()
                paired = narrow["model_group_comparison"]
                assert set(paired.family_id) == {
                    item["model_id"] for item in configuration["models"]
                }
                if end_date == "2019-01-15":
                    assert paired.observation_count.eq(0).all()
                    assert paired.reason.eq("no_common_observations").all()
                    assert paired.bootstrap_reason.eq("no_common_observations").all()
                    assert paired.status.eq("UNDEFINED").all()
                    narrow_payload = narrow_envelope.payload()
                    quarter_envelope = build_business_manifest_envelope(
                        **{
                            key: value
                            for key, value in narrow_payload.items()
                            if key
                            not in {
                                "instance_identity",
                                "status",
                                "logical_artifact_id",
                            }
                        },
                        logical_artifact_id="pv_all_quarterly_zero_mature_statistics",
                    )
                    quarterly_root = write_statistics_artifact(
                        tmp_path / "quarterly_zero_mature_statistics",
                        envelope=quarter_envelope,
                        tables=narrow,
                    )
                    saved_quarter = research.read_statistics_artifact(
                        quarterly_root,
                        expected_identity=quarter_envelope.instance_identity,
                    )
                    with saved_quarter.table("model_group_comparison.parquet") as saved:
                        assert saved.observation_count.eq(0).all()
                        assert saved.status.eq("UNDEFINED").all()
                        assert saved.reason.eq("no_common_observations").all()
        model = narrow_global = saved_quarter = None
        narrow_envelope = quarter_envelope = narrow_payload = None
    model = research.read_quarterly_model_artifact(model_root, trial_layout=layout)
    (_manifest, lookup) = read_prediction_output_bundle(
        damaged,
        expected_output_id=prediction_id,
        time_window=TimeWindow(request["start"], request["end"]),
    )
    with pytest.raises(ValueError, match="missing_required_prediction_keys"):
        build_a_share_statistics_tables(
            layout=data_layout,
            dataset_catalog=profile.research_datasets,
            dataset_id="pv_all",
            prediction_id=prediction_id,
            persisted_model=model,
            protocol_manifest=protocol.target_manifest(statistics_id, "STATISTIC"),
            time_window=TimeWindow(request["start"], request["end"]),
            prediction_lookup=lookup,
            global_statistics=None,
        )
    model = None

    def forged_writer(staging):
        shutil.copytree(
            native_filesystem_path(bundle),
            native_filesystem_path(staging / "outputs" / prediction_id),
        )
        path = staging / "outputs" / prediction_id / "manifest.json"
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
        if target.target_id == prediction_id
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
            with closing_iterator(iter(frames)) as stream:
                first = next(stream)
                observed.append(len(first))
                yield first
                raise primary

        return native_write(interrupted_frames(), path, **kwargs)

    with monkeypatch.context() as patch:
        model = research.read_quarterly_model_artifact(model_root, trial_layout=layout)
        patch.setattr(prediction_writer, "write_parquet_chunks", fail_merged_write)
        pspec = PredictionSpec(
            prediction_id=prediction_id,
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
            selection_group="global",
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
                )
        assert caught.value is primary
        assert observed
        output = model = None
    assert (
        Path(native_filesystem_path(bundle / "prediction.parquet")).read_bytes()
        == original_bytes
    )
    requested_years = {}
    for target_id in portfolio_ids:
        identity = results[target_id]["instance_identity"]
        root = layout.instance_root(
            layout.semantic_artifact_root("a_share", target_id),
            identity,
        )
        artifact = research.read_portfolio_ledger_artifact(
            root, expected_identity=identity
        )
        requested_years[target_id.removeprefix("pv_all_").removesuffix("_ledger")] = (
            set()
        )
        with closing_iterator(artifact.iter_chunks()) as stream:
            for batch in stream:
                demanded = {
                    item.asset_id
                    for item in batch.diagnostics
                    if item.name == "decision_delta_weight"
                    and item.asset_id != "__cash__"
                    and item.number != 0.0
                }
                requested_years[
                    target_id.removeprefix("pv_all_").removesuffix("_ledger")
                ].update(
                    str(pd.Timestamp(item.text).year)
                    for item in batch.diagnostics
                    if item.name == "decision_timestamp" and item.asset_id in demanded
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
        artifact = None

    reused = research.research_run(
        profile=profile,
        layout=layout,
        research_plan=configuration,
    )
    reused_results = {item["target_id"]: item for item in reused["target_results"]}
    assert result["run_id"] != reused["run_id"]
    assert all(
        reused_results[target_id]["status"] == "REUSED"
        for target_id in (
            *portfolio_ids,
            *protocol.prediction_target_ids,
        )
    )

    assert {
        item["target_id"]: item["instance_identity"]
        for item in result["target_results"]
    } == {
        item["target_id"]: item["instance_identity"]
        for item in reused["target_results"]
    }
    from quant_research_platform.artifacts import portfolio_ledger

    scanned = []
    original_batches = portfolio_ledger._iter_batches

    def count_batches(root, *args):
        scanned.append(root)
        yield from original_batches(root, *args)

    with monkeypatch.context() as report_only:
        report_only.setattr(portfolio_ledger, "_iter_batches", count_batches)
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
    assert scanned and len(scanned) == len(set(scanned))
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
        report_root, expected_identity=report_identity
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
    assert report["sections"]["factor diagnostics"]["global"]["results"]
    assert report["sections"]["model selection"]["selections"]
    assert report["sections"]["model selection"]["configurations"]
    blocked_ids = {
        "fundamental_factors",
        *(target_id for target_id in portfolio_ids if "enhanced" in target_id),
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

    model = research.read_quarterly_model_artifact(model_root, trial_layout=layout)
    partial_report = produce_a_share_report(
        layout=layout, record=partial, selections=model.selections
    )
    assert partial_report["report_status"] == "BLOCKED"
    retained = partial_report["sections"]["portfolio"]["results"]
    assert len(retained) == 4
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
    expected_capacity_keys = {
        (candidate, participation, period, metric)
        for candidate, years in requested_years.items()
        for participation in protocol.manifest()["execution_assumptions"][
            "adv_sensitivities"
        ]
        for period in {"all", *years}
        for metric in ("p05", "median", "minimum")
    }
    assert {
        (row["candidate_id"], row["participation"], row["period"], row["metric_id"])
        for row in capacity["results"]
    } == expected_capacity_keys
    assert len(capacity["results"]) == len(expected_capacity_keys)
    assert len(undefined_capacity) == 2 * 4 * 3
    assert {row["candidate_id"] for row in undefined_capacity} == {
        candidate.candidate_id
        for candidate in protocol.portfolio_candidates()
        if candidate.construction == "benchmark_enhanced"
    }
    assert all(
        row["value"] is not None and row["reason"] == "" for row in defined_capacity
    )
    assert all(
        row["value"] is None and row["reason"] == "no_rebalance_requests"
        for row in undefined_capacity
    )
    assert report["sections"]["risk"]["status"] == "BLOCKED"
    assert report["sections"]["attribution"]["status"] == "BLOCKED"

    report = partial_report = coverage = retained = capacity = defined_capacity = (
        undefined_capacity
    ) = model = None

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
    filesystem_rename(statistics_manifest, saved_statistics_manifest)
    try:
        damaged_report = research.report_build(
            profile=profile,
            layout=layout,
            run_id=reused["run_id"],
        )
    finally:
        filesystem_rename(saved_statistics_manifest, statistics_manifest)
    assert damaged_report["source_run_status"] == "COMPLETE"
    assert damaged_report["status"] == "FAILED"
    assert damaged_report["report"]["sections"]["factor diagnostics"]["global"] == {
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
            "end": "2019-03-01",
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
        first[prediction_id]["requested_identity"]
        == second[prediction_id]["requested_identity"]
    )
