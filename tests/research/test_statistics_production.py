from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.catalog.datasets import ResearchDatasetCatalog
from quant_research_platform.catalog.evaluations import EvaluationCatalog
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.experiments import ExperimentFold
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.contracts.targets import unmeasured_operation_evidence
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research import statistics_production
from quant_research_platform.research.evaluations.runner import (
    build_evaluation_result_from_research_dataset,
)
from quant_research_platform.research.models.persisted import (
    read_quarterly_model_artifact,
    write_quarterly_model_artifact,
)
from quant_research_platform.research.models.production import (
    QuarterlyModelResult,
    _fit_selected_model_result,
)
from quant_research_platform.research.models.supervised import preflight_model_specs
from quant_research_platform.research.protocols import a_share
from quant_research_platform.research.randomness import derive_seed
from quant_research_platform.research.statistics_production import (
    build_a_share_statistics_tables,
)
from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.storage.datasets import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from tests.helpers.research_configuration import (
    A_SHARE_MODEL_ALGORITHMS,
    CORE6_FACTORS,
    PV25_FACTORS,
    research_configuration,
)
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.helpers.runtime_resources import capture_test_policy
from tests.research.workflow_test_support import _selection_metadata, _trial_reference


def _statistics_models(layout, spec, protocol, window, *, config_suffix=""):
    factor = spec.factor_columns[0]
    models = tuple(
        ModelSpec(
            model_id=f"{spec.data_id}_{algorithm}{config_suffix}",
            model_algorithm=algorithm,
            input_factor_columns=(factor,),
            label_column=a_share.PRIMARY_LABEL_COLUMN,
        )
        for algorithm in A_SHARE_MODEL_ALGORITHMS
    )
    preflight_model_specs(models)
    families = tuple(f"{spec.data_id}_{model.model_algorithm}" for model in models)
    protocol["research_configuration"] = {
        **research_configuration(),
        "evaluation": {
            "factors": list(spec.factor_columns),
            "labels": [a_share.PRIMARY_LABEL_COLUMN],
            "primary_label": a_share.PRIMARY_LABEL_COLUMN,
        },
        "comparison_model_id": families[0],
        "models": [
            {"model_id": family, "hyperparameters": dict(model.hyperparameters)}
            for family, model in zip(families, models, strict=True)
        ],
    }
    year = pd.Timestamp(window.start).year
    training_calendar = pd.bdate_range(f"{year - 1}-01-02", periods=53)
    rng = np.random.default_rng(101)
    training = pd.DataFrame(
        {
            "market_id": "a_share",
            "asset_id": [f"asset_{i}" for _ in range(32) for i in range(6)],
            "timestamp": training_calendar[:32].repeat(6),
            "eligible": True,
            factor: rng.normal(size=192),
            "forward_return_20d_available_at": training_calendar[21:].repeat(6),
        }
    )
    training[a_share.PRIMARY_LABEL_COLUMN] = training[factor] * 0.3 + rng.normal(
        scale=0.01, size=192
    )

    def chunks(columns, **_kwargs):
        yield training.loc[:, list(columns)]

    def fitted():
        for period in pd.period_range(window.start, window.end, freq="Q"):
            fold = ExperimentFold(
                fold_id=f"{period.year}-q{period.quarter}",
                train=TimeWindow(f"{year - 5}-01-01", f"{year - 2}-12-31"),
                validation=TimeWindow(f"{year - 1}-01-01", f"{year - 1}-12-31"),
                development_test=TimeWindow(
                    period.start_time.date().isoformat(),
                    period.end_time.date().isoformat(),
                ),
            )
            references = tuple(
                _trial_reference(
                    layout,
                    fold,
                    family_id=family,
                    model_id=model.model_id,
                    dataset_id=spec.data_id,
                    protocol=protocol,
                    model_spec=model,
                    root_seed=42,
                )
                for family, model in zip(families, models, strict=True)
            )
            selection = _selection_metadata(
                family_id=families[0], model_id=models[0].model_id, columns=(factor,)
            )
            selection.update(
                validation_start=f"{year - 1}-01-01",
                validation_end=f"{year - 1}-12-31",
                candidate_family_ids=list(families),
                tie_break_order=list(families),
                families=[
                    {
                        "family_id": family,
                        "config_tie_break_order": [model.model_id],
                        "selected_config_id": model.model_id,
                        "trial_ledger_semantic_identity": reference[
                            "semantic_identity"
                        ],
                    }
                    for family, model, reference in zip(
                        families, models, references, strict=True
                    )
                ],
            )
            for family, model, reference in zip(
                families, models, references, strict=True
            ):
                result, lease = _fit_selected_model_result(
                    training_counts={},
                    dataset_spec=spec,
                    selected_spec=model,
                    candidate_parameters=dict(model.hyperparameters),
                    chunks=chunks,
                    fitting_cutoff=f"{year - 1}-12-31",
                    quarter_id=str(period),
                    root_seed=42,
                )
                owned = OwnedValue(
                    QuarterlyModelResult(
                        fold, model.model_id, result, family, selection, (reference,)
                    ),
                    lease,
                )
                result = None
                try:
                    yield owned
                finally:
                    if owned.lease.active:
                        owned.release(owner="research.model.artifact")

    envelope = build_business_manifest_envelope(
        logical_artifact_id="model-artifact",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": str(pd.Timestamp(window.start).date()),
            "end": str(pd.Timestamp(window.end).date()),
            "access": "development",
        },
        upstream_identities={spec.data_id: "d" * 64},
        semantic_metadata={
            "target_contract": {
                "protocol": protocol,
                "model_specs": [{"model_id": family} for family in families],
            }
        },
    )
    with runtime_resource_policy_scope():
        root = write_quarterly_model_artifact(
            layout.semantic_artifact_root("a_share", "model-artifact"),
            model_id="model-artifact",
            envelope=envelope,
            quarterly_results=fitted(),
            trial_layout=layout,
        )
        return read_quarterly_model_artifact(root, trial_layout=layout)


class _PredictionLookup:
    def coverage_counts(self):
        return (0, 0)

    @contextmanager
    def values_for_frame(self, frame: pd.DataFrame, *, mark_matched: bool):
        assert mark_matched
        assets = frame["asset_id"].str.removeprefix("asset_").astype(float).to_numpy()
        signs = np.where(frame["timestamp"].dt.day.to_numpy() % 2, -1.0, 1.0)
        values = assets * signs
        yield values, np.ones(len(frame), dtype=bool)


@pytest.mark.parametrize("full_factor_history", (False, True))
def test_family_comparison_production_matches_arch_on_persisted_common_date_losses(
    tmp_path: Path,
    full_factor_history: bool,
):
    from arch.bootstrap import MCS, SPA, StepM

    layout = StorageLayout(tmp_path)
    protocol = deepcopy(a_share.protocol("pv_all").manifest())
    protocol["statistics"]["factor_fdr_families"] = {"fixture": ["x"]}
    protocol["statistics"]["fama_macbeth_core_factors"] = ["x"]
    families = tuple(f"pv_all_{algorithm}" for algorithm in A_SHARE_MODEL_ALGORITHMS)
    dates = pd.bdate_range("2019-01-01", periods=64)
    label_calendar = pd.bdate_range(dates[0], periods=len(dates) + 21)
    cutoff = label_calendar[-1]
    assets = [f"asset_{i}" for i in range(12)]
    rng = np.random.default_rng(901)
    frame = pd.DataFrame(
        {
            "timestamp": dates.repeat(len(assets)),
            "asset_id": assets * len(dates),
            "market_id": "a_share",
            "eligible": True,
            "x": rng.normal(size=len(dates) * len(assets)),
            a_share.PRIMARY_LABEL_COLUMN: np.tile(
                np.arange(len(assets), dtype=float), len(dates)
            ),
            "forward_return_20d_available_at": label_calendar[21:].repeat(len(assets)),
        }
    )
    scores = rng.normal(size=(len(dates), len(assets), len(families)))
    scores = (np.argsort(np.argsort(scores, axis=1), axis=1) + 0.5) / len(assets) - 0.5
    lookup_values = {
        (timestamp, asset, family): scores[day, index, model]
        for day, timestamp in enumerate(dates)
        for index, asset in enumerate(assets)
        for model, family in enumerate(families)
    }

    class Lookup:
        def __init__(self):
            self.matched = set()

        @contextmanager
        def values_for_frame(self, keys, *, mark_matched):
            coordinates = tuple(
                keys.loc[:, ["timestamp", "asset_id", "family_id"]].itertuples(
                    index=False, name=None
                )
            )
            assert mark_matched
            self.matched.update(coordinates)
            yield (
                np.asarray([lookup_values[key] for key in coordinates]),
                np.ones(len(coordinates), dtype=bool),
            )

        def coverage_counts(self):
            return len(set(lookup_values) - self.matched), len(
                self.matched - set(lookup_values)
            )

    spec = make_demo_research_dataset_spec(
        data_id="pv_all",
        factor_columns=("x",),
        label_columns=(a_share.PRIMARY_LABEL_COLUMN, "forward_return_20d_available_at"),
        filter_columns=("eligible",),
        context_columns=(),
    )
    if full_factor_history:
        historical = frame.iloc[: len(assets)].copy()
        historical["timestamp"] = pd.Timestamp("2014-01-02")
        historical["forward_return_20d_available_at"] = pd.Timestamp("2014-02-03")
        frame = pd.concat((historical, frame), ignore_index=True)
        protocol["development_window"] = {
            "start": dates[0].date().isoformat(),
            "end": cutoff.date().isoformat(),
        }
    write_dataset(layout.research_dataset_root("a_share", "pv_all"), frame, spec=spec)
    model = _statistics_models(layout, spec, protocol, TimeWindow(dates[0], cutoff))
    envelope = build_business_manifest_envelope(
        logical_artifact_id="pv_all_statistics",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": "2019-01-01",
            "end": str(cutoff.date()),
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={},
    )
    with runtime_resource_policy_scope() as policy:
        owned = build_a_share_statistics_tables(
            layout=layout,
            dataset_catalog=ResearchDatasetCatalog((spec,)),
            dataset_id="pv_all",
            prediction_id="prediction",
            persisted_model=model,
            protocol_manifest=protocol,
            time_window=TimeWindow(dates[0], cutoff),
            prediction_lookup=Lookup(),
            resource_policy=policy,
        )
        try:
            root = write_statistics_artifact(
                layout.semantic_artifact_root("a_share", "pv_all_statistics"),
                envelope=envelope,
                tables=owned.value,
            )
        finally:
            owned.release(owner="research.statistics.output")
        artifact = read_statistics_artifact(
            root, expected_identity=envelope.instance_identity
        )
        with artifact.table("factor_diagnostics.parquet") as factors:
            assert factors.observation_count.eq(65 if full_factor_history else 64).all()
        with artifact.table("model_losses.parquet") as table:
            assert len(table) == 64 * 8
            assert table.common_date.eq(1).all()
            losses = (
                table.pivot(index="timestamp", columns="family_id", values="loss")
                .loc[:, list(families)]
                .copy()
            )
        expected = pd.DataFrame(
            {
                family: [
                    -pd.Series(scores[day, :, model_index]).corr(
                        pd.Series(np.arange(len(assets))), method="spearman"
                    )
                    for day in range(len(dates))
                ]
                for model_index, family in enumerate(families)
            },
            index=dates,
        )
        np.testing.assert_allclose(losses, expected, rtol=0, atol=1e-14)
        benchmark, challengers = losses[families[0]], losses.loc[:, list(families[1:])]
        with artifact.table("model_reality_checks.parquet") as checks:
            assert checks.status.eq("DEFINED").all()
            assert checks.observation_count.eq(64).all()
            for test_id, constructor in (
                ("spa_reality_check", SPA),
                ("stepm_superior_models", StepM),
                ("model_confidence_set", MCS),
            ):
                seed = derive_seed(
                    42, "a_share_model_reality_check", "pv_all", "development", test_id
                )
                options = dict(block_size=20, reps=10000, seed=seed)
                native = (
                    constructor(losses, size=0.05, **options)
                    if constructor is MCS
                    else constructor(benchmark, challengers, size=0.05, **options)
                    if constructor is StepM
                    else constructor(benchmark, challengers, **options)
                )
                native.compute()
                rows = checks.loc[checks.test_id.eq(test_id)]
                assert rows.seed.eq(seed).all()
                if constructor is SPA:
                    for label in ("lower", "consistent", "upper"):
                        assert (
                            rows.loc[
                                rows.statistic_id.eq(label + "_p_value"), "value"
                            ].iloc[0]
                            == native.pvalues[label]
                        )
                elif constructor is StepM:
                    assert set(rows.loc[rows.value.eq(1), "family_id"]) == set(
                        native.superior_models
                    )
                else:
                    for family in families:
                        assert (
                            rows.loc[
                                rows.family_id.eq(family)
                                & rows.statistic_id.eq("p_value"),
                                "value",
                            ].iloc[0]
                            == native.pvalues.loc[family].iloc[0]
                        )
                    assert set(
                        rows.loc[
                            rows.statistic_id.eq("included") & rows.value.eq(1),
                            "family_id",
                        ]
                    ) == set(native.included)
        with artifact.table("model_comparison.parquet") as table:
            assert len(table) == 7
            assert table.observation_count.eq(64).all()
            for row in table.itertuples():
                assert row.mean == pytest.approx(
                    float((losses[families[0]] - losses[row.prediction_id]).mean())
                )
        assert policy.used_memory_bytes == 0
    from quant_research_platform.artifacts.research import (
        load_a_share_run_evidence,
        read_research_report_artifact,
        write_research_report_artifact,
    )
    from quant_research_platform.contracts.targets import (
        ProducerKind,
        ResearchRunRecord,
        ResearchRunTargetResult,
        RunStatus,
        TargetStatus,
    )
    from quant_research_platform.reports.a_share import (
        REQUIRED_REPORT_SECTIONS,
        build_a_share_report,
    )

    record = ResearchRunRecord(
        operation_evidence=unmeasured_operation_evidence("fixture_record"),
        run_id="run_statistics",
        profile_id="fixture",
        scenario_id="pv_all",
        protocol=protocol,
        protocol_identity=canonical_json_hash(protocol),
        research_window=envelope.research_window,
        software_identity="c" * 64,
        terminal_targets=("pv_all_statistics",),
        target_results=(
            ResearchRunTargetResult(
                target_id="pv_all_statistics",
                producer_kind=ProducerKind.STATISTIC,
                status=TargetStatus.BUILT,
                instance_identity=envelope.instance_identity,
                schema_identity=envelope.schema_identity,
                artifact_root="data/artifacts/a_share/pv_all_statistics",
            ),
        ),
        status=RunStatus.COMPLETE,
    )
    from quant_research_platform.artifacts.research import report_payload_working_set

    with (
        runtime_resource_policy_scope() as report_policy,
        report_policy.lease(
            "reports.test_payload", report_payload_working_set(record.payload())
        ) as report_lease,
    ):
        evidence = load_a_share_run_evidence(
            layout=layout,
            record=record,
            report_lease=report_lease,
            quarterly_selections={},
        )
        report = build_a_share_report(record=record, evidence=evidence)
        projected = deepcopy(evidence)
        projected["issues"] = []
        projected["portfolio_statistics"] = {
            "status": "UNDEFINED",
            "capacity": [{"status": "UNDEFINED", "reason": "no_rebalance_requests"}],
            "performance": [],
            "attribution": [],
            "robustness": [],
        }
        unavailable = build_a_share_report(record=record, evidence=projected)
        assert unavailable["sections"]["capacity"]["status"] == "UNDEFINED"
        assert unavailable["report_status"] == "COMPLETE"
        projected["portfolio_statistics"]["performance"] = [
            {
                "metric_id": "tracking_error",
                "status": "BLOCKED",
                "reason": "benchmark_total_return_open_missing",
            }
        ]
        unavailable = build_a_share_report(record=record, evidence=projected)
        assert unavailable["sections"]["risk"]["status"] == "BLOCKED"
        assert unavailable["report_status"] == "BLOCKED"
        assert unavailable["source_run_status"] == "COMPLETE"
        actual = report["sections"]["OOS prediction"]["reality_checks"]
        assert len(actual) == 26 and all(
            row["status"] == "DEFINED" and row["observation_count"] == 64
            for row in actual
        )
        source_identity = canonical_json_hash(record.payload())
        contract = {
            "schema_id": "qrp-a-share-research-report",
            "required_sections": list(REQUIRED_REPORT_SECTIONS),
            "source_run_identity": source_identity,
        }
        report_envelope = build_business_manifest_envelope(
            logical_artifact_id="pv_all_report_artifact",
            schema_identity=canonical_json_hash(contract),
            protocol_or_spec_identity="a" * 64,
            software_identity="c" * 64,
            research_window=envelope.research_window,
            upstream_identities={"source_run": source_identity},
            semantic_metadata={
                "target_contract": contract,
                "source_run_id": record.run_id,
                "source_run_status": "COMPLETE",
                "report_status": report["report_status"],
            },
        )
        report_root = write_research_report_artifact(
            tmp_path / "report", envelope=report_envelope, payload=report
        )
        saved_report = read_research_report_artifact(
            report_root, expected_identity=report_envelope.instance_identity
        )
        assert saved_report["sections"]["OOS prediction"]["reality_checks"] == actual
    assert report_policy.used_memory_bytes == 0


def test_empty_fama_macbeth_periods_do_not_invent_dates_or_coefficients() -> None:
    periods = pd.DataFrame(columns=statistics_production._FAMA_MACBETH_PERIOD_COLUMNS)
    summary = statistics_production._fama_macbeth_summary(
        periods,
        factors=("factor",),
        core_factors=("factor",),
        max_lags=19,
    )

    assert periods.empty
    assert set(summary["regression_id"]) == {"single:factor", "core_multifactor"}
    assert set(summary["coefficient_id"]) == {"intercept", "factor"}
    assert summary["period_count"].eq(0).all()
    assert summary["white_defined_period_count"].eq(0).all()
    assert summary["white_undefined_period_count"].eq(0).all()
    assert summary["status"].eq("UNDEFINED").all()


def test_statistics_excludes_immature_labels_and_keeps_fixed_family_fdr(
    tmp_path: Path,
) -> None:
    outputs = []
    evaluation_summaries = []
    for name, immature_sign in (("first", 1.0), ("changed", -1.0), ("zero", 1.0)):
        root = tmp_path / name
        layout = StorageLayout(project_root=root)
        dates = pd.bdate_range("2023-01-02", periods=25)
        label_calendar = pd.bdate_range(dates[0], periods=len(dates) + 21)
        cutoff = label_calendar[40]
        rows = []
        for date_index, timestamp in enumerate(dates):
            mature = date_index < 20
            for asset_index in range(6):
                x = float(asset_index - 2)
                label = (
                    0.1
                    + 0.4 * x
                    + 0.03 * x * x
                    + 0.08 * np.sin((asset_index + 1) * (date_index + 1))
                )
                # Five-day blocks vary independently of the daily prediction sign.
                if (date_index // 5) % 2:
                    label *= -1.0
                if not mature:
                    label *= immature_sign
                if name == "zero":
                    label = 0.0
                rows.append(
                    {
                        "market_id": "a_share",
                        "asset_id": f"asset_{asset_index}",
                        "timestamp": timestamp,
                        **{factor: np.nan for factor in PV25_FACTORS},
                        CORE6_FACTORS[0]: x,
                        CORE6_FACTORS[1]: x * x,
                        a_share.PRIMARY_LABEL_COLUMN: label,
                        "forward_return_20d_available_at": label_calendar[
                            date_index + 21
                        ],
                        "eligible": True,
                    }
                )
        spec = make_demo_research_dataset_spec(
            data_id="stats_demo",
            factor_columns=PV25_FACTORS,
            label_columns=(
                a_share.PRIMARY_LABEL_COLUMN,
                "forward_return_20d_available_at",
            ),
            filter_columns=("eligible",),
            context_columns=(),
            filter_column_roles={"eligible": ColumnRole.UNIVERSE},
        )
        write_dataset(
            layout.research_dataset_root(spec.market_id, spec.data_id),
            pd.DataFrame(rows),
            spec=spec,
        )
        evaluation_policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
        evaluation = build_evaluation_result_from_research_dataset(
            layout=layout,
            dataset_catalog=ResearchDatasetCatalog((spec,)),
            evaluation_catalog=EvaluationCatalog(
                (
                    EvaluationSpec(
                        evaluation_id="stats_demo_evaluation",
                        market_id=MarketId("a_share"),
                        research_dataset_id=spec.data_id,
                        signals=(
                            SignalSpec(
                                signal_id=CORE6_FACTORS[0],
                                source_kind="factor",
                                source_column=CORE6_FACTORS[0],
                            ),
                        ),
                        label_columns=(a_share.PRIMARY_LABEL_COLUMN,),
                        label_availability_columns={
                            a_share.PRIMARY_LABEL_COLUMN: (
                                "forward_return_20d_available_at"
                            )
                        },
                    ),
                )
            ),
            evaluation_id="stats_demo_evaluation",
            time_window=TimeWindow(dates[0], cutoff),
            resource_policy=evaluation_policy,
        )
        try:
            for _output in evaluation.output_chunks:
                pass
            evaluation_summaries.append(evaluation.summaries()[1].copy())
        finally:
            evaluation.close()
        protocol = deepcopy(a_share.protocol("pv_all").manifest())
        protocol["statistics"]["factor_fdr_families"] = {
            "price_volume": list(PV25_FACTORS)
        }
        protocol["statistics"]["fama_macbeth_core_factors"] = list(CORE6_FACTORS)
        model = _statistics_models(
            layout, spec, protocol, TimeWindow(dates[0], cutoff), config_suffix="_0"
        )
        policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
        with runtime_resource_policy_scope(policy):
            owned_tables = build_a_share_statistics_tables(
                layout=layout,
                dataset_catalog=ResearchDatasetCatalog((spec,)),
                dataset_id=spec.data_id,
                prediction_id="prediction",
                persisted_model=model,
                protocol_manifest=protocol,
                time_window=TimeWindow(dates[0], cutoff),
                prediction_lookup=_PredictionLookup(),
                resource_policy=policy,
            )
            try:
                outputs.append(
                    {name: table.copy() for name, table in owned_tables.value.items()}
                )
            finally:
                owned_tables.release(owner="research.statistics.output")

    for table in ("factor_diagnostics", "prediction_diagnostics", "fama_macbeth"):
        pd.testing.assert_frame_equal(
            outputs[0][table], outputs[1][table], check_exact=True
        )
    pd.testing.assert_frame_equal(
        evaluation_summaries[0], evaluation_summaries[1], check_exact=True
    )
    evaluation_row = evaluation_summaries[0].iloc[0]
    assert evaluation_row["valid_rankic_count"] == 20
    assert evaluation_row["mean_rankic"] == pytest.approx(0.0, abs=1e-15)
    factors = outputs[0]["factor_diagnostics"]
    assert set(factors["family_id"]) == {"price_volume"}
    assert set(factors["fdr_method"]) == {"BH", "BY"}
    assert set(factors["planned_hypothesis_count"]) == {len(PV25_FACTORS)}
    assert set(factors["defined_hypothesis_count"]) == {2}
    assert set(factors["missing_hypothesis_count"]) == {len(PV25_FACTORS) - 2}
    prediction = outputs[0]["prediction_diagnostics"].iloc[0]
    assert prediction["observation_count"] == 20
    assert prediction["status"] == "DEFINED"
    assert prediction["mean"] == pytest.approx(0.2, abs=1e-15)
    assert prediction["bootstrap_seed"] == derive_seed(
        42, "a_share_statistics_bootstrap", "pv_all", "prediction"
    )
    fmb = outputs[0]["fama_macbeth"]
    assert {"single:" + CORE6_FACTORS[0], "core_multifactor"}.issubset(
        set(fmb["regression_id"])
    )
    assert "intercept" in set(fmb["coefficient_id"])
    assert {0, 20}.issubset(set(fmb["period_count"]))
    trials = outputs[0]["model_trials"]
    assert trials.loc[0, ["quarter_id", "family_id", "config_id"]].tolist() == [
        "2023Q1",
        "stats_demo_equal_weight_rank_ensemble",
        "stats_demo_equal_weight_rank_ensemble_0",
    ]
    assert trials.loc[0, "selection_status"] == "CHAMPION"
    assert set(outputs[0]["robustness"]["reason"]) == {
        "bootstrap_block_exceeds_common_observations",
    }
    envelope = build_business_manifest_envelope(
        logical_artifact_id="stats_demo_statistics",
        schema_identity=canonical_json_hash({"statistics": "test"}),
        protocol_or_spec_identity="a" * 64,
        software_identity="b" * 64,
        research_window={
            "start": "2023-01-02",
            "end": str(cutoff.date()),
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={},
    )
    artifact_policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    with runtime_resource_policy_scope(artifact_policy):
        artifact_root = write_statistics_artifact(
            tmp_path / "statistics",
            envelope=envelope,
            tables=outputs[0],
        )
        artifact = read_statistics_artifact(
            artifact_root,
            expected_identity=envelope.instance_identity,
        )
    members = {
        "factor_diagnostics.parquet",
        "prediction_diagnostics.parquet",
        "model_trials.parquet",
        "robustness.parquet",
        "fama_macbeth.parquet",
        "fama_macbeth_periods.parquet",
        "model_losses.parquet",
        "model_comparison.parquet",
        "model_reality_checks.parquet",
    }
    for member in members:
        with artifact.table(member) as table:
            assert not table.empty
    zero_policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    with runtime_resource_policy_scope(zero_policy):
        zero_root = write_statistics_artifact(
            tmp_path / "statistics_zero_residual",
            envelope=envelope,
            tables=outputs[2],
        )
        zero_artifact = read_statistics_artifact(
            zero_root,
            expected_identity=envelope.instance_identity,
        )
    for member in members:
        with zero_artifact.table(member) as table:
            assert not table.empty
    with zero_artifact.table("fama_macbeth_periods.parquet") as table:
        periods = table.copy()
    single_factor = periods.loc[
        periods["regression_id"].eq("single:" + CORE6_FACTORS[0])
    ]
    assert single_factor["status"].eq("DEFINED").all()
    assert single_factor["value"].eq(0.0).all()
    assert single_factor["white_p_value"].isna().all()
    assert single_factor["white_reason"].eq("white_test_undefined").all()
    with zero_artifact.table("fama_macbeth.parquet") as table:
        zero_summary = table.copy()
    single_summary = zero_summary.loc[
        zero_summary["regression_id"].eq("single:" + CORE6_FACTORS[0])
    ]
    assert single_summary["white_defined_period_count"].eq(0).all()
    assert single_summary["white_undefined_period_count"].eq(20).all()
    empty_period_tables = dict(outputs[0])
    empty_period_tables["fama_macbeth_periods"] = empty_period_tables[
        "fama_macbeth_periods"
    ].iloc[:0]
    empty_policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    with runtime_resource_policy_scope(empty_policy):
        empty_root = write_statistics_artifact(
            tmp_path / "statistics_empty_periods",
            envelope=envelope,
            tables=empty_period_tables,
        )
        empty_artifact = read_statistics_artifact(
            empty_root,
            expected_identity=envelope.instance_identity,
        )
    with empty_artifact.table("fama_macbeth_periods.parquet") as table:
        assert table.empty
    capped = capture_test_policy(
        available_memory_bytes=64 * 1024**2, explicit_cap_bytes=1024
    )
    with runtime_resource_policy_scope(capped):
        with pytest.raises(MemoryError):
            with artifact.table("model_losses.parquet"):
                pytest.fail("decode should not be admitted")
        assert capped.used_memory_bytes == 0
    interrupted = capture_test_policy(available_memory_bytes=64 * 1024**2)
    primary = KeyboardInterrupt("table_consumer_interrupted")
    with runtime_resource_policy_scope(interrupted):
        with pytest.raises(KeyboardInterrupt) as caught:
            with artifact.table("model_losses.parquet") as table:
                assert not table.empty
                table = None
                raise primary
        assert caught.value is primary
        assert interrupted.used_memory_bytes == 0
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.storage.identity import native_filesystem_path
    from quant_research_platform.storage.semantic_artifacts import (
        write_semantic_artifact,
    )

    def wrong_schema(staging):
        for member in members:
            if member == "model_losses.parquet":
                pq.write_table(
                    pa.table({"wrong": [1]}), native_filesystem_path(staging / member)
                )
            else:
                shutil.copyfile(
                    native_filesystem_path(artifact_root / member),
                    native_filesystem_path(staging / member),
                )

    wrong_root = write_semantic_artifact(
        tmp_path / "bad_schema", envelope, wrong_schema
    )
    with pytest.raises(ValueError, match="statistics_table_schema_invalid"):
        read_statistics_artifact(
            wrong_root, expected_identity=envelope.instance_identity
        )


@pytest.mark.parametrize(
    "values", [[], [0.1], [np.nan], [np.nan, np.nan], [0.1, np.nan]]
)
def test_sample_risk_undefined_inputs_do_not_emit_finite_values_or_warnings(values):
    import warnings

    from quant_research_platform.research.statistics import return_standard_deviation

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = return_standard_deviation(values)
    assert np.isnan(result)


@pytest.mark.parametrize("value", [0.0, 0.01, -0.125])
def test_constant_risk_and_sharpe_have_distinct_definition_status(value):
    from quant_research_platform.research.statistics import (
        probabilistic_and_deflated_sharpe,
        return_standard_deviation,
    )

    # Exact constant observations have variance zero; Sharpe divides by that zero.
    assert return_standard_deviation([value] * 60) == 0
    with pytest.raises(ValueError, match="sharpe_zero_variance"):
        probabilistic_and_deflated_sharpe(
            [value] * 60, benchmark_sharpe=0.0, trial_sharpes=[0.1, 0.2]
        )
    # Hand-calculated sample variance for [-1, 0, 1] is 1.
    assert return_standard_deviation([-1.0, 0.0, 1.0]) == 1.0
