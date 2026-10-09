from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.statistics import (
    STATISTICS_TABLE_GROUPS,
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
from quant_research_platform.contracts.experiments import _model_inference_blocker
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
from quant_research_platform.research.protocols import a_share
from quant_research_platform.research.randomness import derive_seed
from quant_research_platform.research.statistics_production import (
    build_a_share_factor_statistics_tables,
    build_a_share_statistics_tables,
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
from tests.helpers.statistics_artifacts import (
    statistics_envelope as bind_statistics_envelope,
)
from tests.research.workflow_test_support import (
    shared_model_protocol,
)


def _statistics_model_selection_fixture(spec, protocol, window):
    factor = spec.factor_columns[0]
    configuration = research_configuration(small_models=True)
    models = tuple(
        ModelSpec(
            model_id=f"{spec.data_id}_{item['model_algorithm']}",
            model_algorithm=item["model_algorithm"],
            input_factor_columns=(factor,),
            label_column=a_share.PRIMARY_LABEL_COLUMN,
            hyperparameters=item["hyperparameters"],
        )
        for item in configuration["models"]
    )
    development_start = pd.Timestamp(window.start).replace(month=1, day=1)
    development_end = pd.Timestamp(window.end).date().isoformat()
    model_protocol = shared_model_protocol(models, development_end=development_end)
    model_protocol.update(
        scenario_id=spec.data_id,
        root_seed=protocol["root_seed"],
        development_window={
            "start": development_start.date().isoformat(),
            "end": development_end,
        },
        model_chronology={
            "scoring_lookback_months": 12,
            "warmup_window": {
                "start": (development_start - pd.DateOffset(years=1))
                .date()
                .isoformat(),
                "end": (development_start - pd.Timedelta(days=1)).date().isoformat(),
            },
        },
    )
    protocol.update(
        scenario_id=spec.data_id,
        selection_group="global",
        development_window=model_protocol["development_window"],
        research_configuration={
            **configuration,
            "evaluation": {
                "factors": list(spec.factor_columns),
                "labels": [a_share.PRIMARY_LABEL_COLUMN],
                "primary_label": a_share.PRIMARY_LABEL_COLUMN,
            },
            "comparison_model_id": models[0].model_id,
            "models": model_protocol["research_configuration"]["models"],
        },
    )
    families = []
    for model in models:
        metrics = {
            "model_id": model.model_id,
            "validation_rows": 12,
            "validation_ic": 0.1,
            "validation_rankic": 0.1,
            "validation_ic_reason": None,
            "validation_rankic_reason": None,
            "validation_mse": 1.0,
            **{
                f"validation_{key}": value
                for prefix in ("ic", "rankic")
                for key, value in (
                    (f"std_{prefix}", 0.0),
                    (f"{prefix}ir", None),
                    (f"positive_{prefix}_rate", 1.0),
                    (f"valid_{prefix}_count", 1),
                    (f"{prefix}ir_reason", "insufficient_observations"),
                )
            },
        }
        families.append(
            {
                "family_id": model.model_id,
                "selected_config_id": model.model_id,
                "trial_metrics": [metrics],
            }
        )
    selection = {"selected_family_id": models[0].model_id, "families": families}
    quarters = tuple(
        str(q) for q in pd.period_range(window.start, window.end, freq="Q")
    )

    def prediction_models(requested_window, **_kwargs):
        return tuple(
            {
                "quarter_id": str(quarter),
                "family_id": model.model_id,
                "model_id": model.model_id,
                "model_metadata": {
                    "input_columns": [factor],
                    "model_algorithm": model.model_algorithm,
                },
            }
            for quarter in pd.period_range(
                requested_window.start, requested_window.end, freq="Q"
            )
            for model in models
        )

    return SimpleNamespace(
        selections={
            "global": {"final": selection},
            "quarterly": {q: selection for q in quarters},
        },
        prediction_models=prediction_models,
        selection=lambda _group, _quarter, **_kwargs: selection,
    )


class _PredictionLookup:
    def coverage_counts(self):
        return (0, 0)

    def values_for_frame(self, frame: pd.DataFrame, *, mark_matched: bool):
        assert mark_matched
        assets = frame["asset_id"].str.removeprefix("asset_").astype(float).to_numpy()
        signs = np.where(frame["timestamp"].dt.day.to_numpy() % 2, -1.0, 1.0)
        values = assets * signs
        return (values, np.ones(len(frame), dtype=bool))


def test_fmb_primary_horizon_uses_native_models_predictions_and_maturity(tmp_path):
    from functools import partial

    from quant_research_platform.artifacts.outputs import (
        read_prediction_output_bundle,
        write_prediction_outputs,
    )
    from quant_research_platform.catalog.models import ModelCatalog
    from quant_research_platform.contracts.models import model_spec_payload
    from quant_research_platform.contracts.predictions import PredictionSpec
    from quant_research_platform.profiles.research import (
        A_SHARE_PROFILE_ID,
        bind_research_configuration,
        research_profile,
    )
    from quant_research_platform.research.models import persisted, production
    from quant_research_platform.research.predictions.runner import (
        build_prediction_result_from_persisted_model,
    )
    from quant_research_platform.storage.datasets.reading import open_dataset_session
    from quant_research_platform.workflows.research import _output_spec_for_target

    factor = CORE6_FACTORS[0]
    dates = pd.bdate_range("2023-01-02", periods=48)
    cutoff = dates[44]
    window = TimeWindow(dates[0].date(), cutoff.date())
    history = pd.to_datetime(
        ["2014-01-02", "2016-01-04", "2021-12-29"]
        + [f"2022-{month:02d}-{day:02d}" for month in (1, 4, 7, 10) for day in (4, 5)]
        + ["2022-03-31", "2022-06-30", "2022-09-30", "2022-12-30"]
    ).sort_values()
    frame = pd.DataFrame(
        [
            {
                "market_id": "a_share",
                "asset_id": f"asset_{asset}",
                "timestamp": timestamp,
                "eligible": True,
                factor: float(asset - 5.0 + index * 0.001),
                "forward_return_5d": 0.1
                + 0.4 * asset
                + 0.03 * asset**2
                + 0.02 * np.sin(index + asset),
                "forward_return_20d": -0.2 - 0.7 * asset + 0.02 * np.cos(index + asset),
                "forward_return_5d_available_at": timestamp + pd.offsets.BDay(6),
                "forward_return_20d_available_at": timestamp + pd.offsets.BDay(21),
            }
            for index, timestamp in enumerate(history.append(dates))
            for asset in range(12)
        ]
    )
    for horizon in (5, 20):
        label = f"forward_return_{horizon}d"
        unavailable = frame[f"{label}_available_at"].gt(cutoff)
        frame.loc[unavailable, label] = np.nan
        frame.loc[unavailable, f"{label}_available_at"] = pd.NaT
    spec = make_demo_research_dataset_spec(
        data_id="pv_all",
        factor_columns=(factor,),
        label_columns=(
            "forward_return_5d",
            "forward_return_5d_available_at",
            "forward_return_20d",
            "forward_return_20d_available_at",
        ),
        filter_columns=("eligible",),
        context_columns=("total_market_cap",),
    )
    dataset_catalog = ResearchDatasetCatalog((spec,))
    identities = []
    mature_dates = {}
    for horizon in (5, 20):
        primary = f"forward_return_{horizon}d"
        configuration = research_configuration()
        model = next(
            item
            for item in configuration["models"]
            if item["model_algorithm"] == "linear_regression"
        )
        model.update(
            model_id="horizon_linear",
            input_factor_columns=[factor],
            input_feature_columns=[],
            label_column=primary,
        )
        model.pop("parameter_candidates", None)
        configuration.update(
            models=[model],
            portfolios=[],
            comparison_model_id=model["model_id"],
            primary_portfolio_id=None,
        )
        configuration["evaluation"].update(
            factors=[factor],
            labels=["forward_return_5d", "forward_return_20d"],
            primary_label=primary,
        )
        bound = bind_research_configuration(
            research_profile(A_SHARE_PROFILE_ID), configuration
        )
        assert (
            bound.protocol("pv_all").manifest()["research_configuration"]["evaluation"][
                "primary_label"
            ]
            == primary
        )
        models = ModelCatalog.from_configuration(configuration["models"])
        protocol = shared_model_protocol(
            models.list(), development_end=cutoff.date().isoformat()
        )
        protocol.update(
            scenario_id=spec.data_id,
            root_seed=42,
            selection_group="global",
            development_window={
                "start": dates[0].date().isoformat(),
                "end": cutoff.date().isoformat(),
            },
            model_chronology={
                "scoring_lookback_months": 12,
                "warmup_window": {
                    "start": (dates[0] - pd.DateOffset(months=12)).date().isoformat(),
                    "end": (dates[0] - pd.Timedelta(days=1)).date().isoformat(),
                },
            },
            research_configuration=configuration,
        )
        protocol["statistics"].update(
            factor_fdr_families={"fixture": [factor]},
            fama_macbeth_core_factors=[factor],
        )
        assert protocol["statistics"]["hac_lag"] == 19
        layout = StorageLayout(tmp_path / str(horizon))
        data_identity = canonical_json_hash(
            {"fixture": "two_horizons", "dataset": spec.data_id}
        )
        root = layout.instance_root(
            layout.research_dataset_root("a_share", spec.data_id), data_identity
        )
        frame["total_market_cap"] = 1.0 + frame.groupby("timestamp").cumcount()
        write_dataset(root, frame, spec=spec)
        layout = layout.with_target_roots({spec.data_id: root})
        session = open_dataset_session(root, columns=None, time_window=None)
        model_id = "horizon_models"
        envelope = build_business_manifest_envelope(
            logical_artifact_id=model_id,
            schema_identity=canonical_json_hash({"models": model_id}),
            protocol_or_spec_identity=canonical_json_hash(protocol),
            software_identity="b" * 64,
            research_window={
                "start": "2022-01-01",
                "end": cutoff.date().isoformat(),
                "access": "development",
            },
            upstream_identities={spec.data_id: data_identity},
            semantic_metadata={
                "target_contract": {
                    "protocol": protocol,
                    "model_specs": [model_spec_payload(item) for item in models.list()],
                }
            },
        )

        results = partial(
            production.build_model_selection_result,
            layout=layout,
            dataset_catalog=dataset_catalog,
            model_catalog=models,
            dataset_id=spec.data_id,
            model_ids=tuple(item.model_id for item in models.list()),
            time_window=TimeWindow("2022-01-01", cutoff.date()),
            protocol_manifest=protocol,
            run_id="run_horizon",
            research_protocol_identity=canonical_json_hash(protocol),
            model_target_id=model_id,
            backend_versions={"internal": "fixture"},
            dataset_session=session,
        )

        model_root = persisted.write_quarterly_model_artifact(
            layout.project_root / "models",
            model_id=model_id,
            envelope=envelope,
            model_results=results,
            trial_layout=layout,
            verify_source=session.verify,
        )
        saved = persisted.read_quarterly_model_artifact(
            model_root,
            expected_identity=envelope.instance_identity,
            trial_layout=layout,
        )
        assert all(
            entry["model_metadata"]["label_column"] == primary
            for entry in saved.final_models
        )
        prediction = PredictionSpec(
            prediction_id="horizon_predictions",
            model_id=model_id,
            research_dataset_id=spec.data_id,
            train_end=None,
            test_start=window.start.isoformat(),
            test_end=window.end.isoformat(),
            prediction_mode="a_share_oos",
            prediction_column="raw_prediction",
        )
        with build_prediction_result_from_persisted_model(
            layout=layout,
            dataset_catalog=dataset_catalog,
            prediction_spec=prediction,
            persisted_model=saved,
            selection_group="global",
        ) as result:
            prediction_root = write_prediction_outputs(
                spec=prediction,
                output_spec=_output_spec_for_target(
                    prediction.prediction_id, "a_share", "prediction"
                ),
                layout=layout,
                output_chunks=result.output_chunks,
                model_metadata=result.model_metadata,
                prediction_metadata_factory=result.prediction_metadata_factory,
            )
        metadata, lookup = read_prediction_output_bundle(
            prediction_root, expected_output_id=prediction.prediction_id
        )
        assert (
            metadata["metadata"]["model_metadata"]["model_artifact_identity"]
            == envelope.instance_identity
        )
        keys = frame.loc[
            frame.timestamp.between(dates[0], cutoff), ["timestamp", "asset_id"]
        ].copy()
        keys["family_id"] = model["model_id"]
        scores, found = lookup.values_for_frame(keys, mark_matched=False)
        assert all(found) and np.isfinite(scores).all()
        assert len(keys) == 45 * 12 and not keys.duplicated().any()
        tables = build_a_share_statistics_tables(
            layout=layout,
            dataset_catalog=dataset_catalog,
            dataset_id=spec.data_id,
            prediction_id=prediction.prediction_id,
            persisted_model=saved,
            protocol_manifest=protocol,
            time_window=window,
            prediction_lookup=lookup,
            global_statistics=None,
        )

        tables.update(
            build_a_share_factor_statistics_tables(
                layout=layout,
                dataset_catalog=dataset_catalog,
                dataset_id=spec.data_id,
                protocol_manifest=protocol,
                time_window=window,
            )
        )
        statistics_envelope = build_business_manifest_envelope(
            logical_artifact_id="horizon_statistics",
            schema_identity=canonical_json_hash({"statistics": "horizon"}),
            protocol_or_spec_identity=canonical_json_hash(protocol),
            software_identity="b" * 64,
            research_window={
                "start": window.start.isoformat(),
                "end": window.end.isoformat(),
                "access": "development",
            },
            upstream_identities={
                spec.data_id: data_identity,
                model_id: envelope.instance_identity,
            },
            semantic_metadata={"primary_label": primary, "hac_lag": 19},
        )
        statistics_envelope = bind_statistics_envelope(
            statistics_envelope, group="research", protocol=protocol
        )
        factor_statistics_envelope = bind_statistics_envelope(
            statistics_envelope, group="factor", protocol=protocol
        )
        factor_statistics_root = write_statistics_artifact(
            (layout.project_root / "statistics").with_name(
                (layout.project_root / "statistics").name + "_factor"
            ),
            envelope=factor_statistics_envelope,
            tables={
                key: value
                for key, value in (tables).items()
                if key + ".parquet" in STATISTICS_TABLE_GROUPS["factor"]
            },
            table_group="factor",
        )
        statistics_root = write_statistics_artifact(
            layout.project_root / "statistics",
            envelope=statistics_envelope,
            tables={
                key: value
                for key, value in (tables).items()
                if key + ".parquet" in STATISTICS_TABLE_GROUPS["research"]
            },
        )
        read_statistics_artifact(
            statistics_root, expected_identity=statistics_envelope.instance_identity
        )
        factor_statistics = read_statistics_artifact(
            factor_statistics_root,
            expected_identity=factor_statistics_envelope.instance_identity,
        )
        identities.append(statistics_envelope.instance_identity)
        expected = frame.loc[
            frame[f"{primary}_available_at"].le(cutoff) & frame.timestamp.le(cutoff)
        ]
        expected_dates = pd.DatetimeIndex(sorted(expected.timestamp.unique())).astype(
            "datetime64[ns]"
        )
        with factor_statistics.table("fama_macbeth_periods.parquet") as periods:
            coefficients = periods.loc[
                periods.regression_id.eq("equal:single:" + factor)
            ]
            assert str(coefficients.timestamp.dtype) == "datetime64[ns]"
            assert not coefficients.duplicated(
                ["regression_id", "coefficient_id", "timestamp"]
            ).any()
            assert coefficients.status.eq("DEFINED").all()
            for coefficient_id in ("intercept", factor):
                actual = coefficients.loc[
                    coefficients.coefficient_id.eq(coefficient_id)
                ].sort_values("timestamp")
                pd.testing.assert_index_equal(
                    pd.DatetimeIndex(actual.timestamp),
                    expected_dates,
                    check_names=False,
                )
                oracle = [
                    np.linalg.lstsq(
                        np.column_stack((np.ones(len(group)), group[factor])),
                        group[primary],
                        rcond=None,
                    )[0][0 if coefficient_id == "intercept" else 1]
                    for _date, group in expected.groupby("timestamp", sort=True)
                ]
                np.testing.assert_allclose(actual.value, oracle, rtol=2e-14, atol=2e-14)
                assert actual.observation_count.eq(12).all()
        with factor_statistics.table("fama_macbeth.parquet") as summary:
            assert summary.period_count.eq(len(expected_dates)).all()
            assert summary.status.eq("DEFINED").all()
        mature_dates[horizon] = expected_dates[expected_dates >= dates[0]]
        assert lookup.coverage_counts() == (0, 0)
        session.verify()
    assert identities[0] != identities[1]
    assert len(mature_dates[5]) == 39 and len(mature_dates[20]) == 24
    assert len(mature_dates[5].difference(mature_dates[20])) == 15


@pytest.mark.parametrize("full_factor_history", (False, True))
def test_family_comparison_production_matches_arch_on_persisted_common_date_losses(
    tmp_path: Path, full_factor_history: bool
):
    from arch.bootstrap import MCS, SPA, StepM

    layout = StorageLayout(tmp_path)
    protocol = deepcopy(a_share.protocol("pv_all").manifest())
    protocol["statistics"]["factor_fdr_families"] = {"fixture": ["x"]}
    if full_factor_history:
        protocol["statistics"]["bootstrap_repetitions"] = 32
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

        def values_for_frame(self, keys, *, mark_matched):
            coordinates = tuple(
                keys.loc[:, ["timestamp", "asset_id", "family_id"]].itertuples(
                    index=False, name=None
                )
            )
            assert mark_matched
            self.matched.update(coordinates)
            return (
                np.asarray([lookup_values[key] for key in coordinates]),
                np.ones(len(coordinates), dtype=bool),
            )

        def coverage_counts(self):
            return (
                len(set(lookup_values) - self.matched),
                len(self.matched - set(lookup_values)),
            )

    spec = make_demo_research_dataset_spec(
        data_id="pv_all",
        factor_columns=("x",),
        label_columns=(a_share.PRIMARY_LABEL_COLUMN, "forward_return_20d_available_at"),
        filter_columns=("eligible",),
        context_columns=("total_market_cap",),
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
    frame["total_market_cap"] = 1.0 + frame.groupby("timestamp").cumcount()
    write_dataset(layout.research_dataset_root("a_share", "pv_all"), frame, spec=spec)
    model = _statistics_model_selection_fixture(
        spec, protocol, TimeWindow(dates[0], cutoff)
    )
    envelope = build_business_manifest_envelope(
        logical_artifact_id="pv_all_global_statistics",
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
    envelope = bind_statistics_envelope(envelope, group="research", protocol=protocol)
    factor_envelope = bind_statistics_envelope(
        envelope, group="factor", protocol=protocol
    )
    statistics_tables = build_a_share_statistics_tables(
        layout=layout,
        dataset_catalog=ResearchDatasetCatalog((spec,)),
        dataset_id="pv_all",
        prediction_id="prediction",
        persisted_model=model,
        protocol_manifest=protocol,
        time_window=TimeWindow(dates[0], cutoff),
        prediction_lookup=Lookup(),
        global_statistics=None,
    )

    statistics_tables.update(
        build_a_share_factor_statistics_tables(
            layout=layout,
            dataset_catalog=ResearchDatasetCatalog((spec,)),
            dataset_id="pv_all",
            protocol_manifest=protocol,
            time_window=TimeWindow(dates[0], cutoff),
        )
    )
    global_tables = {name: table.copy() for name, table in statistics_tables.items()}
    assert global_tables["model_group_comparison"].empty
    factor_root = write_statistics_artifact(
        (
            layout.semantic_artifact_root("a_share", "pv_all_global_statistics")
        ).with_name(
            (layout.semantic_artifact_root("a_share", "pv_all_global_statistics")).name
            + "_factor"
        ),
        envelope=factor_envelope,
        tables={
            key: value
            for key, value in (statistics_tables).items()
            if key + ".parquet" in STATISTICS_TABLE_GROUPS["factor"]
        },
        table_group="factor",
    )
    root = write_statistics_artifact(
        layout.semantic_artifact_root("a_share", "pv_all_global_statistics"),
        envelope=envelope,
        tables={
            key: value
            for key, value in (statistics_tables).items()
            if key + ".parquet" in STATISTICS_TABLE_GROUPS["research"]
        },
    )
    artifact = read_statistics_artifact(
        root, expected_identity=envelope.instance_identity
    )
    factor_artifact = read_statistics_artifact(
        factor_root, expected_identity=factor_envelope.instance_identity
    )
    with factor_artifact.table("factor_diagnostics.parquet") as factors:
        assert factors.observation_count.eq(65 if full_factor_history else 64).all()
    with artifact.table("model_losses.parquet") as table:
        assert len(table) == len(dates) * len(families)
        assert table.common_date.eq(1).all()
        assert table.asset_coverage_id.eq(canonical_json_hash(sorted(assets))).all()
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
    benchmark, challengers = (losses[families[0]], losses.loc[:, list(families[1:])])
    blocked_reason = "global_development_selection_reuses_validation"
    for member in ("prediction_diagnostics.parquet", "model_comparison.parquet"):
        with (
            factor_artifact if member in STATISTICS_TABLE_GROUPS["factor"] else artifact
        ).table(member) as summary:
            assert summary.observation_count.eq(64).all()
            assert summary.status.eq("UNDEFINED").all()
            assert summary.reason.eq(blocked_reason).all()
            assert summary["mean"].notna().all()
            assert (
                summary.loc[
                    :,
                    [
                        "standard_error",
                        "t_statistic",
                        "p_value",
                        "ci_lower",
                        "ci_upper",
                        "bootstrap_lower",
                        "bootstrap_upper",
                    ],
                ]
                .isna()
                .all()
                .all()
            )
    with artifact.table("model_reality_checks.parquet") as checks:
        assert checks.status.eq("UNDEFINED").all()
        assert checks.reason.eq(blocked_reason).all()
        assert checks.value.isna().all()
        assert checks.observation_count.eq(64).all()
    quarterly_protocol = {**protocol, "selection_group": "quarterly"}
    paired_envelope = build_business_manifest_envelope(
        **{
            key: value
            for key, value in envelope.payload().items()
            if key not in {"instance_identity", "status", "logical_artifact_id"}
        },
        logical_artifact_id="pv_all_quarterly_statistics",
    )
    paired_envelope = bind_statistics_envelope(
        paired_envelope, group="research", protocol=quarterly_protocol
    )
    factor_paired_envelope = bind_statistics_envelope(
        paired_envelope, group="factor", protocol=quarterly_protocol
    )
    paired_tables = build_a_share_statistics_tables(
        layout=layout,
        dataset_catalog=ResearchDatasetCatalog((spec,)),
        dataset_id="pv_all",
        prediction_id="quarterly_prediction",
        persisted_model=model,
        protocol_manifest=quarterly_protocol,
        time_window=TimeWindow(dates[0], cutoff),
        prediction_lookup=Lookup(),
        global_statistics=artifact,
    )

    paired_tables.update(
        build_a_share_factor_statistics_tables(
            layout=layout,
            dataset_catalog=ResearchDatasetCatalog((spec,)),
            dataset_id="pv_all",
            protocol_manifest=quarterly_protocol,
            time_window=TimeWindow(dates[0], cutoff),
        )
    )
    quarterly_losses = paired_tables["model_losses"].copy()
    factor_paired_root = write_statistics_artifact(
        (
            layout.semantic_artifact_root("a_share", "pv_all_quarterly_statistics")
        ).with_name(
            (
                layout.semantic_artifact_root("a_share", "pv_all_quarterly_statistics")
            ).name
            + "_factor"
        ),
        envelope=factor_paired_envelope,
        tables={
            key: value
            for key, value in (paired_tables).items()
            if key + ".parquet" in STATISTICS_TABLE_GROUPS["factor"]
        },
        table_group="factor",
    )
    paired_root = write_statistics_artifact(
        layout.semantic_artifact_root("a_share", "pv_all_quarterly_statistics"),
        envelope=paired_envelope,
        tables={
            key: value
            for key, value in (paired_tables).items()
            if key + ".parquet" in STATISTICS_TABLE_GROUPS["research"]
        },
    )
    paired_artifact = read_statistics_artifact(
        paired_root, expected_identity=paired_envelope.instance_identity
    )
    read_statistics_artifact(
        factor_paired_root, expected_identity=factor_paired_envelope.instance_identity
    )
    with paired_artifact.table("model_reality_checks.parquet") as checks:
        assert checks.status.eq("DEFINED").all()
        assert checks.observation_count.eq(64).all()
        if not full_factor_history:
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
        assert len(table) == len(families) - 1
        assert table.observation_count.eq(64).all()
        for row in table.itertuples():
            assert row.mean == pytest.approx(
                float((losses[families[0]] - losses[row.prediction_id]).mean())
            )
    with paired_artifact.table("model_group_comparison.parquet") as paired:
        assert set(paired.family_id) == set(families)
        assert paired.observation_count.eq(64).all()
        np.testing.assert_allclose(paired.mean_difference, 0.0, rtol=0, atol=1e-14)
        assert paired.status.eq("UNDEFINED").all()
        assert paired.reason.eq("global_development_selection_reuses_validation").all()
        assert paired.bootstrap_reason.eq(blocked_reason).all()
    if not full_factor_history:
        robustness_protocol = {
            **quarterly_protocol,
            "training_weighting": {"half_life": 756, "role": "robustness"},
        }
        robustness_tables = build_a_share_statistics_tables(
            layout=layout,
            dataset_catalog=ResearchDatasetCatalog((spec,)),
            dataset_id="pv_all",
            prediction_id="quarterly_prediction",
            persisted_model=model,
            protocol_manifest=robustness_protocol,
            time_window=TimeWindow(dates[0], cutoff),
            prediction_lookup=Lookup(),
            global_statistics=artifact,
        )

        robustness_tables.update(
            build_a_share_factor_statistics_tables(
                layout=layout,
                dataset_catalog=ResearchDatasetCatalog((spec,)),
                dataset_id="pv_all",
                protocol_manifest=robustness_protocol,
                time_window=TimeWindow(dates[0], cutoff),
            )
        )
        for table in ("prediction_diagnostics", "model_comparison"):
            pd.testing.assert_frame_equal(
                robustness_tables[table], paired_tables[table], check_exact=True
            )
        pd.testing.assert_frame_equal(
            robustness_tables["model_trials"],
            paired_tables["model_trials"],
            check_exact=True,
        )
        assert robustness_tables["factor_diagnostics"].status.eq("DEFINED").all()
        assert (
            robustness_tables["factor_diagnostics"]
            .fdr_method.eq("NOT_APPLICABLE")
            .all()
        )
        assert (
            robustness_tables["model_group_comparison"]
            .fdr_method.eq("NOT_APPLICABLE")
            .all()
        )
        for table in ("model_reality_checks", "robustness"):
            assert robustness_tables[table].status.eq("UNDEFINED").all()
            assert (
                robustness_tables[table]
                .reason.eq("training_weighting_robustness_excluded")
                .all()
            )
        factor_robust_root = write_statistics_artifact(
            (
                layout.semantic_artifact_root("a_share", "pv_all_robust_statistics")
            ).with_name(
                (
                    layout.semantic_artifact_root("a_share", "pv_all_robust_statistics")
                ).name
                + "_factor"
            ),
            envelope=factor_paired_envelope,
            tables={
                key: value
                for key, value in (robustness_tables).items()
                if key + ".parquet" in STATISTICS_TABLE_GROUPS["factor"]
            },
            table_group="factor",
        )
        robust_root = write_statistics_artifact(
            layout.semantic_artifact_root("a_share", "pv_all_robust_statistics"),
            envelope=paired_envelope,
            tables={
                key: value
                for key, value in (robustness_tables).items()
                if key + ".parquet" in STATISTICS_TABLE_GROUPS["research"]
            },
        )
        robust_artifact = read_statistics_artifact(
            robust_root, expected_identity=paired_envelope.instance_identity
        )
        factor_robust_artifact = read_statistics_artifact(
            factor_robust_root,
            expected_identity=factor_paired_envelope.instance_identity,
        )
        with factor_robust_artifact.table("factor_diagnostics.parquet") as table:
            assert table.status.eq("DEFINED").all()
            assert table.fdr_method.eq("NOT_APPLICABLE").all()
            assert table.p_value.notna().all()
            assert table.fdr_adjusted_p_value.isna().all()
        with robust_artifact.table("model_reality_checks.parquet") as table:
            assert table.value.isna().all()
            assert table.reason.eq("training_weighting_robustness_excluded").all()
        for mutation in ("key", "count", "asset_identity", "family"):
            malformed = {name: table.copy() for name, table in global_tables.items()}
            losses_table = malformed["model_losses"]
            if mutation == "key":
                losses_table.loc[0, "timestamp"] += pd.Timedelta(nanoseconds=1)
            elif mutation == "count":
                losses_table.loc[0, "observation_count"] += 1
            elif mutation == "family":
                malformed["model_losses"] = losses_table.loc[
                    losses_table.family_id.ne(families[-1])
                ].copy()
            else:
                losses_table.loc[0, "asset_coverage_id"] = canonical_json_hash(
                    sorted([*assets[:-1], "different_asset"])
                )
            mutated_envelope = build_business_manifest_envelope(
                **{
                    key: value
                    for key, value in envelope.payload().items()
                    if key not in {"instance_identity", "status", "logical_artifact_id"}
                },
                logical_artifact_id=f"coverage_{mutation}",
            )
            mutated_envelope = bind_statistics_envelope(
                mutated_envelope, group="research", protocol=protocol
            )
            factor_mutated_envelope = bind_statistics_envelope(
                mutated_envelope, group="factor", protocol=protocol
            )
            factor_mutated_root = write_statistics_artifact(
                (tmp_path / f"coverage_{mutation}").with_name(
                    (tmp_path / f"coverage_{mutation}").name + "_factor"
                ),
                envelope=factor_mutated_envelope,
                tables={
                    key: value
                    for key, value in (malformed).items()
                    if key + ".parquet" in STATISTICS_TABLE_GROUPS["factor"]
                },
                table_group="factor",
            )
            mutated_root = write_statistics_artifact(
                tmp_path / f"coverage_{mutation}",
                envelope=mutated_envelope,
                tables={
                    key: value
                    for key, value in (malformed).items()
                    if key + ".parquet" in STATISTICS_TABLE_GROUPS["research"]
                },
            )
            mutated_artifact = read_statistics_artifact(
                mutated_root, expected_identity=mutated_envelope.instance_identity
            )
            read_statistics_artifact(
                factor_mutated_root,
                expected_identity=factor_mutated_envelope.instance_identity,
            )
            with pytest.raises(
                ValueError,
                match="model_group_comparison_family_incomplete"
                if mutation == "family"
                else "model_group_comparison_coverage_invalid",
            ):
                statistics_production._paired_group_rankic(
                    quarterly_losses.loc[
                        quarterly_losses.family_id.ne(families[-1])
                    ].copy()
                    if mutation == "family"
                    else quarterly_losses,
                    global_statistics=mutated_artifact,
                    selection_group="quarterly",
                    family_ids=families,
                    protocol_manifest=quarterly_protocol,
                )
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
        terminal_targets=("pv_all_global_statistics",),
        target_results=(
            ResearchRunTargetResult(
                target_id="pv_all_global_statistics",
                producer_kind=ProducerKind.STATISTIC,
                status=TargetStatus.BUILT,
                instance_identity=envelope.instance_identity,
                schema_identity=envelope.schema_identity,
                artifact_root="data/artifacts/a_share/pv_all_global_statistics",
            ),
        ),
        status=RunStatus.COMPLETE,
    )
    evidence = load_a_share_run_evidence(
        layout=layout, record=record, selections=model.selections
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
    actual = report["sections"]["OOS prediction"]["global"]["reality_checks"]
    assert len(actual) == 3 * len(families) + 2 and all(
        row["status"] == "UNDEFINED"
        and row["reason"] == blocked_reason
        and (row["observation_count"] == 64)
        for row in actual
    )
    assert report["sections"]["OOS prediction"]["global"][
        "inference_qualification"
    ] == {"status": "UNDEFINED", "reason": blocked_reason}
    assert report["sections"]["OOS prediction"]["quarterly"][
        "inference_qualification"
    ] == {"status": "ELIGIBLE", "reason": None}
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
    assert (
        saved_report["sections"]["OOS prediction"]["global"]["reality_checks"] == actual
    )


def test_robustness_joint_exclusion_preserves_individual_hac(monkeypatch):
    rng = np.random.default_rng(74)
    values = pd.DataFrame(
        {
            "baseline": rng.normal(0.01, 0.04, 64),
            "challenger": rng.normal(0.02, 0.04, 64),
        },
        index=pd.date_range("2023-01-01", periods=64),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("robustness runs must not enter the formal joint test domain")

    for name in ("spa_reality_check", "stepm_superior_models", "model_confidence_set"):
        monkeypatch.setattr(statistics_production, name, forbidden)
    comparison, checks, robustness = statistics_production._model_comparison(
        values,
        family_ids=("baseline", "challenger"),
        benchmark_id="baseline",
        access_window="development",
        inference_blocker=None,
        joint_blocker="expectations_robustness_excluded",
        max_lags=2,
        block_size=4,
        repetitions=32,
        root_seed=17,
        dataset_id="robustness",
    )
    assert comparison.status.eq("DEFINED").all()
    assert checks.status.eq("UNDEFINED").all()
    assert robustness.reason.eq("expectations_robustness_excluded").all()
    assert checks.value.isna().all()
    factors = statistics_production._factor_summary(
        values.reset_index(names="timestamp"),
        factors=("baseline", "challenger"),
        families={"expectations": ("baseline", "challenger")},
        methods=(None,),
        max_lags=2,
    )
    assert factors.status.eq("DEFINED").all()
    assert factors.fdr_method.eq("NOT_APPLICABLE").all()
    assert factors.fdr_adjusted_p_value.isna().all()


@pytest.mark.parametrize("core_factors", ((), ("factor",)))
def test_empty_fama_macbeth_periods_do_not_invent_dates_or_coefficients(
    core_factors,
) -> None:
    periods = pd.DataFrame(columns=statistics_production._FAMA_MACBETH_PERIOD_COLUMNS)
    summary = statistics_production._fama_macbeth_summary(
        periods, factors=("factor",), core_factors=core_factors, max_lags=19
    )
    assert periods.empty
    assert list(summary["regression_id"]) == [
        weighting + ":" + regression
        for weighting in ("equal", "total_market_cap")
        for regression in (
            ["single:factor"] * 2 + (["core_multifactor"] * 2 if core_factors else [])
        )
    ]
    assert set(summary["coefficient_id"]) == {"intercept", "factor"}
    assert summary["period_count"].eq(0).all()
    assert summary["white_defined_period_count"].eq(0).all()
    assert summary["white_undefined_period_count"].eq(0).all()
    assert summary["status"].eq("UNDEFINED").all()
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["a", "b", "c", "d"],
            "factor": [0.0, 1.0, 2.0, 3.0],
            "label": [0.0, 2.0, 4.0, 6.0],
            "total_market_cap": [1.0, 2.0, 3.0, 4.0],
        }
    )
    records = statistics_production._fama_macbeth_period_records(
        frame, return_column="label", factors=("factor",), core_factors=core_factors
    )
    assert [row["regression_id"] for row in records] == summary.regression_id.tolist()
    assert [row["coefficient_id"] for row in records] == summary.coefficient_id.tolist()


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
                if date_index // 5 % 2:
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
            context_columns=("total_market_cap",),
            filter_column_roles={"eligible": ColumnRole.UNIVERSE},
        )
        frame = pd.DataFrame(rows)
        frame["total_market_cap"] = 1.0 + frame.groupby("timestamp").cumcount()
        write_dataset(
            layout.research_dataset_root(spec.market_id, spec.data_id),
            frame,
            spec=spec,
        )
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
        model = _statistics_model_selection_fixture(
            spec, protocol, TimeWindow(dates[0], cutoff)
        )
        statistics_tables = build_a_share_statistics_tables(
            layout=layout,
            dataset_catalog=ResearchDatasetCatalog((spec,)),
            dataset_id=spec.data_id,
            prediction_id="prediction",
            persisted_model=model,
            protocol_manifest=protocol,
            time_window=TimeWindow(dates[0], cutoff),
            prediction_lookup=_PredictionLookup(),
            global_statistics=None,
        )

        statistics_tables.update(
            build_a_share_factor_statistics_tables(
                layout=layout,
                dataset_catalog=ResearchDatasetCatalog((spec,)),
                dataset_id=spec.data_id,
                protocol_manifest=protocol,
                time_window=TimeWindow(dates[0], cutoff),
            )
        )
        outputs.append(
            {name: table.copy() for name, table in statistics_tables.items()}
        )
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
    assert prediction["status"] == "UNDEFINED"
    assert prediction["reason"] == "global_development_selection_reuses_validation"
    assert prediction["mean"] == pytest.approx(0.2, abs=1e-15)
    assert prediction["bootstrap_seed"] == derive_seed(
        42, "a_share_statistics_bootstrap", "stats_demo", "prediction"
    )
    fmb = outputs[0]["fama_macbeth"]
    assert {"equal:single:" + CORE6_FACTORS[0], "equal:core_multifactor"}.issubset(
        set(fmb["regression_id"])
    )
    assert "intercept" in set(fmb["coefficient_id"])
    assert {0, 20}.issubset(set(fmb["period_count"]))
    trials = outputs[0]["model_trials"]
    assert trials.selection_group.eq("global").all()
    assert trials.decision_id.eq("final").all()
    assert set(trials.family_id) == {
        f"stats_demo_{algorithm}" for algorithm in A_SHARE_MODEL_ALGORITHMS
    }
    assert trials.loc[trials.selection_status == "CHAMPION", "config_id"].nunique() == 1
    assert {
        "validation_icir",
        "validation_rankicir",
        "validation_std_ic",
        "validation_std_rankic",
        "validation_valid_ic_count",
        "validation_valid_rankic_count",
    }.issubset(set(trials.metric_id))
    assert set(trials.selection_status) == {"CHAMPION", "FAMILY_WINNER"}
    assert set(outputs[0]["robustness"]["reason"]) == {
        "global_development_selection_reuses_validation"
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
    envelope = bind_statistics_envelope(envelope, group="research", protocol=protocol)
    factor_envelope = bind_statistics_envelope(
        envelope, group="factor", protocol=protocol
    )
    factor_artifact_root = write_statistics_artifact(
        (tmp_path / "statistics").with_name((tmp_path / "statistics").name + "_factor"),
        envelope=factor_envelope,
        tables={
            key: value
            for key, value in (outputs[0]).items()
            if key + ".parquet" in STATISTICS_TABLE_GROUPS["factor"]
        },
        table_group="factor",
    )
    artifact_root = write_statistics_artifact(
        tmp_path / "statistics",
        envelope=envelope,
        tables={
            key: value
            for key, value in (outputs[0]).items()
            if key + ".parquet" in STATISTICS_TABLE_GROUPS["research"]
        },
    )
    artifact = read_statistics_artifact(
        artifact_root, expected_identity=envelope.instance_identity
    )
    factor_artifact = read_statistics_artifact(
        factor_artifact_root, expected_identity=factor_envelope.instance_identity
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
    with artifact.table("model_group_comparison.parquet") as table:
        assert table.empty
    with factor_artifact.table("factor_diagnostics.parquet") as table:
        pd.testing.assert_frame_equal(
            table,
            factors.assign(reason=factors.reason.fillna("")),
            check_dtype=False,
            check_exact=True,
        )
    for member in members:
        with (
            factor_artifact if member in STATISTICS_TABLE_GROUPS["factor"] else artifact
        ).table(member) as table:
            assert not table.empty
    from scipy.stats import norm

    expected_periods = {"intercept": [], CORE6_FACTORS[0]: []}
    x = np.arange(6, dtype="float64") - 2.0
    centered_x = x - x.mean()
    for date_index in range(20):
        y = (
            0.1
            + 0.4 * x
            + 0.03 * x * x
            + 0.08 * np.sin((np.arange(6) + 1) * (date_index + 1))
        )
        if date_index // 5 % 2:
            y = -y
        slope = np.dot(centered_x, y - y.mean()) / np.dot(centered_x, centered_x)
        expected_periods[CORE6_FACTORS[0]].append(slope)
        expected_periods["intercept"].append(y.mean() - slope * x.mean())
    with factor_artifact.table("fama_macbeth_periods.parquet") as table:
        persisted_periods = table.loc[
            table.regression_id.eq("equal:single:" + CORE6_FACTORS[0])
        ].copy()
    with factor_artifact.table("fama_macbeth.parquet") as table:
        persisted_summary = table.loc[
            table.regression_id.eq("equal:single:" + CORE6_FACTORS[0])
        ].set_index("coefficient_id")
        for coefficient, values in expected_periods.items():
            values = np.asarray(values)
            actual = persisted_periods.loc[
                persisted_periods.coefficient_id.eq(coefficient)
            ].sort_values("timestamp")
            np.testing.assert_allclose(actual.value, values, rtol=1e-12, atol=1e-15)
            centered = values - values.mean()
            variance_sum = np.dot(centered, centered) + 2 * sum(
                (1 - lag / 20) * np.dot(centered[lag:], centered[:-lag])
                for lag in range(1, 20)
            )
            se = np.sqrt(variance_sum) / 20
            mean = values.mean()
            expected = [
                mean,
                se,
                mean / se,
                2 * norm.sf(abs(mean / se)),
                mean - norm.ppf(0.975) * se,
                mean + norm.ppf(0.975) * se,
            ]
            fields = [
                "mean",
                "standard_error",
                "t_statistic",
                "p_value",
                "ci_lower",
                "ci_upper",
            ]
            np.testing.assert_allclose(
                persisted_summary.loc[coefficient, fields].astype(float),
                expected,
                rtol=1e-11,
                atol=1e-14,
            )
            assert se != pytest.approx(np.std(values, ddof=0) / np.sqrt(20))
    factor_zero_root = write_statistics_artifact(
        (tmp_path / "statistics_zero_residual").with_name(
            (tmp_path / "statistics_zero_residual").name + "_factor"
        ),
        envelope=factor_envelope,
        tables={
            key: value
            for key, value in (outputs[2]).items()
            if key + ".parquet" in STATISTICS_TABLE_GROUPS["factor"]
        },
        table_group="factor",
    )
    zero_root = write_statistics_artifact(
        tmp_path / "statistics_zero_residual",
        envelope=envelope,
        tables={
            key: value
            for key, value in (outputs[2]).items()
            if key + ".parquet" in STATISTICS_TABLE_GROUPS["research"]
        },
    )
    zero_artifact = read_statistics_artifact(
        zero_root, expected_identity=envelope.instance_identity
    )
    factor_zero_artifact = read_statistics_artifact(
        factor_zero_root, expected_identity=factor_envelope.instance_identity
    )
    for member in members:
        with (
            factor_zero_artifact
            if member in STATISTICS_TABLE_GROUPS["factor"]
            else zero_artifact
        ).table(member) as table:
            assert not table.empty
    with factor_zero_artifact.table("fama_macbeth_periods.parquet") as table:
        periods = table.copy()
    single_factor = periods.loc[
        periods["regression_id"].eq("equal:single:" + CORE6_FACTORS[0])
    ]
    assert single_factor["status"].eq("DEFINED").all()
    assert single_factor["value"].eq(0.0).all()
    assert single_factor["white_p_value"].isna().all()
    assert single_factor["white_reason"].eq("white_test_undefined").all()
    with factor_zero_artifact.table("fama_macbeth.parquet") as table:
        zero_summary = table.copy()
    single_summary = zero_summary.loc[
        zero_summary["regression_id"].eq("equal:single:" + CORE6_FACTORS[0])
    ]
    assert single_summary["white_defined_period_count"].eq(0).all()
    assert single_summary["white_undefined_period_count"].eq(20).all()
    empty_period_tables = dict(outputs[0])
    empty_period_tables["fama_macbeth_periods"] = empty_period_tables[
        "fama_macbeth_periods"
    ].iloc[:0]
    factor_empty_root = write_statistics_artifact(
        (tmp_path / "statistics_empty_periods").with_name(
            (tmp_path / "statistics_empty_periods").name + "_factor"
        ),
        envelope=factor_envelope,
        tables={
            key: value
            for key, value in (empty_period_tables).items()
            if key + ".parquet" in STATISTICS_TABLE_GROUPS["factor"]
        },
        table_group="factor",
    )
    empty_root = write_statistics_artifact(
        tmp_path / "statistics_empty_periods",
        envelope=envelope,
        tables={
            key: value
            for key, value in (empty_period_tables).items()
            if key + ".parquet" in STATISTICS_TABLE_GROUPS["research"]
        },
    )
    read_statistics_artifact(empty_root, expected_identity=envelope.instance_identity)
    factor_empty_artifact = read_statistics_artifact(
        factor_empty_root, expected_identity=factor_envelope.instance_identity
    )
    with factor_empty_artifact.table("fama_macbeth_periods.parquet") as table:
        assert table.empty
    primary = KeyboardInterrupt("table_consumer_interrupted")
    artifact = read_statistics_artifact(
        artifact_root, expected_identity=envelope.instance_identity
    )
    factor_artifact = read_statistics_artifact(
        factor_artifact_root, expected_identity=factor_envelope.instance_identity
    )
    with pytest.raises(KeyboardInterrupt) as caught:
        with artifact.table("model_losses.parquet") as table:
            assert not table.empty
            raise primary
    assert caught.value is primary
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    from quant_research_platform.storage.identity import native_filesystem_path
    from quant_research_platform.storage.semantic_artifacts import (
        write_semantic_artifact,
    )

    def wrong_schema(staging):
        for member in STATISTICS_TABLE_GROUPS["research"]:
            if member == "model_losses.parquet":
                pq.write_table(
                    pa.table({"wrong": [1]}),
                    native_filesystem_path(staging / member),
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

    assert return_standard_deviation([value] * 60) == 0
    with pytest.raises(ValueError, match="sharpe_zero_variance"):
        probabilistic_and_deflated_sharpe(
            [value] * 60, benchmark_sharpe=0.0, trial_sharpes=[0.1, 0.2]
        )
    assert return_standard_deviation([-1.0, 0.0, 1.0]) == 1.0


@pytest.mark.parametrize(
    "access,group",
    [
        ("development", "global"),
        ("development", "quarterly"),
        ("holdout", "global"),
        ("holdout", "quarterly"),
    ],
)
def test_model_inference_uses_selection_information_before_calling_backends(
    monkeypatch, access, group
):
    rng = np.random.default_rng(22019)
    values = pd.DataFrame(
        {
            "timestamp": pd.bdate_range("2023-01-02", periods=64),
            "baseline": rng.normal(0.01, 0.04, 64),
            "challenger": rng.normal(0.02, 0.04, 64),
        }
    )
    blocker = _model_inference_blocker(access_window=access, selection_groups=(group,))
    if blocker is not None:

        def forbidden(*args, **kwargs):
            pytest.fail("retrospective selection must not invoke inference backends")

        for name in (
            "hac_mean",
            "stationary_bootstrap_mean_ci",
            "spa_reality_check",
            "stepm_superior_models",
            "model_confidence_set",
        ):
            monkeypatch.setattr(statistics_production, name, forbidden)
    summary = statistics_production._prediction_summary(
        values.rename(columns={"baseline": "__prediction_alpha__"}),
        prediction_id="prediction",
        inference_blocker=blocker,
        max_lags=2,
        block_size=4,
        repetitions=100,
        seed=17,
    )
    comparison, checks, robustness = statistics_production._model_comparison(
        values.set_index("timestamp"),
        family_ids=("baseline", "challenger"),
        benchmark_id="baseline",
        access_window=access,
        inference_blocker=blocker,
        max_lags=2,
        block_size=4,
        repetitions=100,
        root_seed=17,
        dataset_id="fixture",
    )
    assert summary["mean"].iloc[0] == pytest.approx(values.baseline.mean())
    assert comparison["mean"].iloc[0] == pytest.approx(
        (values.challenger - values.baseline).mean()
    )
    if blocker is not None:
        assert summary.status.eq("UNDEFINED").all()
        assert comparison.status.eq("UNDEFINED").all()
        assert checks.status.eq("UNDEFINED").all()
        assert robustness.reason.eq(blocker).all()
    else:
        assert summary.status.eq("DEFINED").all()
        assert comparison.status.eq("DEFINED").all()
        if access == "development":
            assert checks.status.eq("DEFINED").all()
        else:
            assert checks.status.eq("UNDEFINED").all()
            assert checks.reason.eq("holdout_excluded").all()


@pytest.mark.parametrize("has_observations", (False, True))
def test_paired_model_rankic_consumes_the_statistic_target_manifest(has_observations):
    protocol = a_share.AShareProtocol(
        "pv_all", configuration_json=json.dumps(research_configuration())
    )
    target_manifest = protocol.target_manifest(
        protocol.statistics_target_ids[1], "STATISTIC"
    )
    families = protocol.model_ids_for_target(protocol.model_selection_target_id)
    columns = (
        "timestamp",
        "family_id",
        "asset_coverage_id",
        "observation_count",
        "access_window",
        "loss",
    )
    losses = pd.DataFrame.from_records(
        [
            (pd.Timestamp("2023-01-03"), family, "a" * 64, 2, "development", -0.5)
            for family in families
        ]
        if has_observations
        else [],
        columns=columns,
    )

    class GlobalStatistics:
        @contextmanager
        def table(self, member):
            assert member == "model_losses.parquet"
            yield losses

    paired = statistics_production._paired_group_rankic(
        losses,
        global_statistics=GlobalStatistics(),
        selection_group="quarterly",
        family_ids=families,
        protocol_manifest=target_manifest,
    )
    assert set(paired.family_id) == set(families)
    assert paired.observation_count.eq(int(has_observations)).all()
    assert paired.status.eq("UNDEFINED").all()
    assert paired.reason.eq(
        "global_development_selection_reuses_validation"
        if has_observations
        else "no_common_observations"
    ).all()
    if has_observations:
        assert paired.mean_difference.eq(0.0).all()
    else:
        assert paired.mean_difference.isna().all()


@pytest.mark.parametrize("access", ("development", "holdout"))
@pytest.mark.parametrize(
    "family_id", ("momentum_20d", "__prediction_alpha__", "timestamp")
)
def test_model_family_names_preserve_factor_statistics_and_persisted_identity(
    tmp_path: Path, family_id: str, access: str
) -> None:
    factor = "momentum_20d"
    dates = pd.bdate_range(
        "2024-01-02" if access == "holdout" else "2019-01-02", periods=3
    )
    quarter_id = str(dates[0].to_period("Q"))
    calendar = pd.bdate_range(dates[0], periods=len(dates) + 21)
    cutoff = calendar[-1]
    assets = [f"asset_{index}" for index in range(6)]
    factor_values = np.tile(np.arange(len(assets), dtype=float), len(dates))
    frame = pd.DataFrame(
        {
            "timestamp": dates.repeat(len(assets)),
            "asset_id": assets * len(dates),
            "market_id": "a_share",
            "eligible": True,
            factor: factor_values,
            a_share.PRIMARY_LABEL_COLUMN: 2.0 + 3.0 * factor_values,
            "forward_return_20d_available_at": calendar[21:].repeat(len(assets)),
        }
    )
    layout = StorageLayout(tmp_path)
    spec = make_demo_research_dataset_spec(
        data_id="family_names",
        factor_columns=(factor,),
        label_columns=(a_share.PRIMARY_LABEL_COLUMN, "forward_return_20d_available_at"),
        filter_columns=("eligible",),
        context_columns=("total_market_cap",),
    )
    frame["total_market_cap"] = 1.0 + frame.groupby("timestamp").cumcount()
    write_dataset(
        layout.research_dataset_root("a_share", spec.data_id), frame, spec=spec
    )
    results = []
    for position, selected_family in enumerate((family_id, "renamed_model")):
        families = ("baseline", selected_family)
        protocol = deepcopy(a_share.protocol("pv_all").manifest())
        protocol["research_configuration"] = research_configuration(small_models=True)
        protocol.update(
            selection_group="global",
            access_window=access,
            development_window={
                "start": dates[0].date().isoformat(),
                "end": cutoff.date().isoformat(),
            },
            formal_dataset_window={
                "start": dates[0].date().isoformat(),
                "end": cutoff.date().isoformat(),
            },
        )
        protocol["research_configuration"]["evaluation"]["factors"] = [factor]
        protocol["research_configuration"]["comparison_model_id"] = families[0]
        protocol["statistics"].update(
            hac_lag=1,
            bootstrap_block=2,
            bootstrap_repetitions=16,
            factor_fdr_families={"fixture": [factor]},
            fama_macbeth_core_factors=[factor],
        )
        selection_payload = {
            "selected_family_id": selected_family,
            "families": [
                {
                    "family_id": family,
                    "selected_config_id": family + "_fit",
                    "trial_metrics": [
                        {
                            "model_id": family + "_fit",
                            "validation_rankic": 0.1,
                            "validation_ic": 0.1,
                            "validation_ic_reason": None,
                            "validation_mse": 1.0,
                            **{
                                f"validation_{key}": value
                                for prefix in ("ic", "rankic")
                                for key, value in (
                                    (f"std_{prefix}", 0.0),
                                    (f"{prefix}ir", None),
                                    (f"positive_{prefix}_rate", 1.0),
                                    (f"valid_{prefix}_count", 1),
                                    (f"{prefix}ir_reason", "insufficient_observations"),
                                )
                            },
                            "validation_rows": 6,
                            "validation_rankic_reason": None,
                        }
                    ],
                }
                for family in families
            ],
        }

        class Models:
            family_ids = families
            payload = selection_payload
            selections = {"global": {quarter_id: selection_payload}}

            def prediction_models(self, _window, **_kwargs):
                return [
                    {
                        "quarter_id": quarter_id,
                        "family_id": family,
                        "model_id": family + "_fit",
                        "model_metadata": {
                            "input_columns": [factor],
                            "model_algorithm": A_SHARE_MODEL_ALGORITHMS[0],
                        },
                    }
                    for family in self.family_ids
                ]

            def selection(self, _group, _quarter, **_kwargs):
                return self.payload

        class Lookup:
            family = selected_family

            def __init__(self):
                self.matched = set()

            def values_for_frame(self, keys, *, mark_matched):
                assert mark_matched
                self.matched.update(
                    keys[["timestamp", "asset_id", "family_id"]].itertuples(
                        index=False, name=None
                    )
                )
                values = keys.asset_id.str.removeprefix("asset_").astype(float)
                signs = np.where(keys.family_id.eq(self.family), -1.0, 1.0)
                return (values.to_numpy() * signs, np.ones(len(keys), dtype=bool))

            def coverage_counts(self):
                return (len(dates) * len(assets) * 2 - len(self.matched), 0)

        envelope = build_business_manifest_envelope(
            logical_artifact_id=f"family_names_{position}",
            schema_identity="a" * 64,
            protocol_or_spec_identity="b" * 64,
            software_identity="c" * 64,
            research_window={
                "start": str(dates[0].date()),
                "end": str(cutoff.date()),
                "access": access,
            },
            upstream_identities={},
            semantic_metadata={},
        )
        envelope = bind_statistics_envelope(
            envelope, group="research", protocol=protocol
        )
        factor_envelope = bind_statistics_envelope(
            envelope, group="factor", protocol=protocol
        )
        statistics_tables = build_a_share_statistics_tables(
            layout=layout,
            dataset_catalog=ResearchDatasetCatalog((spec,)),
            dataset_id=spec.data_id,
            prediction_id="selected_prediction",
            persisted_model=Models(),
            protocol_manifest=protocol,
            time_window=TimeWindow(dates[0], cutoff),
            prediction_lookup=Lookup(),
            global_statistics=None,
        )

        statistics_tables.update(
            build_a_share_factor_statistics_tables(
                layout=layout,
                dataset_catalog=ResearchDatasetCatalog((spec,)),
                dataset_id=spec.data_id,
                protocol_manifest=protocol,
                time_window=TimeWindow(dates[0], cutoff),
            )
        )
        members = tuple(statistics_tables)
        factor_root = write_statistics_artifact(
            (tmp_path / f"statistics_{position}").with_name(
                (tmp_path / f"statistics_{position}").name + "_factor"
            ),
            envelope=factor_envelope,
            tables={
                key: value
                for key, value in (statistics_tables).items()
                if key + ".parquet" in STATISTICS_TABLE_GROUPS["factor"]
            },
            table_group="factor",
        )
        root = write_statistics_artifact(
            tmp_path / f"statistics_{position}",
            envelope=envelope,
            tables={
                key: value
                for key, value in (statistics_tables).items()
                if key + ".parquet" in STATISTICS_TABLE_GROUPS["research"]
            },
        )
        artifact = read_statistics_artifact(
            root, expected_identity=envelope.instance_identity
        )
        factor_artifact = read_statistics_artifact(
            factor_root, expected_identity=factor_envelope.instance_identity
        )
        tables = {}
        for member in members:
            with (
                factor_artifact
                if member + ".parquet" in STATISTICS_TABLE_GROUPS["factor"]
                else artifact
            ).table(member + ".parquet") as table:
                tables[member] = table.copy()
        factor_diagnostics = tables["factor_diagnostics"]
        assert factor_diagnostics.observation_count.eq(len(dates)).all()
        assert factor_diagnostics.status.eq("UNDEFINED").all()
        assert factor_diagnostics.reason.eq("hac_inference_undefined").all()
        assert factor_diagnostics["mean"].isna().all()
        assert tables["prediction_diagnostics"]["mean"].iloc[0] == pytest.approx(-1.0)
        periods = tables["fama_macbeth_periods"]
        slopes = periods.loc[periods.coefficient_id.eq(factor), "value"]
        assert len(slopes) == 4 * len(dates)
        assert slopes.tolist() == pytest.approx([3.0] * len(slopes))
        losses = tables["model_losses"]
        assert set(losses.family_id) == set(families)
        assert set(losses.config_id) == {family + "_fit" for family in families}
        assert losses.loc[
            losses.family_id.eq(selected_family), "loss"
        ].tolist() == pytest.approx([1.0] * len(dates))
        assert losses.loc[
            losses.family_id.eq("baseline"), "loss"
        ].tolist() == pytest.approx([-1.0] * len(dates))
        assert losses.common_date.eq(1).all()
        assert losses.common_observation_count.eq(len(assets)).all()
        assert tables["model_comparison"]["mean"].iloc[0] == pytest.approx(-2.0)
        assert tables["model_comparison"].prediction_id.tolist() == [selected_family]
        assert set(tables["model_trials"].family_id) == set(families)
        quarterly_losses = losses.copy()
        quarterly_losses["loss"] *= np.repeat([0.7, 0.8, 0.9], len(families))
        paired = statistics_production._paired_group_rankic(
            quarterly_losses,
            global_statistics=artifact,
            selection_group="quarterly",
            family_ids=families,
            protocol_manifest={**protocol, "selection_group": "quarterly"},
        )
        assert paired.family_id.tolist() == list(families)
        assert paired.observation_count.eq(len(dates)).all()
        assert paired.calendar_count.eq(len(dates)).all()
        assert paired.fdr_family_size.eq(len(families)).all()
        assert paired.mean_difference.tolist() == pytest.approx([-0.2, 0.2])
        assert paired.status.eq("DEFINED" if access == "holdout" else "UNDEFINED").all()
        assert paired.bootstrap_status.eq(
            "DEFINED" if access == "holdout" else "UNDEFINED"
        ).all()
        if access == "development":
            assert paired.reason.eq(
                "global_development_selection_reuses_validation"
            ).all()
        else:
            assert paired.standard_error.notna().all()
            assert paired.fdr_adjusted_p_value.notna().all()
            assert paired.bootstrap_lower.notna().all()
        paired_tables = {
            **tables,
            "model_losses": quarterly_losses,
            "model_group_comparison": paired,
        }
        factor_paired_root = write_statistics_artifact(
            (tmp_path / f"paired_statistics_{position}").with_name(
                (tmp_path / f"paired_statistics_{position}").name + "_factor"
            ),
            envelope=factor_envelope,
            tables={
                key: value
                for key, value in (paired_tables).items()
                if key + ".parquet" in STATISTICS_TABLE_GROUPS["factor"]
            },
            table_group="factor",
        )
        paired_root = write_statistics_artifact(
            tmp_path / f"paired_statistics_{position}",
            envelope=envelope,
            tables={
                key: value
                for key, value in (paired_tables).items()
                if key + ".parquet" in STATISTICS_TABLE_GROUPS["research"]
            },
        )
        paired_artifact = read_statistics_artifact(
            paired_root, expected_identity=envelope.instance_identity
        )
        read_statistics_artifact(
            factor_paired_root, expected_identity=factor_envelope.instance_identity
        )
        with paired_artifact.table("model_group_comparison.parquet") as persisted:
            pd.testing.assert_frame_equal(
                persisted,
                paired.assign(
                    reason=paired.reason.fillna("").astype(str),
                    bootstrap_reason=paired.bootstrap_reason.fillna("").astype(str),
                ),
                check_exact=True,
            )
            tables["model_group_comparison"] = persisted.copy()
        for table in tables.values():
            assert all(isinstance(column, str) for column in table.columns)
            for column in ("family_id", "prediction_id", "config_id"):
                if column in table:
                    table[column] = table[column].replace(
                        {
                            selected_family: "same_model",
                            selected_family + "_fit": "same_model_fit",
                        }
                    )
        tables["model_comparison"] = tables["model_comparison"].drop(
            columns="bootstrap_seed"
        )
        results.append(tables)
    for member in results[0]:
        pd.testing.assert_frame_equal(
            results[0][member], results[1][member], check_exact=True
        )


def test_factor_fmb_weightings_match_independent_weighted_normal_equations():
    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-02"),
            "asset_id": list("abcdef"),
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [0.0, 1.0, 3.0, 2.0, 6.0, 9.0],
            "total_market_cap": [1.0, 2.0, 4.0, 8.0, 16.0, np.nan],
        }
    )
    records = pd.DataFrame(
        statistics_production._fama_macbeth_period_records(
            frame,
            return_column="y",
            factors=("x",),
            core_factors=(),
        )
    )
    for weighting in ("equal", "total_market_cap"):
        sample = frame if weighting == "equal" else frame.dropna()
        weights = (
            np.ones(len(sample))
            if weighting == "equal"
            else sample.total_market_cap.to_numpy()
        )
        x = sample.x.to_numpy()
        y = sample.y.to_numpy()
        x_mean, y_mean = np.average(x, weights=weights), np.average(y, weights=weights)
        slope = np.sum(weights * (x - x_mean) * (y - y_mean)) / np.sum(
            weights * (x - x_mean) ** 2
        )
        expected = {"intercept": y_mean - slope * x_mean, "x": slope}
        actual = records.loc[records.regression_id.eq(weighting + ":single:x")]
        assert actual.observation_count.eq(len(sample)).all()
        for row in actual.itertuples():
            assert row.value == pytest.approx(
                expected[row.coefficient_id], rel=1e-13, abs=1e-14
            )
    assert records.loc[records.coefficient_id.eq("x"), "value"].nunique() == 2
    frame.loc[0, "total_market_cap"] = 0.0
    with pytest.raises(ValueError, match="fama_macbeth_weights_invalid"):
        statistics_production._fama_macbeth_period_records(
            frame,
            return_column="y",
            factors=("x",),
            core_factors=(),
        )
