from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import (
    prediction_model_metadata,
)
from quant_research_platform.artifacts.statistics import (
    STATISTICS_TABLE_GROUPS,
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.json_codec import canonical_json_hash
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research import statistics_production
from quant_research_platform.research.protocols import a_share
from quant_research_platform.research.randomness import derive_seed
from quant_research_platform.research.statistics_production import (
    build_a_share_factor_statistics_tables,
    build_a_share_statistics_tables,
)
from quant_research_platform.storage.datasets import write_dataset
from tests.helpers.paths import StorageLayout
from tests.helpers.research_configuration import (
    A_SHARE_MODEL_ALGORITHMS,
    CORE6_FACTORS,
    PV25_FACTORS,
)
from tests.helpers.research_specs import make_demo_research_dataset_spec
from tests.helpers.statistics_artifacts import (
    model_statistics_inputs,
    publish_statistics_pair,
    statistics_config,
    statistics_metadata,
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


@pytest.mark.parametrize("full_factor_history", (False, True))
def test_family_comparison_production_matches_arch_on_persisted_common_date_losses(
    tmp_path: Path, full_factor_history: bool
):
    from arch.bootstrap import MCS, SPA, StepM

    layout = StorageLayout(tmp_path)
    config = statistics_config()
    config["statistics"]["factor_fdr_families"] = {"fixture": ["x"]}
    # Native-backend parity needs reproducible draws, not a research-sized run.
    config["statistics"]["bootstrap_repetitions"] = 32 if full_factor_history else 199
    config["statistics"]["fama_macbeth_core_factors"] = ["x"]
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
        label_columns=(a_share.PRIMARY_LABEL_COLUMN,),
        filter_columns=("eligible",),
        context_columns=(
            "total_market_cap",
            "forward_return_20d_available_at",
        ),
    )
    if full_factor_history:
        historical = frame.iloc[: len(assets)].copy()
        historical["timestamp"] = pd.Timestamp("2014-01-02")
        historical["forward_return_20d_available_at"] = pd.Timestamp("2014-02-03")
        frame = pd.concat((historical, frame), ignore_index=True)
    frame["total_market_cap"] = 1.0 + frame.groupby("timestamp").cumcount()
    write_dataset(layout.research_dataset_root("a_share", "pv_all"), frame, spec=spec)
    model = model_statistics_inputs(spec, config, TimeWindow(dates[0], cutoff))
    for selection in model["selections"].values():
        selection["cutoff"] = str(cutoff)
    metadata = {
        **statistics_production._model_target_metadata(
            config, spec, TimeWindow(dates[0], cutoff)
        ),
        "selection_cutoffs": [str(cutoff)],
        "sample_role": "validation",
        "window": {"start": str(dates[0]), "end": str(cutoff)},
    }
    factor_metadata = {
        **metadata,
        "window": {
            **metadata["window"],
            "start": "2014-01-02 00:00:00"
            if full_factor_history
            else metadata["window"]["start"],
        },
    }
    statistics_tables = build_a_share_statistics_tables(
        dataset_path=layout.research_dataset_root("a_share", spec.data_id),
        dataset_spec=spec,
        config=config,
        time_window=TimeWindow(dates[0], cutoff),
        prediction_id="prediction",
        prediction_lookup=Lookup(),
        global_prediction_lookup=None,
        **model,
    )
    statistics_tables.update(
        build_a_share_factor_statistics_tables(
            dataset_path=layout.research_dataset_root("a_share", spec.data_id),
            dataset_spec=spec,
            config=config,
            time_window=TimeWindow(
                pd.Timestamp("2014-01-02") if full_factor_history else dates[0], cutoff
            ),
        )
    )
    global_tables = {name: table.copy() for name, table in statistics_tables.items()}
    assert global_tables["model_group_comparison"].empty
    factor_root, root = publish_statistics_pair(
        layout.project_root / "pv_all_global_statistics",
        statistics_tables,
        research_metadata=metadata,
        factor_metadata=factor_metadata,
    )
    artifact = read_statistics_artifact(root)
    factor_artifact = read_statistics_artifact(factor_root)
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
    blocked_reason = "selection_reuses_evaluation_window"
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
    quarterly_config = {**config, "selection_group": "quarterly"}
    for selection in model["selections"].values():
        selection["cutoff"] = str(dates[0] - pd.Timedelta(days=1))
    paired_metadata = metadata
    factor_paired_metadata = factor_metadata
    paired_tables = build_a_share_statistics_tables(
        dataset_path=layout.research_dataset_root("a_share", spec.data_id),
        dataset_spec=spec,
        config=quarterly_config,
        time_window=TimeWindow(dates[0], cutoff),
        prediction_id="quarterly_prediction",
        prediction_lookup=Lookup(),
        global_prediction_lookup=Lookup(),
        global_inference_blocker="selection_reuses_evaluation_window",
        **model,
    )
    paired_tables.update(
        build_a_share_factor_statistics_tables(
            dataset_path=layout.research_dataset_root("a_share", spec.data_id),
            dataset_spec=spec,
            config=quarterly_config,
            time_window=TimeWindow(dates[0], cutoff),
        )
    )
    factor_paired_root, paired_root = publish_statistics_pair(
        layout.project_root / "pv_all_quarterly_statistics",
        paired_tables,
        research_metadata=paired_metadata,
        factor_metadata=factor_paired_metadata,
    )
    paired_artifact = read_statistics_artifact(paired_root)
    read_statistics_artifact(factor_paired_root)
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
                    42, "a_share_model_reality_check", "pv_all", "validation", test_id
                )
                options = dict(
                    block_size=20,
                    reps=config["statistics"]["bootstrap_repetitions"],
                    seed=seed,
                )
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
        assert paired.reason.eq("selection_reuses_evaluation_window").all()
        assert paired.bootstrap_reason.eq(blocked_reason).all()
    if not full_factor_history:
        robustness_config = {
            **quarterly_config,
            "training_weighting": {"half_life_periods": 756, "role": "robustness"},
        }
        robustness_tables = build_a_share_statistics_tables(
            dataset_path=layout.research_dataset_root("a_share", spec.data_id),
            dataset_spec=spec,
            config=robustness_config,
            time_window=TimeWindow(dates[0], cutoff),
            prediction_id="quarterly_prediction",
            prediction_lookup=Lookup(),
            global_prediction_lookup=Lookup(),
            global_inference_blocker="selection_reuses_evaluation_window",
            **model,
        )
        robustness_tables.update(
            build_a_share_factor_statistics_tables(
                dataset_path=layout.research_dataset_root("a_share", spec.data_id),
                dataset_spec=spec,
                config=robustness_config,
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
        factor_robust_root, robust_root = publish_statistics_pair(
            layout.project_root / "pv_all_robust_statistics",
            robustness_tables,
            research_metadata=paired_metadata,
            factor_metadata=factor_paired_metadata,
        )
        robust_artifact = read_statistics_artifact(robust_root)
        factor_robust_artifact = read_statistics_artifact(factor_robust_root)
        with factor_robust_artifact.table("factor_diagnostics.parquet") as table:
            assert table.status.eq("DEFINED").all()
            assert table.fdr_method.eq("NOT_APPLICABLE").all()
            assert table.p_value.notna().all()
            assert table.fdr_adjusted_p_value.isna().all()
        with robust_artifact.table("model_reality_checks.parquet") as table:
            assert table.value.isna().all()
            assert table.reason.eq("training_weighting_robustness_excluded").all()


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
        sample_role="validation",
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
        for regression in ["single:factor"] * 2
        + (["core_multifactor"] * 2 if core_factors else [])
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
            label_columns=(a_share.PRIMARY_LABEL_COLUMN,),
            filter_columns=("eligible",),
            context_columns=(
                "total_market_cap",
                "forward_return_20d_available_at",
            ),
            filter_column_roles={"eligible": ColumnRole.UNIVERSE},
        )
        frame = pd.DataFrame(rows)
        frame["total_market_cap"] = 1.0 + frame.groupby("timestamp").cumcount()
        write_dataset(
            layout.research_dataset_root(spec.market_id, spec.data_id), frame, spec=spec
        )
        from quant_research_platform.artifacts.outputs.evaluation import (
            read_evaluation_outputs,
        )
        from quant_research_platform.research.evaluations.runner import evaluate

        evaluation_path = evaluate(
            {
                "inputs": str(layout.research_dataset_root("a_share", spec.data_id)),
                "output": str(root / "evaluation"),
                "window": {"start": str(dates[0]), "end": str(cutoff)},
                "cutoff": str(cutoff),
                "sample_role": "validation",
                "evaluation": {
                    "evaluation_id": "stats_demo_evaluation",
                    "market_id": "a_share",
                    "research_dataset_id": spec.data_id,
                    "signals": [
                        {
                            "signal_id": CORE6_FACTORS[0],
                            "source_kind": "factor",
                            "source_column": CORE6_FACTORS[0],
                        }
                    ],
                    "label_columns": [a_share.PRIMARY_LABEL_COLUMN],
                    "label_availability_columns": {
                        a_share.PRIMARY_LABEL_COLUMN: "forward_return_20d_available_at"
                    },
                },
            }
        )
        _, summaries = read_evaluation_outputs(evaluation_path)
        evaluation_summaries.append(summaries["ic_summary"].copy())
        config = statistics_config()
        config["statistics"]["factor_fdr_families"] = {
            "price_volume": list(PV25_FACTORS)
        }
        config["statistics"]["fama_macbeth_core_factors"] = list(CORE6_FACTORS)
        model = model_statistics_inputs(
            spec,
            config,
            TimeWindow(dates[0], cutoff),
            selected="stats_demo_equal_weight_rank_ensemble",
        )
        for entry in model["model_entries"].values():
            entry["model_metadata"]["input_columns"] = [CORE6_FACTORS[0]]
        for selection in model["selections"].values():
            selection["cutoff"] = str(cutoff)
        statistics_tables = build_a_share_statistics_tables(
            dataset_path=layout.research_dataset_root("a_share", spec.data_id),
            dataset_spec=spec,
            config=config,
            time_window=TimeWindow(dates[0], cutoff),
            prediction_id="prediction",
            prediction_lookup=_PredictionLookup(),
            global_prediction_lookup=None,
            **model,
        )
        statistics_tables.update(
            build_a_share_factor_statistics_tables(
                dataset_path=layout.research_dataset_root("a_share", spec.data_id),
                dataset_spec=spec,
                config=config,
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
    assert set(factors["defined_hypothesis_count"]) == {0}
    assert factors.reason.eq("incomplete_session_clock").all()
    assert factors.loc[factors.observation_count.eq(20), "mean"].notna().all()
    assert set(factors["missing_hypothesis_count"]) == {len(PV25_FACTORS)}
    prediction = outputs[0]["prediction_diagnostics"].iloc[0]
    assert prediction["observation_count"] == 20
    assert prediction["status"] == "UNDEFINED"
    assert prediction["reason"] == "selection_reuses_evaluation_window"
    assert prediction["mean"] == pytest.approx(0.2, abs=1e-15)
    assert prediction["bootstrap_seed"] == derive_seed(
        42, "a_share_statistics_bootstrap", "stats_demo", "prediction"
    )
    fmb = outputs[0]["fama_macbeth"]
    assert fmb.reason.eq("incomplete_session_clock").all()
    assert fmb.loc[fmb.period_count.eq(20), "mean"].notna().all()
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
        "selection_reuses_evaluation_window"
    }
    metadata = {
        "sample_role": "validation",
        "window": {"start": "2023-01-02", "end": str(cutoff.date())},
    }
    factor_metadata = metadata
    factor_artifact_root, artifact_root = publish_statistics_pair(
        tmp_path / "statistics",
        outputs[0],
        research_metadata=metadata,
        factor_metadata=factor_metadata,
    )
    artifact = read_statistics_artifact(artifact_root)
    factor_artifact = read_statistics_artifact(factor_artifact_root)
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
        assert persisted_summary.status.eq("UNDEFINED").all()
        assert persisted_summary.reason.eq("incomplete_session_clock").all()
        # The declared 25-date window has five unavailable dates. Preserve the
        # HAC oracle separately on an explicitly bounded complete 20-date input.
        complete_summary = (
            statistics_production._fama_macbeth_summary(
                persisted_periods,
                factors=(CORE6_FACTORS[0],),
                core_factors=(),
                max_lags=19,
                input_date_count=20,
            )
            .query("regression_id == @persisted_periods.regression_id.iloc[0]")
            .set_index("coefficient_id")
        )
        for coefficient, values in expected_periods.items():
            assert persisted_summary.loc[coefficient, "mean"] == pytest.approx(
                np.mean(values)
            )
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
                complete_summary.loc[coefficient, fields].astype(float),
                expected,
                rtol=1e-11,
                atol=1e-14,
            )
            assert se != pytest.approx(np.std(values, ddof=0) / np.sqrt(20))
    factor_zero_root, zero_root = publish_statistics_pair(
        tmp_path / "statistics_zero_residual",
        outputs[2],
        research_metadata=metadata,
        factor_metadata=factor_metadata,
    )
    zero_artifact = read_statistics_artifact(zero_root)
    factor_zero_artifact = read_statistics_artifact(factor_zero_root)
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
    factor_empty_root, empty_root = publish_statistics_pair(
        tmp_path / "statistics_empty_periods",
        empty_period_tables,
        research_metadata=metadata,
        factor_metadata=factor_metadata,
    )
    read_statistics_artifact(empty_root)
    factor_empty_artifact = read_statistics_artifact(factor_empty_root)
    with factor_empty_artifact.table("fama_macbeth_periods.parquet") as table:
        assert table.empty
    primary = KeyboardInterrupt("table_consumer_interrupted")
    artifact = read_statistics_artifact(artifact_root)
    factor_artifact = read_statistics_artifact(factor_artifact_root)
    with pytest.raises(KeyboardInterrupt) as caught:
        with artifact.table("model_losses.parquet") as table:
            assert not table.empty
            raise primary
    assert caught.value is primary
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    wrong_root = tmp_path / "bad_schema"
    shutil.copytree(artifact_root, wrong_root)
    pq.write_table(pa.table({"wrong": [1]}), wrong_root / "model_losses.parquet")
    unchecked = read_statistics_artifact(wrong_root)
    with pytest.raises(ValueError, match="statistics_table_schema_invalid"):
        with unchecked.table("model_losses.parquet"):
            pass


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
    "access,reused_selection",
    [("validation", True), ("validation", False), ("test", True), ("test", False)],
)
def test_model_inference_uses_selection_information_before_calling_backends(
    monkeypatch, access, reused_selection
):
    rng = np.random.default_rng(22019)
    values = pd.DataFrame(
        {
            "timestamp": pd.bdate_range("2023-01-02", periods=64),
            "baseline": rng.normal(0.01, 0.04, 64),
            "challenger": rng.normal(0.02, 0.04, 64),
        }
    )
    blocker = "selection_reuses_evaluation_window" if reused_selection else None
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
        sample_role=access,
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
        assert checks.status.eq("DEFINED").all()


@pytest.mark.parametrize("has_observations", (False, True))
def test_paired_model_rankic_consumes_explicit_comparison_config(has_observations):
    comparison_config = statistics_config()
    families = ("baseline", "challenger")
    columns = (
        "timestamp",
        "family_id",
        "asset_coverage_id",
        "observation_count",
        "sample_role",
        "loss",
        "status",
    )
    losses = pd.DataFrame.from_records(
        [
            (
                pd.Timestamp("2023-01-03"),
                family,
                "a" * 64,
                2,
                "validation",
                -0.5,
                "DEFINED",
            )
            for family in families
        ]
        if has_observations
        else [],
        columns=columns,
    )

    paired = statistics_production._paired_group_rankic(
        losses.assign(**{"global": -losses.loss, "quarterly": -losses.loss}),
        selection_group="quarterly",
        family_ids=families,
        config=comparison_config,
    )
    assert set(paired.family_id) == set(families)
    assert paired.observation_count.eq(int(has_observations)).all()
    assert paired.status.eq("UNDEFINED").all()
    assert paired.reason.eq(
        "hac_lag_exceeds_observations" if has_observations else "no_common_observations"
    ).all()
    if has_observations:
        assert paired.mean_difference.eq(0.0).all()
    else:
        assert paired.mean_difference.isna().all()


@pytest.mark.parametrize("access", ("validation", "test"))
@pytest.mark.parametrize(
    "family_id", ("momentum_20d", "__prediction_alpha__", "timestamp")
)
def test_model_family_names_preserve_factor_statistics_and_native_names(
    tmp_path: Path, family_id: str, access: str
) -> None:
    factor = "momentum_20d"
    dates = pd.bdate_range(
        "2024-01-02" if access == "test" else "2019-01-02", periods=3
    )
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
        label_columns=(a_share.PRIMARY_LABEL_COLUMN,),
        filter_columns=("eligible",),
        context_columns=(
            "total_market_cap",
            "forward_return_20d_available_at",
        ),
    )
    frame["total_market_cap"] = 1.0 + frame.groupby("timestamp").cumcount()
    write_dataset(
        layout.research_dataset_root("a_share", spec.data_id), frame, spec=spec
    )
    results = []
    for position, selected_family in enumerate((family_id, "renamed_model")):
        families = ("baseline", selected_family)
        config = statistics_config()
        config.update(selection_group="global", sample_role=access)
        config["evaluation"]["factors"] = [factor]
        config["comparison_model_id"] = families[0]
        config["statistics"].update(
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

        metadata = {
            **statistics_production._model_target_metadata(
                config,
                spec,
                TimeWindow(dates[0], cutoff),
            ),
            "sample_role": access,
            "window": {"start": str(dates[0]), "end": str(cutoff)},
        }
        factor_metadata = metadata
        statistics_tables = build_a_share_statistics_tables(
            dataset_path=layout.research_dataset_root("a_share", spec.data_id),
            dataset_spec=spec,
            config=config,
            time_window=TimeWindow(dates[0], cutoff),
            prediction_id="selected_prediction",
            prediction_lookup=Lookup(),
            global_prediction_lookup=None,
            **model_statistics_inputs(
                spec,
                config,
                TimeWindow(dates[0], cutoff),
                families=families,
                selection=selection_payload,
            ),
        )
        statistics_tables.update(
            build_a_share_factor_statistics_tables(
                dataset_path=layout.research_dataset_root("a_share", spec.data_id),
                dataset_spec=spec,
                config=config,
                time_window=TimeWindow(dates[0], cutoff),
            )
        )
        members = tuple(statistics_tables)
        factor_root, root = publish_statistics_pair(
            tmp_path / f"statistics_{position}",
            statistics_tables,
            research_metadata=metadata,
            factor_metadata=factor_metadata,
        )
        artifact = read_statistics_artifact(root)
        factor_artifact = read_statistics_artifact(factor_root)
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
        assert factor_diagnostics["mean"].tolist() == pytest.approx(
            [1.0] * len(factor_diagnostics)
        )
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
            quarterly_losses.assign(
                **{"global": -losses.loss, "quarterly": -quarterly_losses.loss}
            ),
            selection_group="quarterly",
            family_ids=families,
            config={**config, "selection_group": "quarterly"},
        )
        assert paired.family_id.tolist() == list(families)
        assert paired.observation_count.eq(len(dates)).all()
        assert paired.calendar_count.eq(len(dates)).all()
        assert paired.fdr_family_size.eq(len(families)).all()
        assert paired.mean_difference.tolist() == pytest.approx([-0.2, 0.2])
        assert paired.status.eq("DEFINED").all()
        assert paired.bootstrap_status.eq("DEFINED").all()
        assert paired.standard_error.notna().all()
        assert paired.fdr_adjusted_p_value.notna().all()
        assert paired.bootstrap_lower.notna().all()
        paired_tables = {
            **tables,
            "model_losses": quarterly_losses,
            "model_group_comparison": paired,
        }
        factor_paired_root, paired_root = publish_statistics_pair(
            tmp_path / f"paired_statistics_{position}",
            paired_tables,
            research_metadata=metadata,
            factor_metadata=factor_metadata,
        )
        paired_artifact = read_statistics_artifact(paired_root)
        read_statistics_artifact(factor_paired_root)
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
            frame, return_column="y", factors=("x",), core_factors=()
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
        x_mean, y_mean = (
            np.average(x, weights=weights),
            np.average(y, weights=weights),
        )
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
    for invalid_cap in (0.0, -1.0, np.inf, "vendor-invalid"):
        changed = frame.copy()
        changed["total_market_cap"] = changed.total_market_cap.astype(object)
        changed.loc[0, "total_market_cap"] = invalid_cap
        actual = pd.DataFrame(
            statistics_production._fama_macbeth_period_records(
                changed, return_column="y", factors=("x",), core_factors=()
            )
        )
        equal = actual.regression_id.str.startswith("equal:")
        pd.testing.assert_frame_equal(actual.loc[equal], records.loc[equal])
        assert actual.loc[~equal, "status"].eq("DEFINED").all()
        assert actual.loc[~equal, "observation_count"].eq(len(frame.dropna()) - 1).all()


def test_fmb_primary_horizon_uses_native_models_predictions_and_maturity(tmp_path):
    import pyarrow as pa

    from quant_research_platform.artifacts.outputs.prediction import (
        _PredictionLookup as NativeLookup,
    )
    from quant_research_platform.artifacts.outputs.prediction import read_predictions
    from quant_research_platform.research.models.persisted import load_model
    from quant_research_platform.research.models.production import fit
    from quant_research_platform.research.models.trial_selection import (
        read_selection,
        score,
        select,
    )
    from quant_research_platform.research.predictions.runner import predict
    from quant_research_platform.storage.exact_keys import ExactPrimaryKeyIndex

    factor = CORE6_FACTORS[0]
    dates = pd.bdate_range("2023-01-02", periods=48)
    cutoff = dates[44]
    window = TimeWindow(dates[0], cutoff)
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
                factor: float(asset - 5 + index * 0.001),
                "forward_return_5d": 0.1
                + 0.4 * asset
                + 0.03 * asset**2
                + 0.02 * np.sin(index + asset),
                "forward_return_20d": (80 - 5 * asset + 2 * np.cos(index + asset) - 100)
                / 100,
                "forward_return_5d_available_at": timestamp + pd.offsets.BDay(6),
                "forward_return_20d_available_at": timestamp + pd.offsets.BDay(21),
            }
            for index, timestamp in enumerate(history.append(dates))
            for asset in range(12)
        ]
    )
    for horizon in (5, 20):
        label = f"forward_return_{horizon}d"
        unavailable = frame[label + "_available_at"].gt(cutoff)
        frame.loc[unavailable, label] = np.nan
        frame.loc[unavailable, label + "_available_at"] = pd.NaT
    frame["total_market_cap"] = 1.0 + frame.groupby("timestamp").cumcount()
    spec = make_demo_research_dataset_spec(
        data_id="pv_all",
        factor_columns=(factor,),
        label_columns=(
            "forward_return_5d",
            "forward_return_20d",
        ),
        filter_columns=("eligible",),
        context_columns=(
            "total_market_cap",
            "forward_return_5d_available_at",
            "forward_return_20d_available_at",
        ),
    )
    source = tmp_path / "source"
    write_dataset(source, frame, spec=spec)
    mature_dates = {}
    targets = []
    for horizon in (5, 20):
        root = tmp_path / str(horizon)
        primary = f"forward_return_{horizon}d"
        model_id = "horizon_linear"
        model = fit(
            {
                "market_id": "a_share",
                "inputs": str(source),
                "output": str(root / "model"),
                "model": {
                    "model_id": model_id,
                    "model_algorithm": "linear_regression",
                    "input_factor_columns": [factor],
                    "target_column": primary,
                    "random_seed": 42,
                },
                "fit_windows": [{"start": str(history[0].date()), "end": "2022-12-31"}],
                "fit_cutoff": "2022-12-31",
                "label_unit": "decimal_return",
            }
        )
        fitted, facts = load_model(model)
        assert fitted.metadata["target_column"] == primary
        targets.append(fitted.metadata["target_column"])
        prediction_window = {"start": str(dates[0]), "end": str(cutoff)}
        predictions = predict(
            {
                "model": str(model),
                "inputs": str(source),
                "output": str(root / "prediction"),
                "prediction_window": prediction_window,
                "sample_role": "validation",
            }
        )
        predicted, prediction_meta = read_predictions(predictions)
        assert prediction_meta["model"] == str(model)
        assert len(predicted) == 45 * 12 and (
            not predicted.duplicated(["timestamp", "asset_id"]).any()
        )
        training = frame.loc[
            frame.timestamp.le("2022-12-31")
            & frame[primary + "_available_at"].le("2022-12-31")
            & frame[primary].notna()
        ]
        assert fitted.metadata["rows_after_input_label_dropna"] == len(training)
        # All admitted dates contain 12 assets, so equal-date OLS weights are uniform.
        assert training.groupby("timestamp").size().eq(12).all()
        design = np.column_stack((np.ones(len(training)), training[factor]))
        coefficients = np.linalg.lstsq(design, training[primary], rcond=None)[0]
        selected = predicted.merge(
            frame[["timestamp", "asset_id", factor]],
            on=["timestamp", "asset_id"],
            validate="one_to_one",
        )
        expected = coefficients[0] + coefficients[1] * selected[factor]
        np.testing.assert_allclose(
            selected.prediction_score, expected, rtol=1e-12, atol=1e-14
        )
        scored = score(
            {
                "predictions": [str(predictions)],
                "labels": str(source),
                "candidate_ids": [model_id],
                "validation_windows": [prediction_window],
                "scoring_window": prediction_window,
                "cutoff": str(cutoff),
                "selection_group": "global",
                "validation_axis": "cross_sectional",
                "output": str(root / "scores.json"),
                "label_unit": "decimal_return",
            }
        )
        selection_path = select(
            {
                "scores": str(scored),
                "output": str(root / "selection.json"),
                "selection_metric": "validation_rankic",
                "selection_direction": "maximize",
                "tie_break_order": [model_id],
                "refit_windows": [
                    {"start": str(history[0].date()), "end": "2022-12-31"},
                    prediction_window,
                ],
                "refit_cutoff": str(cutoff),
            }
        )
        selection = read_selection(selection_path)
        family = prediction_meta["family_id"]
        assert selection["selected_family_id"] == family
        config = statistics_config(factors=(factor,))
        config["evaluation"]["primary_label"] = primary
        config["evaluation"]["labels"] = ["forward_return_5d", "forward_return_20d"]
        config["comparison_model_id"] = family
        factor_window = TimeWindow(history[0], cutoff)
        factor_tables = build_a_share_factor_statistics_tables(
            dataset_path=source,
            dataset_spec=spec,
            config=config,
            time_window=factor_window,
        )
        index = ExactPrimaryKeyIndex(primary_key=("timestamp", "asset_id", "family_id"))
        lookup = NativeLookup(
            index=index,
            primary_key=("timestamp", "asset_id", "family_id"),
            prediction_column="prediction_score",
        )
        try:
            projected = predicted.assign(family_id=family)
            index.add_float_value_table(
                pa.Table.from_pandas(projected, preserve_index=False),
                namespace="prediction",
                value_column="prediction_score",
                validation_column="prediction_score",
                retain_values=True,
            )
            tables = build_a_share_statistics_tables(
                dataset_path=source,
                dataset_spec=spec,
                prediction_id="horizon_predictions",
                model_entries={
                    ("explicit_horizon", family): {
                        "family_id": family,
                        "model_id": model_id,
                        "model_metadata": prediction_model_metadata(prediction_meta),
                    }
                },
                selections={"explicit_horizon": selection},
                period_windows={"explicit_horizon": window},
                trial_rows=statistics_production._selection_trial_rows(
                    {"final": selection}, "global"
                ),
                config=config,
                time_window=window,
                prediction_lookup=lookup,
                global_prediction_lookup=None,
            )
            assert lookup.coverage_counts() == (0, 0)
        finally:
            lookup.close()
        write_statistics_artifact(
            root / "statistics", metadata=statistics_metadata(window), tables=tables
        )
        public_output = statistics_production.model_statistics(
            {
                **config,
                "evaluation": {"primary_label": primary},
                "statistics": {
                    key: config["statistics"][key]
                    for key in ("hac_lag", "bootstrap_block", "bootstrap_repetitions")
                },
                "dataset": str(source),
                "output": str(root / "public_statistics"),
                "window": prediction_window,
                "prediction_id": "horizon_predictions",
                "periods": [
                    {
                        "period_id": "explicit_horizon",
                        "window": prediction_window,
                        "selection": str(selection_path),
                        "predictions": [str(predictions)],
                    }
                ],
            }
        )
        with read_statistics_artifact(public_output).table(
            "model_losses.parquet"
        ) as actual:
            with read_statistics_artifact(root / "statistics").table(
                "model_losses.parquet"
            ) as expected_losses:
                pd.testing.assert_frame_equal(actual, expected_losses, check_exact=True)
            assert actual.period_id.eq("explicit_horizon").all()
        published = write_statistics_artifact(
            root / "factor_statistics",
            metadata=statistics_metadata(factor_window),
            tables=factor_tables,
            table_group="factor",
        )
        factor_statistics = read_statistics_artifact(published)
        expected = frame.loc[
            frame[primary + "_available_at"].le(cutoff) & frame.timestamp.le(cutoff)
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
                    for _, group in expected.groupby("timestamp", sort=True)
                ]
                np.testing.assert_allclose(actual.value, oracle, rtol=2e-14, atol=2e-14)
                assert actual.observation_count.eq(12).all()
        with factor_statistics.table("fama_macbeth.parquet") as summary:
            assert summary.period_count.eq(len(expected_dates)).all()
            assert summary.status.eq("UNDEFINED").all()
            assert summary.reason.eq("incomplete_session_clock").all()
            assert summary["mean"].notna().all()
        mature_dates[horizon] = expected_dates[expected_dates >= dates[0]]
    assert targets == ["forward_return_5d", "forward_return_20d"]
    assert len(mature_dates[5]) == 39 and len(mature_dates[20]) == 24
    assert len(mature_dates[5].difference(mature_dates[20])) == 15


@pytest.mark.parametrize(
    "invalid_cap",
    [True, np.bool_(True), 10**400, 1 + 2j, [1.0], "vendor-invalid", np.inf],
    ids=["bool", "numpy-bool", "oversized-int", "complex", "list", "text", "infinity"],
)
@pytest.mark.parametrize("asset_count", [3, 6])
def test_fmb_invalid_real_weights_only_remove_weighted_observations(
    invalid_cap, asset_count
):
    from quant_research_platform.research.statistics import fama_macbeth_period

    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-02"),
            "asset_id": list("abcdef")[:asset_count],
            "x": np.arange(asset_count, dtype=float),
            "y": [0.0, 1.0, 3.0, 2.0, 6.0, 9.0][:asset_count],
            "total_market_cap": pd.Series([1.0] * asset_count, dtype=object),
        }
    )
    baseline = pd.DataFrame(
        statistics_production._fama_macbeth_period_records(
            frame, return_column="y", factors=("x",), core_factors=()
        )
    )
    frame.at[0, "total_market_cap"] = invalid_cap
    actual = pd.DataFrame(
        statistics_production._fama_macbeth_period_records(
            frame, return_column="y", factors=("x",), core_factors=()
        )
    )
    equal = actual.regression_id.str.startswith("equal:")
    pd.testing.assert_frame_equal(
        actual.loc[equal].fillna(""), baseline.loc[equal].fillna(""), check_dtype=False
    )
    weighted = actual.loc[~equal]
    assert weighted.observation_count.eq(asset_count - 1).all()
    if asset_count == 3:
        assert weighted.status.eq("UNDEFINED").all()
        assert weighted.reason.eq("insufficient_complete_cross_section").all()
        with pytest.raises(ValueError, match="fama_macbeth_period_insufficient"):
            fama_macbeth_period(
                frame,
                return_column="y",
                factor_columns=("x",),
                weight_column="total_market_cap",
            )
    else:
        assert weighted.status.eq("DEFINED").all()
        expected = fama_macbeth_period(
            frame.iloc[1:],
            return_column="y",
            factor_columns=("x",),
            weight_column="total_market_cap",
        )
        result = fama_macbeth_period(
            frame,
            return_column="y",
            factor_columns=("x",),
            weight_column="total_market_cap",
        )
        pd.testing.assert_series_equal(result.coefficients, expected.coefficients)
        assert result.observation_count == expected.observation_count
        assert result.white_p_value == expected.white_p_value
        assert result.white_reason == expected.white_reason
