"""Independent analysis results retain only their actual prerequisites."""

from copy import deepcopy
from shutil import copytree
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.research import statistics_production as production
from quant_research_platform.research.evaluations.runner import (
    _evaluation_source_facts,
    evaluate,
)
from quant_research_platform.research.statistics import time_series_inference_summary
from tests.research.test_stratified_evaluation_reuse import _fixture


def test_hac_failure_keeps_mean_and_independent_bootstrap(tmp_path):
    frame = pd.DataFrame({production._PREDICTION_COLUMN: [1.0, 2.0, 4.0]})
    result = production._prediction_summary(
        frame,
        prediction_id="one",
        inference_blocker=None,
        max_lags=20,
        block_size=2,
        repetitions=30,
        seed=1,
    )
    assert result.status.iloc[0] == "UNDEFINED"
    assert result.bootstrap_status.iloc[0] == "DEFINED"
    assert result["mean"].iloc[0] == pytest.approx(7 / 3)
    root = write_statistics_artifact(
        tmp_path / "summary",
        metadata={
            "sample_role": "test",
            "window": {"start": "2020-01-01", "end": "2020-01-03"},
        },
        tables={"prediction_diagnostics": result},
    )
    with read_statistics_artifact(root).table(
        "prediction_diagnostics.parquet"
    ) as saved:
        assert saved.bootstrap_status.iloc[0] == "DEFINED"
        assert np.isfinite(saved.bootstrap_lower.iloc[0])


def test_bootstrap_failure_keeps_hac(tmp_path, monkeypatch):
    def undefined(*args, **kwargs):
        raise ValueError("bootstrap_inference_undefined")

    monkeypatch.setattr(production, "stationary_bootstrap_mean_ci", undefined)
    result = production._prediction_summary(
        pd.DataFrame({production._PREDICTION_COLUMN: [1.0, 2.0, 4.0]}),
        prediction_id="one",
        inference_blocker=None,
        max_lags=0,
        block_size=2,
        repetitions=30,
        seed=1,
    )
    assert result.status.iloc[0] == "DEFINED"
    assert result.bootstrap_status.iloc[0] == "UNDEFINED"
    root = write_statistics_artifact(
        tmp_path / "summary",
        metadata={
            "sample_role": "test",
            "window": {"start": "2020-01-01", "end": "2020-01-03"},
        },
        tables={"prediction_diagnostics": result},
    )
    with read_statistics_artifact(root).table(
        "prediction_diagnostics.parquet"
    ) as saved:
        assert np.isfinite(saved.standard_error.iloc[0])
        assert pd.isna(saved.bootstrap_lower.iloc[0])


@pytest.mark.parametrize("values,lag", [([1.0, 2.0, 4.0], 20), ([2.0, 2.0, 2.0], 0)])
def test_descriptive_mean_survives_undefined_hac(values, lag):
    result = time_series_inference_summary(
        pd.DataFrame(
            {"timestamp": pd.date_range("2020-01-01", periods=3), "x": values}
        ),
        identity_column="timestamp",
        max_lags=lag,
        fdr_method=None,
        fdr_alpha=0.05,
    )
    assert result.status.iloc[0] == "UNDEFINED"
    assert result["mean"].iloc[0] == pytest.approx(np.mean(values))


def test_explicit_own_is_identical_to_default_daily_facts(tmp_path):
    _, config = _fixture(tmp_path, days=2)
    config.pop("strata")
    explicit = evaluate(config)
    config.pop("sample_scope")
    implicit = evaluate({**config, "output": str(tmp_path / "implicit")})
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )

    left_metadata, left = read_evaluation_outputs(
        explicit, tables=("metrics", "daily_coverage")
    )
    right_metadata, right = read_evaluation_outputs(
        implicit, tables=("metrics", "daily_coverage")
    )
    assert left_metadata["metadata"] == right_metadata["metadata"]
    for member in left:
        pd.testing.assert_frame_equal(left[member], right[member])


def test_relocated_source_can_reduce_existing_daily_facts(tmp_path):
    _, config = _fixture(tmp_path)
    own = evaluate(config)
    moved = tmp_path / "relocated"
    copytree(config["inputs"], moved)
    common = evaluate(
        {
            **config,
            "inputs": str(moved),
            "sample_scope": "common",
            "output": str(tmp_path / "common"),
        }
    )
    root = production.factor_statistics(
        {
            "evaluation_inputs": [str(own), str(common)],
            "output": str(tmp_path / "statistics"),
            "analyses": ["stratified_ic"],
            "statistics": {"hac_lag": 20, "fdr_method": "BY", "fdr_alpha": 0.05},
        }
    )
    artifact = read_statistics_artifact(root)
    assert artifact.metadata["hac_lag_unit"] == "supplied_complete_input_date_clock"
    with artifact.table("factor_stratified_diagnostics.parquet") as saved:
        assert set(saved.sample_scope) == {"own", "common"}


def test_evaluation_science_ignores_unconsumed_transform_and_keeps_borrowed(tmp_path):
    transform = {
        "recipe_id": "a_share_canonical",
        "source_columns": ["x", "unused"],
        "center": True,
    }
    spec = {"column_roles": {"feature_x": "feature"}, "feature_transforms": [transform]}
    original = SimpleNamespace(spec=spec, manifest={})
    borrowed = SimpleNamespace(
        spec={**spec, "feature_transforms": []},
        manifest={
            "producer_diagnostics": {
                "reused_feature_transforms": {"feature_x": transform}
            }
        },
    )
    kwargs = dict(columns=("feature_x",), signals=("feature_x",), availability={})
    left = _evaluation_source_facts(str(tmp_path / "a"), original, **kwargs)
    right = _evaluation_source_facts(str(tmp_path / "b"), borrowed, **kwargs)
    assert (
        left["feature_transforms"]
        == right["feature_transforms"]
        == [{**transform, "source_columns": ["x"]}]
    )
    changed = deepcopy(borrowed)
    changed.manifest["producer_diagnostics"]["reused_feature_transforms"]["feature_x"][
        "center"
    ] = False
    assert (
        left["feature_transforms"]
        != _evaluation_source_facts(str(tmp_path / "b"), changed, **kwargs)[
            "feature_transforms"
        ]
    )


@pytest.mark.parametrize(
    "analysis,policy,extra,members,seed_mode",
    [
        (*case, seed_mode)
        for case in [
            (
                "model_losses",
                {},
                {},
                {"model_losses.parquet", "model_prediction_metrics.parquet"},
            ),
            ("prediction_hac", {"hac_lag": 0}, {}, {"prediction_diagnostics.parquet"}),
            (
                "prediction_bootstrap",
                {"bootstrap_block": 1, "bootstrap_repetitions": 10},
                {"root_seed": 1},
                {"prediction_diagnostics.parquet"},
            ),
            (
                "model_confidence_set",
                {"bootstrap_block": 1, "bootstrap_repetitions": 10},
                {"root_seed": 1},
                {"model_reality_checks.parquet", "robustness.parquet"},
            ),
            (
                "spa_reality_check",
                {"bootstrap_block": 1, "bootstrap_repetitions": 10},
                {"root_seed": 1, "comparison_model_id": "linear_regression"},
                {"model_reality_checks.parquet", "robustness.parquet"},
            ),
        ]
        for seed_mode in ("override", "omitted", "explicit42", "null")
        if not (
            case[0] in {"model_losses", "prediction_hac"} and seed_mode == "override"
        )
    ],
)
def test_public_model_analysis_subsets_require_only_used_inputs(
    tmp_path, analysis, policy, extra, members, seed_mode
):
    from tests.research.test_prediction_runner_model_selection import candidates

    _, scoring = candidates(tmp_path)
    config = {
        "dataset": scoring["labels"],
        "input_spec": scoring["label_spec"],
        "output": str(tmp_path / "statistics"),
        "window": scoring["scoring_window"],
        "label_cutoff": "2020-01-10",
        "family_ids": scoring["candidate_ids"],
        "evaluation": {"primary_label": "label"},
        "statistics": policy,
        "analyses": [analysis],
        **extra,
        "periods": [
            {
                "period_id": "explicit",
                "window": scoring["scoring_window"],
                "predictions": scoring["predictions"],
            }
        ],
    }
    if seed_mode != "override":
        config.pop("root_seed", None)
        if seed_mode in {"explicit42", "null"}:
            config["root_seed"] = 42 if seed_mode == "explicit42" else None
    randomized = analysis not in {"model_losses", "prediction_hac"}
    if randomized and seed_mode == "null":
        with pytest.raises(ValueError, match="model_statistics_config_invalid"):
            production.model_statistics(config)
        return
    artifact = read_statistics_artifact(production.model_statistics(config))
    assert set(artifact._members) == members
    if "prediction_diagnostics.parquet" in members:
        with artifact.table("prediction_diagnostics.parquet") as saved:
            assert set(saved.prediction_id) == set(scoring["candidate_ids"])
            if analysis == "prediction_hac":
                assert saved.bootstrap_reason.eq("not_requested").all()
                assert saved.bootstrap_seed.isna().all()
            else:
                from quant_research_platform.research.randomness import derive_seed

                assert saved.reason.eq("not_requested").all()
                root_seed = 1 if seed_mode == "override" else 42
                assert saved.bootstrap_seed.tolist() == [
                    derive_seed(
                        root_seed,
                        "a_share_statistics_bootstrap",
                        scoring["label_spec"]["data_id"],
                        identity,
                    )
                    for identity in saved.prediction_id
                ]
    if "model_reality_checks.parquet" in members:
        with artifact.table("model_reality_checks.parquet") as saved:
            assert set(saved.test_id) == {analysis}


@pytest.mark.parametrize("change", ["assets", "labels", "row_order"])
def test_saved_evaluation_pair_uses_declared_science_and_daily_values(tmp_path, change):
    from tests.research.crypto_evaluation_test_support import write_evaluation_input

    frame, config = _fixture(tmp_path)
    own = evaluate(config)
    if change == "assets":
        frame["asset_id"] = list("wxyz")
    elif change == "labels":
        frame["label"] *= -1
    else:
        frame = frame.iloc[::-1]
    other = write_evaluation_input(
        tmp_path / "other_input", frame, factors=("x", "z"), labels=("label",)
    )
    common = evaluate(
        {
            **config,
            "inputs": str(other),
            "sample_scope": "common",
            "output": str(tmp_path / "common"),
        }
    )
    reduction = {
        "evaluation_inputs": [str(own), str(common)],
        "output": str(tmp_path / "statistics"),
        "analyses": ["stratified_ic"],
        "statistics": {"hac_lag": 20, "fdr_method": "BY", "fdr_alpha": 0.05},
    }
    # These are separate daily-statistic series, not paired raw observations.
    # Selecting their original inputs belongs to the caller.
    output = production.factor_statistics(reduction)
    with read_statistics_artifact(output).table(
        "factor_stratified_diagnostics.parquet"
    ) as result:
        rows = result.loc[
            result.stratification.eq("all") & result.statistic.eq("rankic")
        ]
        assert rows.observation_pairs.tolist() == [3, 3]
        actual = rows.set_index("sample_scope")["mean"].to_dict()
        assert actual["own"] == pytest.approx(0.5)
        assert actual["common"] == pytest.approx(-0.5 if change == "labels" else 0.5)


def test_paired_group_requires_its_bootstrap_inputs_before_reading(tmp_path):
    with pytest.raises(ValueError, match="model_statistics_policy_invalid"):
        production.model_statistics(
            {
                "dataset": "missing",
                "output": str(tmp_path / "statistics"),
                "window": {"start": "2020-01-01", "end": "2020-01-02"},
                "evaluation": {"primary_label": "label"},
                "analyses": ["model_losses"],
                "statistics": {
                    "hac_lag": 0,
                    "group_comparison": {
                        "fdr_method": "BY",
                        "alpha": 0.05,
                        "seed": 1,
                        "confidence": 0.95,
                    },
                },
                "periods": [],
                "global_predictions": ["missing"],
                "selection_group": "quarterly",
            }
        )


def test_borrowed_forecast_age_transform_remains_available_to_strata(tmp_path):
    import json
    from pathlib import Path

    from tests.research.test_stratified_evaluation_reuse import _write_native

    frame, config = _fixture(tmp_path)
    age = "median_forecast_age_days_fy1"
    feature = "feature_" + age
    frame[age] = [1.0, 10.0, 40.0, 80.0]
    frame[feature] = [-1.0, -0.5, 0.5, 1.0]
    source = _write_native(
        tmp_path / "borrowed",
        frame,
        signals=(feature,),
        labels=("label",),
        availability={"label": "label_available_at"},
        units={age: "days"},
    )
    manifest_path = Path(source) / "_manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.setdefault("producer_diagnostics", {})["reused_feature_transforms"] = {
        feature: {"recipe_id": "a_share_canonical", "source_columns": [age]}
    }
    manifest_path.write_text(json.dumps(manifest))
    config["inputs"] = source
    config["evaluation"]["signals"] = [
        {"signal_id": "age", "source_kind": "feature", "source_column": feature}
    ]
    config["strata"] = {
        "primary_label": "label",
        "kinds": ["forecast_age"],
        "forecast_freshness_days": 90,
    }
    assert evaluate(config).exists()


@pytest.mark.parametrize(
    "encoding", ["categories", "missing_categories", "string", "float"]
)
def test_native_evaluation_ignores_null_payload_and_dictionary_encoding(
    tmp_path, encoding
):
    from quant_research_platform.artifacts.outputs.evaluation import (
        read_evaluation_outputs,
    )
    from tests.research.crypto_evaluation_test_support import write_evaluation_input

    frame, config = _fixture(tmp_path)
    if encoding == "categories":
        frame["industry_level1_code"] = pd.Categorical(
            ["A", None, "B", "A"], categories=["A", "B"]
        )
    else:
        frame["industry_level1_code"] = pd.Series([None] * len(frame), dtype="string")
    config["inputs"] = str(
        write_evaluation_input(
            tmp_path / "reference", frame, factors=("x", "z"), labels=("label",)
        )
    )
    own = evaluate(config)
    frame.loc[3, "x"] = np.array([0x7FF8000000000001], dtype="uint64").view("float64")[
        0
    ]
    if encoding == "categories":
        frame["industry_level1_code"] = (
            frame.industry_level1_code.cat.reorder_categories(["B", "A"])
        )
    elif encoding == "missing_categories":
        frame["industry_level1_code"] = pd.Categorical(
            [None] * len(frame), categories=["A", "B"]
        )
    else:
        frame["industry_level1_code"] = frame.industry_level1_code.astype(
            {"string": "string", "float": "float64"}[encoding]
        )
    alternative = write_evaluation_input(
        tmp_path / "alternative",
        frame.iloc[::-1, ::-1],
        factors=("x", "z"),
        labels=("label",),
    )
    alternative_eval = evaluate(
        {
            **config,
            "inputs": str(alternative),
            "output": str(tmp_path / "common"),
            "sample_scope": "own",
        }
    )
    tables = ("metrics", "daily_coverage", "daily_strata")
    own_tables = read_evaluation_outputs(own, tables=tables)[1]
    alternative_tables = read_evaluation_outputs(alternative_eval, tables=tables)[1]
    for name in tables:
        keys = [
            column
            for column in ("timestamp", "signal", "label", "stratification", "stratum")
            if column in own_tables[name]
        ]
        pd.testing.assert_frame_equal(
            own_tables[name].sort_values(keys).reset_index(drop=True),
            alternative_tables[name].sort_values(keys).reset_index(drop=True),
            check_exact=True,
        )


def test_public_prediction_inference_is_local_to_consumed_family(tmp_path):
    from quant_research_platform.artifacts.outputs.prediction import (
        read_predictions,
        write_prediction_outputs,
    )
    from tests.research.test_prediction_runner_model_selection import candidates

    _, scoring = candidates(tmp_path)
    paths = []
    for index, path in enumerate(scoring["predictions"]):
        frame, metadata = read_predictions(path)
        # Nonconstant daily RankIC makes HAC defined for the causal family.
        middle = frame.timestamp.eq(pd.Timestamp("2020-01-05"))
        for column in ("prediction_score", "prediction_rank_score"):
            frame.loc[middle, column] *= -1
        metadata["model_facts"]["selection_cutoff"] = "2020-01-03"
        if index:
            # This family was selected during the requested evaluation period,
            # but its own saved forecasts still follow its actual fitting time.
            metadata["model_facts"]["selection_cutoff"] = "2020-01-04"
            metadata["model_facts"]["fit_cutoff"] = "2020-01-04"
            metadata["prediction_window"]["start"] = "2020-01-05"
            frame = frame.loc[frame.timestamp.ge(pd.Timestamp("2020-01-05"))]
            metadata.pop("prediction_rows")
        paths.append(
            str(
                write_prediction_outputs(
                    output_path=tmp_path / f"saved-{index}",
                    output_chunks=[frame],
                    metadata=metadata,
                )
            )
        )
    config = {
        "dataset": scoring["labels"],
        "input_spec": scoring["label_spec"],
        "window": scoring["scoring_window"],
        "family_ids": scoring["candidate_ids"],
        "comparison_model_id": scoring["candidate_ids"][0],
        "evaluation": {"primary_label": "label"},
        "analyses": [
            "prediction_hac",
            "prediction_bootstrap",
            "model_comparison",
            "spa_reality_check",
        ],
        "statistics": {"hac_lag": 0, "bootstrap_block": 1, "bootstrap_repetitions": 10},
        "periods": [
            {
                "period_id": "explicit",
                "window": scoring["scoring_window"],
                "predictions": paths,
            }
        ],
    }
    both = read_statistics_artifact(
        production.model_statistics({**config, "output": str(tmp_path / "both")})
    )
    single = read_statistics_artifact(
        production.model_statistics(
            {
                **config,
                "output": str(tmp_path / "single"),
                "analyses": ["prediction_hac", "prediction_bootstrap"],
                "family_ids": config["family_ids"][:1],
                "periods": [{**config["periods"][0], "predictions": paths[:1]}],
            }
        )
    )
    with (
        both.table("prediction_diagnostics.parquet") as mixed,
        single.table("prediction_diagnostics.parquet") as alone,
    ):
        causal = mixed.loc[mixed.prediction_id.eq(config["family_ids"][0])]
        pd.testing.assert_frame_equal(
            causal.reset_index(drop=True), alone, check_exact=True
        )
        assert causal.status.tolist() == ["DEFINED"]
        assert causal.bootstrap_status.tolist() == ["DEFINED"]
        retrospective = mixed.loc[mixed.prediction_id.eq(config["family_ids"][1])]
        assert retrospective.reason.tolist() == ["selection_reuses_evaluation_window"]
        assert retrospective.bootstrap_reason.tolist() == [
            "selection_reuses_evaluation_window"
        ]
        assert retrospective["mean"].notna().all()
    for member in ("model_comparison.parquet", "model_reality_checks.parquet"):
        with both.table(member) as joint:
            assert joint.status.eq("UNDEFINED").all()
            assert joint.reason.eq("selection_reuses_evaluation_window").all()


@pytest.mark.parametrize(
    "retrospective", ["unselected_family", "selected_family", "selection"]
)
def test_selected_prediction_checks_only_its_selection_path(tmp_path, retrospective):
    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.storage.datasets.reading import open_dataset_session
    from tests.helpers.statistics_artifacts import (
        model_statistics_inputs,
        statistics_config,
    )
    from tests.research.test_direct_statistics import _factor_tables
    from tests.research.test_statistics_production import _PredictionLookup

    _, metadata, dates = _factor_tables(tmp_path)
    session = open_dataset_session(tmp_path / "input")
    from tests.helpers.research_specs import make_demo_research_dataset_spec

    spec = make_demo_research_dataset_spec(
        data_id="direct_factor",
        factor_columns=("factor",),
        label_columns=("forward_return_20d", "forward_return_20d_available_at"),
        filter_columns=("eligible",),
        context_columns=("total_market_cap",),
    )
    config = statistics_config(factors=["factor"])
    config["statistics"] = {
        "hac_lag": 0,
        "bootstrap_block": 1,
        "bootstrap_repetitions": 10,
    }
    config["analyses"] = ["prediction_hac", "prediction_bootstrap", "model_comparison"]
    config["label_cutoff"] = "2021-01-01"
    window = TimeWindow(dates[0].date(), dates[-1].date())
    inputs = model_statistics_inputs(spec, config, window, families=("chosen", "other"))
    if retrospective == "selection":
        inputs["selections"]["explicit_period"]["cutoff"] = str(dates[-1])
    else:
        family = "chosen" if retrospective == "selected_family" else "other"
        inputs["model_entries"]["explicit_period", family]["model_facts"][
            "selection_cutoff"
        ] = str(dates[-1])
    tables = production.build_a_share_statistics_tables(
        dataset_path=tmp_path / "input",
        dataset_spec=spec,
        dataset_session=session,
        prediction_id="chosen-path",
        config=config,
        time_window=window,
        prediction_lookup=_PredictionLookup(),
        **inputs,
    )
    artifact = read_statistics_artifact(
        write_statistics_artifact(
            tmp_path / "selected-path",
            metadata=metadata,
            tables=tables,
        )
    )
    with artifact.table("prediction_diagnostics.parquet") as saved:
        if retrospective == "unselected_family":
            assert saved.status.tolist() == ["DEFINED"]
            assert saved.bootstrap_status.tolist() == ["DEFINED"]
        else:
            assert saved.reason.tolist() == ["selection_reuses_evaluation_window"]
            assert saved.bootstrap_reason.tolist() == [
                "selection_reuses_evaluation_window"
            ]
    with artifact.table("model_comparison.parquet") as joint:
        assert joint.reason.eq("selection_reuses_evaluation_window").all()
