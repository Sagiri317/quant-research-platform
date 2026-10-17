"""Public-operation counterexamples for label-scale losses and declared FDR families."""

import json
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.models.explanations import explain
from quant_research_platform.research.models.trial_selection import score
from quant_research_platform.research.statistics_production import (
    _prediction_quality,
    factor_statistics,
    model_statistics,
)
from tests.research.direct_model_test_support import fit_frame, predict_frame


def _frame(*, intercept=0.1, slope=0.2):
    return pd.DataFrame(
        [
            dict(
                timestamp=date,
                asset_id=str(asset),
                x=float(asset),
                label=intercept + slope * ((asset + 1 - 2.5) / 4),
                eligible=True,
                label_available_at=date,
            )
            for date in pd.date_range("2020-01-01", periods=5)
            for asset in range(4)
        ]
    )


def _case(root, algorithm="linear_regression", *, intercept=0.1, slope=0.2):
    frame = _frame(intercept=intercept, slope=slope)
    case = fit_frame(
        root / "fit",
        frame.loc[frame.timestamp.le("2020-01-02")],
        ModelSpec(
            model_id="model",
            model_algorithm=algorithm,
            input_factor_columns=("x",),
            target_column="label",
        ),
        market_id="a_share",
    )
    window = {"start": "2020-01-03", "end": "2020-01-05"}
    _, prediction, metadata = predict_frame(root / "pred", frame, case, window=window)
    labels = root / "labels.parquet"
    frame.to_parquet(labels, index=False)
    config = {
        "dataset": str(labels),
        "input_spec": case.config["input_spec"],
        "output": str(root / "stats"),
        "window": window,
        "label_cutoff": "2020-01-06",
        "family_ids": ["model"],
        "analyses": ["model_losses"],
        "evaluation": {"primary_label": "label", "raw_return_label": "label"},
        "statistics": {},
        "periods": [
            {"period_id": "p", "window": window, "predictions": [str(prediction)]}
        ],
    }
    return frame, case, prediction, metadata, config


@pytest.mark.parametrize(
    "algorithm",
    ["equal_weight_rank_ensemble", "ic_weighted_rank_ensemble", "linear_regression"],
)
def test_frozen_label_scale_matches_selection_and_statistics(tmp_path, algorithm):
    frame, case, prediction, metadata, config = _case(tmp_path, algorithm)
    before, _ = read_predictions(prediction)
    scored = json.loads(
        score(
            {
                "predictions": [str(prediction)],
                "candidate_ids": ["model"],
                "labels": config["dataset"],
                "label_spec": config["input_spec"],
                "validation_windows": [config["window"]],
                "scoring_window": config["window"],
                "cutoff": "2020-01-06",
                "selection_group": "global",
                "validation_axis": "cross_sectional",
                "label_unit": "decimal_return",
                "output": str(tmp_path / "score.json"),
            }
        ).read_text()
    )
    artifact = read_statistics_artifact(model_statistics(config))
    with artifact.table("model_prediction_metrics.parquet") as metrics:
        assert metrics.mse.to_numpy() == pytest.approx(0.0, abs=1e-30)
        assert metrics.oos_r2.to_numpy() == pytest.approx(1.0)
        assert metrics.mse_reason.eq("").all()
        assert metrics.mse.mean() == pytest.approx(
            scored["metrics"][0]["validation_mse"], abs=1e-30
        )
        assert metrics.head_selected_count.eq(1).all()
    after, _ = read_predictions(prediction)
    pd.testing.assert_frame_equal(before, after)
    if "rank_ensemble" in algorithm:
        assert metadata["model_metadata"]["rank_parameters"]["label_calibration"] == [
            0.1,
            0.2,
        ]
        assert before.prediction_score.iloc[:4].tolist() == [
            -0.375,
            -0.125,
            0.125,
            0.375,
        ]
    else:
        assert "rank_parameters" not in metadata["model_metadata"]
        np.testing.assert_allclose(
            before.prediction_score,
            frame.loc[frame.timestamp.ge("2020-01-03"), "label"],
        )


@pytest.mark.parametrize(
    "algorithm", ["equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"]
)
def test_each_prediction_period_uses_its_own_training_calibration(tmp_path, algorithm):
    first, _, _, first_meta, config = _case(tmp_path / "first", algorithm)
    second, _, _, second_meta, other = _case(
        tmp_path / "second", algorithm, intercept=1.5, slope=3.0
    )
    labels = pd.concat(
        [
            first.loc[first.timestamp.eq("2020-01-03")],
            second.loc[second.timestamp.ge("2020-01-04")],
        ]
    )
    labels.to_parquet(config["dataset"], index=False)
    config["periods"][0]["window"] = {"start": "2020-01-03", "end": "2020-01-03"}
    config["periods"].append(
        {
            "period_id": "second",
            "window": {"start": "2020-01-04", "end": "2020-01-05"},
            "predictions": other["periods"][0]["predictions"],
        }
    )
    assert (
        first_meta["model_metadata"]["rank_parameters"]
        != second_meta["model_metadata"]["rank_parameters"]
    )
    with read_statistics_artifact(model_statistics(config)).table(
        "model_prediction_metrics.parquet"
    ) as metrics:
        assert metrics.mse.to_numpy() == pytest.approx(0.0, abs=1e-30)
        assert metrics.oos_r2.to_numpy() == pytest.approx(1.0)
        assert metrics.target_count.eq(4).all()


def test_negative_label_calibration_does_not_reverse_fixed_head():
    scores = pd.Series([-1.0, 0.0, 1.0])
    labels = 2 - 3 * scores
    result = _prediction_quality(
        scores, labels, labels, head_fraction=0.2, label_predictions=labels
    )
    assert result["mse"] == 0.0
    assert result["oos_r2"] == 1.0
    assert result["head_mean_raw_return"] == -1.0


@pytest.mark.parametrize(
    ("unit", "target", "reason"),
    [
        ("percent", "label", "prediction_target_unit_mismatch"),
        ("decimal_return", "other_label", "prediction_target_mismatch"),
        (None, "label", "target_unit_missing"),
    ],
)
def test_incomparable_loss_is_local_and_keeps_ic_and_head(
    tmp_path, unit, target, reason
):
    frame, _, _, _, config = _case(tmp_path)
    frame["raw_return"] = frame.label
    frame[target] = frame.label * 100 if unit == "percent" else frame.label
    frame.to_parquet(config["dataset"], index=False)
    config["evaluation"] = {"primary_label": target, "raw_return_label": "raw_return"}
    config["input_spec"]["column_roles"].update(
        {target: "label", "raw_return": "label"}
    )
    config["input_spec"]["label_availability_columns"] = {
        target: "label_available_at",
        "raw_return": "label_available_at",
    }
    config["input_spec"]["column_units"] = {} if unit is None else {target: unit}
    config["input_spec"]["column_units"]["raw_return"] = "decimal_return"
    config["analyses"].append("prediction_hac")
    config["statistics"] = {"hac_lag": 0}
    artifact = read_statistics_artifact(model_statistics(config))
    with artifact.table("model_prediction_metrics.parquet") as metrics:
        assert metrics.mse.isna().all()
        assert metrics.oos_r2.isna().all()
        assert metrics.mse_reason.eq(reason).all()
        assert metrics.oos_r2_reason.eq(reason).all()
        assert metrics.baseline_mse.notna().all()
        assert metrics.target_count.eq(4).all()
        assert metrics.head_mean_raw_return.notna().all()
    with artifact.table("model_losses.parquet") as losses:
        assert losses.status.eq("DEFINED").all()
        assert losses.loss.to_numpy() == pytest.approx(-1.0)
    with artifact.table("prediction_diagnostics.parquet") as diagnostics:
        assert diagnostics["mean"].to_numpy() == pytest.approx(1.0)


def test_ic_only_cross_target_avoids_all_loss_calculation(tmp_path, monkeypatch):
    frame, _, _, _, config = _case(tmp_path)
    frame["other_label"] = 100 * frame.label
    frame.to_parquet(config["dataset"], index=False)
    config["input_spec"]["column_roles"]["other_label"] = "label"
    config["input_spec"]["column_units"] = {"other_label": "percent"}
    config["input_spec"]["label_availability_columns"] = {
        "other_label": "label_available_at"
    }
    config["evaluation"] = {"primary_label": "other_label"}
    config["analyses"] = ["prediction_hac"]
    config["statistics"] = {"hac_lag": 0}

    def forbidden(*args, **kwargs):
        pytest.fail("IC-only statistics must not calculate unused prediction losses")

    monkeypatch.setattr(
        "quant_research_platform.research.statistics_production._prediction_quality",
        forbidden,
    )
    artifact = read_statistics_artifact(model_statistics(config))
    with artifact.table("prediction_diagnostics.parquet") as diagnostics:
        assert diagnostics["mean"].to_numpy() == pytest.approx(1.0)


def test_permutation_and_selection_weight_dates_equally(tmp_path):
    train = _frame().loc[lambda frame: frame.timestamp.eq("2020-01-01")].copy()
    train["label"] = 0.0
    case = fit_frame(
        tmp_path / "fit",
        train,
        ModelSpec(
            model_id="zero",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            target_column="label",
        ),
        market_id="a_share",
    )
    future = pd.DataFrame(
        [
            dict(
                timestamp=pd.Timestamp(date),
                asset_id=str(asset),
                x=float(asset),
                label=label,
                eligible=True,
                label_available_at=pd.Timestamp(date),
            )
            for date, size, label in [("2020-01-02", 2, 1.0), ("2020-01-03", 4, 3.0)]
            for asset in range(size)
        ]
    )
    source = tmp_path / "future.parquet"
    future.to_parquet(source, index=False)
    window = {"start": "2020-01-02", "end": "2020-01-03"}
    output = explain(
        dict(
            model=str(case.root),
            inputs=str(source),
            input_spec=case.config["input_spec"],
            output=str(tmp_path / "explain"),
            methods=["permutation"],
            window=window,
            sample_role="validation",
            selection={"max_rows": 100, "max_dates": 20},
            output_space="model_output",
            permutation={
                "metrics": ["mse"],
                "cutoff": "2020-01-04",
                "axis": "within_timestamp",
                "groups": {"x": ["x"]},
                "repeats": 2,
            },
        )
    )
    from quant_research_platform.research.models.trial_selection import (
        evaluate_model_trial_metrics_from_chunks,
    )

    selected = evaluate_model_trial_metrics_from_chunks(
        [future.assign(prediction=0.0)],
        model_id="zero",
        label_column="label",
        prediction_column="prediction",
    )
    repeats = pd.read_parquet(output / "permutation_repeats.parquet")
    assert repeats.baseline.eq(5.0).all()
    assert repeats.perturbed.eq(5.0).all()
    assert repeats.delta.eq(0.0).all()
    assert repeats.period_count.eq(2).all()
    assert repeats.observation_count.eq(6).all()
    assert selected["validation_mse"] == 5.0
    _, prediction, _ = predict_frame(
        tmp_path / "prediction", future, case, window=window
    )
    statistics = model_statistics(
        dict(
            dataset=str(source),
            input_spec=case.config["input_spec"],
            output=str(tmp_path / "statistics"),
            window=window,
            label_cutoff="2020-01-04",
            family_ids=["zero"],
            analyses=["model_losses"],
            evaluation={"primary_label": "label"},
            statistics={},
            periods=[
                {"period_id": "p", "window": window, "predictions": [str(prediction)]}
            ],
        )
    )
    with read_statistics_artifact(statistics).table(
        "model_prediction_metrics.parquet"
    ) as daily:
        assert daily.target_count.tolist() == [2, 4]
        assert daily.mse.tolist() == [1.0, 9.0]
        assert daily.mse.mean() == selected["validation_mse"]


@pytest.mark.parametrize("magnitude", [1e155, 1e-200, 1e308])
def test_exact_extreme_losses_keep_defined_r2(magnitude):
    labels = pd.Series([-magnitude, 0.0, magnitude])
    result = _prediction_quality(-labels, labels, labels, head_fraction=1.0)
    assert np.isnan(result["mse"])
    assert result["mse_reason"] == "nonfinite_result"
    assert result["oos_r2"] == -3.0
    assert result["oos_r2_reason"] is None


def test_exact_large_prediction_cancellation_preserves_perfect_loss():
    labels = pd.Series([-1e308, 1e308])
    result = _prediction_quality(labels, labels, labels, head_fraction=1.0)
    assert result["mse"] == 0.0
    assert result["mse_reason"] is None
    assert np.isnan(result["baseline_mse"])
    assert result["oos_r2"] == 1.0


def test_representable_tiny_loss_is_not_rounded_to_zero():
    labels = pd.Series([0.0] * 100)
    scores = pd.Series([2e-161] + [0.0] * 99)
    result = _prediction_quality(scores, labels, labels, head_fraction=1.0)
    assert result["mse"] == float(Fraction(2e-161) ** 2 / 100)
    assert result["mse"] > 0.0
    assert result["oos_r2_reason"] == "zero_baseline_error"


@pytest.mark.parametrize("method", ["BY", "BH"])
def test_stratified_fdr_keeps_undefined_declared_hypotheses(tmp_path, method):
    dates = pd.date_range("2020-01-01", periods=8)
    permutations = [[1.0, 2.0, 3.0]] * 4 + [[1.0, 3.0, 2.0]] * 2 + [[2.0, 3.0, 1.0]] * 2
    frame = pd.DataFrame(
        [
            dict(
                timestamp=date,
                asset_id=str(asset),
                x=xs[asset],
                constant=1.0,
                label=float(asset + 1),
                label_available_at=date,
                eligible=True,
            )
            for date, xs in zip(dates, permutations, strict=True)
            for asset in range(3)
        ]
    )
    source = tmp_path / "data.parquet"
    frame.to_parquet(source, index=False)
    spec = dict(
        data_id="synthetic",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=["timestamp", "asset_id"],
        partition_columns=[],
        column_roles={
            "timestamp": "timestamp",
            "asset_id": "key",
            "x": "factor",
            "constant": "factor",
            "label": "label",
            "label_available_at": "context",
            "eligible": "mask",
        },
        column_units={"label": "decimal_return"},
    )
    evaluation = evaluate(
        dict(
            inputs=str(source),
            input_spec=spec,
            output=str(tmp_path / "eval"),
            window={"start": "2020-01-01", "end": "2020-01-08"},
            cutoff="2020-01-09",
            evaluation=dict(
                evaluation_id="full-family",
                market_id="a_share",
                research_dataset_id="synthetic",
                signals=[
                    dict(
                        signal_id=column,
                        source_kind="factor",
                        source_column=column,
                        direction="higher_is_better",
                        transform="identity",
                    )
                    for column in ("x", "constant")
                ],
                label_columns=["label"],
            ),
        )
    )
    output = factor_statistics(
        dict(
            evaluation_inputs=[str(evaluation)],
            output=str(tmp_path / "statistics"),
            analyses=["stratified_ic"],
            statistics={"hac_lag": 0, "fdr_method": method, "fdr_alpha": 0.05},
        )
    )
    with read_statistics_artifact(output).table(
        "factor_stratified_diagnostics.parquet"
    ) as diagnostics:
        defined = diagnostics.loc[diagnostics.status.eq("DEFINED")]
        undefined = diagnostics.loc[diagnostics.status.eq("UNDEFINED")]
        assert len(diagnostics) == 4
        assert diagnostics.defined_hypothesis_count.eq(2).all()
        assert undefined.p_value.isna().all()
        assert undefined.fdr_adjusted_p_value.isna().all()
        expected = (
            2
            * (sum(1 / index for index in range(1, 5)) if method == "BY" else 1)
            * defined.p_value.iloc[0]
        )
        assert defined.fdr_adjusted_p_value.to_numpy() == pytest.approx(expected)
        if method == "BY":
            assert expected == pytest.approx(0.08717223057414171)
            assert expected > 0.05


def test_temporal_permutation_retains_one_time_series_period(tmp_path):
    def frame(start, labels):
        dates = pd.date_range(start, periods=len(labels), freq="s")
        return pd.DataFrame(
            dict(
                timestamp=dates,
                asset_id="asset",
                x=np.arange(len(labels), dtype=float),
                label=labels,
                eligible=True,
                label_available_at=dates,
                episode="one",
            )
        )

    case = fit_frame(
        tmp_path / "fit",
        frame("2020-01-01", [0.0] * 6),
        ModelSpec(
            model_id="zero",
            model_algorithm="linear_regression",
            input_factor_columns=("x",),
            target_column="label",
        ),
    )
    future = frame("2020-01-02", [1.0, 1.0, 3.0, 3.0, 3.0, 3.0])
    source = tmp_path / "future.parquet"
    future.to_parquet(source, index=False)
    output = explain(
        dict(
            model=str(case.root),
            inputs=str(source),
            input_spec=case.config["input_spec"],
            output=str(tmp_path / "explain"),
            methods=["permutation"],
            window={"start": "2020-01-02", "end": "2020-01-02"},
            sample_role="validation",
            selection={"max_rows": 100, "max_dates": None},
            output_space="model_output",
            permutation={
                "metrics": ["mse"],
                "cutoff": "2020-01-03",
                "axis": "temporal_blocks",
                "episode_column": "episode",
                "step_seconds": 1,
                "block_size": 2,
                "groups": {"x": ["x"]},
                "repeats": 1,
            },
        )
    )
    repeats = pd.read_parquet(output / "permutation_repeats.parquet")
    assert repeats.baseline.iloc[0] == pytest.approx(38 / 6)
    assert repeats.period_count.iloc[0] == 1
    assert repeats.observation_count.iloc[0] == 6


def _defined_target_case(root, *, transform="raw", clip="none", semantics=None):
    """Consume a native dataset's actual target recipe through fit and prediction."""
    from quant_research_platform.contracts.datasets import ResearchDatasetSpec
    from quant_research_platform.contracts.models import model_spec_payload
    from quant_research_platform.research.models.persisted import load_model
    from quant_research_platform.research.models.production import fit
    from quant_research_platform.storage.datasets import write_dataset

    frame = _frame().rename(
        columns={
            "label": "model_target",
            "label_available_at": "model_target_available_at",
        }
    )
    frame["forward_return_20d"] = frame.model_target
    frame["forward_return_20d_available_at"] = frame.model_target_available_at
    roles = {column: "context" for column in frame}
    roles.update(timestamp="timestamp", asset_id="key", x="factor", eligible="mask")
    roles.update(model_target="label", forward_return_20d="label")
    spec = ResearchDatasetSpec(
        data_id="training_data",
        market_id="a_share",
        frequency="1d",
        grain="asset_timestamp",
        primary_key=("timestamp", "asset_id"),
        partition_columns=("year",),
        column_roles=roles,
        input_data_ids=("input",),
        base_input_data_id="input",
        factor_columns=("x",),
        label_columns=("model_target", "forward_return_20d"),
        filter_columns=("eligible",),
        target_transform=transform,
        target_clip=clip,
        column_units={
            "model_target": "decimal_return"
            if transform in {"raw", "industry_size_residual"}
            else "dimensionless",
            "forward_return_20d": "decimal_return",
        },
        label_availability_columns={
            "model_target": "model_target_available_at",
            "forward_return_20d": "forward_return_20d_available_at",
        },
    )
    source = root / "training"
    metadata = {
        "producer_diagnostics": {
            "label_semantics": {
                "forward_return_20d": semantics
                or {"price_basis": "adjusted_open", "exit_offset_sessions": 21}
            }
        }
    }
    write_dataset(source, frame, spec=spec, metadata=metadata)
    model_spec = ModelSpec(
        model_id="model",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="model_target",
    )
    model = fit(
        {
            "inputs": str(source),
            "output": str(root / "model"),
            "model": model_spec_payload(model_spec),
            "market_id": "a_share",
            "fit_windows": [{"start": "2020-01-01", "end": "2020-01-02"}],
            "fit_cutoff": "2020-01-02",
        }
    )
    _, facts = load_model(model)
    return frame, spec, metadata, model, facts, source


@pytest.mark.parametrize(
    ("trained_transform", "evaluated_transform", "clip", "semantics", "reason"),
    [
        ("raw", "raw", "none", None, None),
        (
            "raw",
            "industry_size_residual",
            "none",
            None,
            "prediction_target_definition_mismatch",
        ),
        ("raw", "raw", "mad", None, "prediction_target_definition_mismatch"),
        (
            "date_zscore",
            "percentile",
            "none",
            None,
            "prediction_target_definition_mismatch",
        ),
        (
            "raw",
            "raw",
            "none",
            {"price_basis": "adjusted_open", "exit_offset_sessions": 22},
            "prediction_target_definition_mismatch",
        ),
        ("raw", "raw", "none", {}, "prediction_target_definition_missing"),
    ],
)
def test_frozen_target_definition_controls_only_comparable_losses(
    tmp_path, trained_transform, evaluated_transform, clip, semantics, reason
):
    from dataclasses import replace

    from quant_research_platform.research.predictions.runner import predict
    from quant_research_platform.storage.datasets import write_dataset

    frame, spec, metadata, model, facts, original = _defined_target_case(
        tmp_path, transform=trained_transform
    )
    frozen = facts["target_definition"]
    assert frozen["target_transform"] == trained_transform
    assert frozen["target_clip"] == "none"
    assert frozen["source_label"] == "forward_return_20d"
    # Later source edits must not be used to invent historical training facts.
    original_spec = json.loads((original / "spec.json").read_text())
    original_spec["target_transform"] = "industry_size_residual"
    (original / "spec.json").write_text(json.dumps(original_spec))
    spec = replace(
        spec,
        data_id="different_storage_identity",
        target_transform=evaluated_transform,
        target_clip=clip,
        minimum_listing_sessions=1,
        exclude_st=False,
    )
    if semantics is not None:
        metadata = {
            "producer_diagnostics": {
                "label_semantics": {"forward_return_20d": semantics}
                if semantics
                else {}
            }
        }
    source = tmp_path / "evaluation"
    write_dataset(source, frame, spec=spec, metadata=metadata)
    window = {"start": "2020-01-03", "end": "2020-01-05"}
    prediction = predict(
        {
            "model": str(model),
            "inputs": str(source),
            "output": str(tmp_path / "prediction"),
            "prediction_window": window,
            "sample_role": "validation",
        }
    )
    _, prediction_facts = read_predictions(prediction)
    assert prediction_facts["model_facts"]["target_definition"] == frozen
    artifact = read_statistics_artifact(
        model_statistics(
            {
                "dataset": str(source),
                "output": str(tmp_path / "statistics"),
                "window": window,
                "label_cutoff": "2020-01-06",
                "family_ids": ["model"],
                "analyses": ["model_losses", "prediction_hac"],
                "evaluation": {
                    "primary_label": "model_target",
                    "raw_return_label": "forward_return_20d",
                },
                "statistics": {"hac_lag": 0},
                "periods": [
                    {
                        "period_id": "p",
                        "window": window,
                        "predictions": [str(prediction)],
                    }
                ],
            }
        )
    )
    with artifact.table("model_prediction_metrics.parquet") as metrics:
        assert metrics.target_count.eq(4).all()
        assert metrics.baseline_mse.notna().all()
        assert metrics.head_mean_raw_return.notna().all()
        if reason is None:
            assert metrics.mse.to_numpy() == pytest.approx(0.0, abs=1e-30)
            assert metrics.oos_r2.eq(1.0).all()
        else:
            assert metrics.mse.isna().all() and metrics.oos_r2.isna().all()
            assert metrics.mse_reason.eq(reason).all()
            assert metrics.oos_r2_reason.eq(reason).all()
    with artifact.table("prediction_diagnostics.parquet") as diagnostics:
        assert diagnostics["mean"].to_numpy() == pytest.approx(1.0)
    with artifact.table("model_losses.parquet") as losses:
        assert losses.status.eq("DEFINED").all()


@pytest.mark.parametrize(
    "difference", ["definition", "unit", "unknown_training", "unknown_evaluation"]
)
def test_selection_and_explanation_share_loss_comparability(tmp_path, difference):
    from dataclasses import replace

    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.research.predictions.runner import predict
    from quant_research_platform.storage.datasets import write_dataset

    frame, spec, metadata, model, _, _ = _defined_target_case(tmp_path)
    reason = "prediction_target_definition_mismatch"
    if difference == "definition":
        spec = replace(spec, target_transform="industry_size_residual")
    elif difference == "unit":
        spec = replace(
            spec, column_units={**spec.column_units, "model_target": "percent"}
        )
        reason = "prediction_target_unit_mismatch"
    elif difference == "unknown_training":
        saved = json.loads((model / "model.json").read_text())
        saved["facts"].pop("target_definition")
        (model / "model.json").write_text(json.dumps(saved))
        reason = "prediction_target_definition_missing"
    else:
        spec = DataSpec(
            **{
                key: getattr(spec, key)
                for key in (
                    "data_id",
                    "frequency",
                    "grain",
                    "primary_key",
                    "partition_columns",
                    "column_roles",
                    "column_units",
                    "label_availability_columns",
                )
            }
        )
        reason = "prediction_target_definition_missing"
    source = tmp_path / "evaluation"
    write_dataset(source, frame, spec=spec, metadata=metadata)
    window = {"start": "2020-01-03", "end": "2020-01-05"}
    prediction = predict(
        {
            "model": str(model),
            "inputs": str(source),
            "output": str(tmp_path / "prediction"),
            "prediction_window": window,
            "sample_role": "validation",
        }
    )
    scored = json.loads(
        score(
            {
                "predictions": [str(prediction)],
                "candidate_ids": ["model"],
                "labels": str(source),
                "validation_windows": [window],
                "scoring_window": window,
                "cutoff": "2020-01-06",
                "selection_group": "global",
                "validation_axis": "cross_sectional",
                "output": str(tmp_path / "score.json"),
            }
        ).read_text()
    )
    assert scored["metrics"][0]["validation_mse"] is None
    assert scored["metrics"][0]["validation_ic"] == pytest.approx(1.0)
    assert scored["metrics"][0]["validation_rows"] == 12
    assert scored["candidates"][0]["mse_reason"] == reason
    from copy import deepcopy

    from quant_research_platform.research.models.trial_selection import (
        _selection_choice,
    )

    decision = {
        **scored,
        "tie_break_order": ["model"],
        "selection_metric": "validation_ic",
        "selection_direction": "maximize",
    }
    assert _selection_choice(decision) == "model"
    for mutation in ("unknown_reason", "numeric_mse"):
        broken = deepcopy(decision)
        if mutation == "unknown_reason":
            broken["candidates"][0]["mse_reason"] = "invented"
        else:
            broken["metrics"][0]["validation_mse"] = 0.0
        with pytest.raises(ValueError, match="selection_mse_reason_invalid"):
            _selection_choice(broken)
    output = explain(
        {
            "model": str(model),
            "inputs": str(source),
            "output": str(tmp_path / "explanation"),
            "methods": ["permutation"],
            "window": window,
            "sample_role": "validation",
            "selection": {"max_rows": 100, "max_dates": 20},
            "output_space": "model_output",
            "permutation": {
                "metrics": ["mse", "ic"],
                "cutoff": "2020-01-06",
                "axis": "within_timestamp",
                "groups": {"x": ["x"]},
                "repeats": 2,
            },
        }
    )
    repeats = pd.read_parquet(output / "permutation_repeats.parquet")
    mse = repeats.loc[repeats.metric.eq("mse")]
    assert mse.status.eq("UNDEFINED").all() and mse.reason.eq(reason).all()
    assert mse[["baseline", "perturbed", "delta"]].isna().all().all()
    assert repeats.loc[repeats.metric.eq("ic"), "status"].eq("DEFINED").all()
    artifact = read_statistics_artifact(
        model_statistics(
            {
                "dataset": str(source),
                "output": str(tmp_path / "statistics"),
                "window": window,
                "label_cutoff": "2020-01-06",
                "family_ids": ["model"],
                "analyses": ["model_losses"],
                "evaluation": {
                    "primary_label": "model_target",
                    "raw_return_label": "forward_return_20d",
                },
                "statistics": {},
                "periods": [
                    {
                        "period_id": "p",
                        "window": window,
                        "predictions": [str(prediction)],
                    }
                ],
            }
        )
    )
    with artifact.table("model_prediction_metrics.parquet") as metrics:
        assert metrics.mse_reason.eq(reason).all()
        assert metrics.target_count.eq(4).all()


@pytest.mark.parametrize(
    "mutation", ["source", "transform", "clip", "semantics", "extra"]
)
def test_model_and_prediction_validate_frozen_target_definition(tmp_path, mutation):
    from copy import deepcopy

    from quant_research_platform.research.models.persisted import load_model

    _, case, prediction, _, _ = _case(tmp_path)
    definition = deepcopy(case.facts["target_definition"])
    if mutation == "source":
        definition["source_label"] = "unrelated_label"
    elif mutation == "transform":
        definition["target_transform"] = []
    elif mutation == "clip":
        definition["target_clip"] = "mad"
    elif mutation == "semantics":
        definition["label_semantics"] = []
    else:
        definition["spec_path"] = "not_a_scientific_definition"
    manifest = json.loads((prediction / "manifest.json").read_text())
    manifest["metadata"]["model_facts"]["target_definition"] = definition
    (prediction / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="model_target_definition_invalid"):
        read_predictions(prediction)
    saved = json.loads((case.root / "model.json").read_text())
    saved["facts"]["target_definition"] = definition
    (case.root / "model.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="model_target_definition_invalid"):
        load_model(case.root)


def test_raw_unclipped_model_target_can_compare_with_its_identical_source_label(
    tmp_path,
):
    from quant_research_platform.research.predictions.runner import predict

    _, _, _, model, _, source = _defined_target_case(tmp_path)
    window = {"start": "2020-01-03", "end": "2020-01-05"}
    prediction = predict(
        {
            "model": str(model),
            "inputs": str(source),
            "output": str(tmp_path / "prediction"),
            "prediction_window": window,
            "sample_role": "validation",
        }
    )
    artifact = read_statistics_artifact(
        model_statistics(
            {
                "dataset": str(source),
                "output": str(tmp_path / "statistics"),
                "window": window,
                "label_cutoff": "2020-01-06",
                "family_ids": ["model"],
                "analyses": ["model_losses"],
                "evaluation": {"primary_label": "forward_return_20d"},
                "statistics": {},
                "periods": [
                    {
                        "period_id": "p",
                        "window": window,
                        "predictions": [str(prediction)],
                    }
                ],
            }
        )
    )
    with artifact.table("model_prediction_metrics.parquet") as metrics:
        assert metrics.mse.to_numpy() == pytest.approx(0.0, abs=1e-30)
        assert metrics.oos_r2.eq(1.0).all()
        assert metrics.mse_reason.eq("").all()


@pytest.mark.parametrize(
    ("declared_unit", "explicit_unit", "has_definition", "reason", "error"),
    [
        (None, "decimal_return", True, None, None),
        (None, "percent", True, "prediction_target_unit_mismatch", None),
        ("percent", "decimal_return", True, None, "scoring_label_unit_mismatch"),
        (None, None, True, None, "label_unit_required"),
        (None, "decimal_return", False, "prediction_target_definition_missing", None),
        ("percent", None, True, "prediction_target_unit_mismatch", None),
    ],
)
def test_scoring_honors_explicit_units_without_weakening_target_definition(
    tmp_path, declared_unit, explicit_unit, has_definition, reason, error
):
    from quant_research_platform.artifacts.outputs.prediction import (
        write_prediction_outputs,
    )
    from quant_research_platform.contracts.data import DataSpec
    from quant_research_platform.storage.datasets import write_dataset

    labels, _, prediction, metadata, config = _case(tmp_path)
    config["input_spec"]["column_units"] = (
        {} if declared_unit is None else {"label": declared_unit}
    )
    source = tmp_path / "native-labels"
    write_dataset(
        source,
        labels,
        spec=DataSpec(
            data_id="score_labels",
            frequency="1d",
            grain="asset_timestamp",
            primary_key=("timestamp", "asset_id"),
            partition_columns=("year",),
            column_roles=config["input_spec"]["column_roles"],
            column_units=config["input_spec"]["column_units"],
        ),
    )
    if not has_definition:
        frame, _ = read_predictions(prediction)
        metadata["model_facts"].pop("target_definition")
        prediction = write_prediction_outputs(
            output_path=tmp_path / "legacy-prediction",
            output_chunks=[frame],
            metadata=metadata,
        )
    request = {
        "predictions": [str(prediction)],
        "candidate_ids": ["model"],
        "labels": str(source),
        "validation_windows": [config["window"]],
        "scoring_window": config["window"],
        "cutoff": "2020-01-06",
        "selection_group": "global",
        "validation_axis": "cross_sectional",
        "output": str(tmp_path / "score.json"),
    }
    if explicit_unit is not None:
        request["label_unit"] = explicit_unit
    if error is not None:
        with pytest.raises(ValueError, match=error):
            score(request)
        return
    result = json.loads(score(request).read_text())
    assert result["label_unit"] == (explicit_unit or declared_unit)
    assert result["metrics"][0]["validation_rows"] == 12
    assert result["metrics"][0]["validation_ic"] == pytest.approx(1.0)
    assert result["candidates"][0]["mse_reason"] == reason
    if reason is None:
        assert result["metrics"][0]["validation_mse"] == pytest.approx(0.0, abs=1e-30)
    else:
        assert result["metrics"][0]["validation_mse"] is None
