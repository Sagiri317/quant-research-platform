"""Real consumer regressions for scoring domains, declared features and X units."""

import json

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.evaluation import read_evaluation_outputs
from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.contracts.evaluations import EvaluationSpec
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.contracts.signals import SignalSpec
from quant_research_platform.research.evaluations.engine import (
    build_evaluation_result_from_frames,
)
from quant_research_platform.research.evaluations.runner import evaluate
from quant_research_platform.research.features.transforms import (
    apply_feature_transforms,
)
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.trial_selection import (
    evaluate_model_trial_metrics_from_chunks,
    score,
)
from quant_research_platform.research.predictions.runner import predict
from tests.research.crypto_evaluation_test_support import evaluation_config
from tests.research.direct_model_test_support import fit_frame


def _training():
    dates = pd.date_range("2024-01-01", periods=3)
    return pd.DataFrame(
        dict(
            timestamp=dates,
            asset_id="btc",
            x=[1.0, 2.0, 3.0],
            label=[1.0, 2.0, 3.0],
            eligible=True,
            label_available_at=dates,
        )
    )


def _linear(model_id="linear", column="x", feature=False):
    return ModelSpec(
        model_id=model_id,
        model_algorithm="linear_regression",
        input_feature_columns=(column,) if feature else (),
        input_factor_columns=() if feature else (column,),
        target_column="label",
        random_seed=1,
    )


@pytest.mark.parametrize("scope", ["mixed", "transfer", "mature_only", "common"])
def test_single_asset_score_checks_final_domain_without_restricting_predictions(
    tmp_path, scope
):
    case = fit_frame(tmp_path / "fit", _training(), _linear())
    frame = pd.DataFrame(
        [
            (date, asset, float(4 + j + offset), float(6 - j + offset), True, date)
            for j, date in enumerate(pd.date_range("2024-01-04", periods=3))
            for asset, offset in [("btc", 0), ("eth", 100)]
        ],
        columns=[
            "timestamp",
            "asset_id",
            "x",
            "label",
            "eligible",
            "label_available_at",
        ],
    )
    if scope == "transfer":
        frame = frame.loc[frame.asset_id.eq("eth")]
    if scope == "mature_only":
        frame.loc[frame.asset_id.eq("btc"), "label_available_at"] = pd.Timestamp(
            "2024-02-01"
        )
    source = tmp_path / "future.parquet"
    frame.to_parquet(source, index=False)
    window = {"start": "2024-01-04", "end": "2024-01-06"}
    request = dict(
        model=str(case.root),
        inputs=str(source),
        input_spec=case.config["input_spec"],
        output=str(tmp_path / "prediction"),
        prediction_window=window,
    )
    prediction = predict(request)
    assert len(read_predictions(prediction)[0]) == len(frame)
    predictions, ids = [str(prediction)], ["linear"]
    if scope == "common":
        second = fit_frame(tmp_path / "fit_second", _training(), _linear("second"))
        predictions.append(
            str(
                predict(
                    {
                        **request,
                        "model": str(second.root),
                        "output": str(tmp_path / "second_prediction"),
                        "asset_ids": ["eth"],
                    }
                )
            )
        )
        ids.append("second")
    scoring = dict(
        predictions=predictions,
        candidate_ids=ids,
        labels=str(source),
        label_spec=case.config["input_spec"],
        validation_windows=[window],
        scoring_window=window,
        cutoff="2024-01-06",
        selection_group="global",
        validation_axis="single_asset_time_series",
        label_unit="decimal_return",
        output=str(tmp_path / "scores.json"),
    )
    if scope == "mixed":
        with pytest.raises(
            ValueError, match="single_asset_time_series_evaluation_requires_one_asset"
        ):
            score(scoring)
        assert not (tmp_path / "scores.json").exists()
    else:
        result = json.loads(score(scoring).read_text())
        assert result["common_rows"] == 3
        assert all(
            row["validation_ic"] == pytest.approx(-1) for row in result["metrics"]
        )


@pytest.mark.parametrize("split", [False, True])
def test_single_asset_metric_rejects_second_asset_within_or_across_chunks(split):
    frame = _training().assign(
        prediction=[1.0, 2.0, 3.0], asset_id=["btc", "btc", "eth"]
    )
    chunks = [frame.iloc[:2], frame.iloc[2:]] if split else [frame]
    with pytest.raises(
        ValueError, match="single_asset_time_series_evaluation_requires_one_asset"
    ):
        evaluate_model_trial_metrics_from_chunks(
            chunks,
            model_id="m",
            label_column="label",
            prediction_column="prediction",
            validation_axis="single_asset_time_series",
        )


def _evaluation(column):
    return EvaluationSpec(
        evaluation_id="features",
        market_id="crypto",
        research_dataset_id="synthetic",
        signals=(
            SignalSpec(signal_id="signal", source_kind="feature", source_column=column),
        ),
        label_columns=("label",),
        n_groups=2,
    )


@pytest.mark.parametrize(
    "column", ["x", "feature_x", "feature_missing_x", "feature_observed_x"]
)
def test_external_feature_names_do_not_invent_companions(tmp_path, column):
    frame = pd.concat(
        [
            _training().assign(asset_id=asset, x=[1.0, 2.0, 3.0])
            for asset in ("a", "b", "c")
        ]
    )
    frame["x"] = np.tile([1.0, 2.0, 3.0], 3) + np.repeat([0.0, 1.0, 2.0], 3)
    frame["label"] = 2 * frame["x"]
    frame = frame.rename(columns={"x": column}).sort_values(["timestamp", "asset_id"])
    case = fit_frame(tmp_path, frame, _linear(column=column, feature=True))
    config = evaluation_config(
        case.config["inputs"],
        tmp_path / "evaluation",
        _evaluation(column),
        start="2024-01-01",
        end="2024-01-03",
    )
    config["input_spec"] = case.config["input_spec"]
    metrics = read_evaluation_outputs(evaluate(config))[1]["metrics"]
    assert metrics.n_obs.tolist() == [3, 3, 3]
    assert metrics.ic.tolist() == pytest.approx([1.0, 1.0, 1.0])
    future = frame.assign(timestamp=frame.timestamp + pd.Timedelta(days=3))
    path = tmp_path / "future.parquet"
    future.to_parquet(path, index=False)
    prediction = predict(
        dict(
            model=str(case.root),
            inputs=str(path),
            input_spec=case.config["input_spec"],
            output=str(tmp_path / "predictions"),
            prediction_window={"start": "2024-01-04", "end": "2024-01-06"},
        )
    )
    assert len(read_predictions(prediction)[0]) == 9


def _canonical():
    frame = pd.concat(
        [
            _training().assign(
                timestamp=pd.Timestamp("2024-01-01"),
                asset_id=list("abc"),
                x=[1.0, np.nan, 3.0],
            ),
            _training().assign(
                timestamp=pd.Timestamp("2024-01-02"), asset_id=list("abc"), x=np.nan
            ),
        ],
        ignore_index=True,
    )
    frame["label_available_at"] = frame.timestamp
    recipe = FeatureTransformSpec(
        "a_share_canonical",
        ("x",),
        clip="none",
        industry_neutralize=False,
        size_neutralize=False,
    )
    return apply_feature_transforms(frame, (recipe,))


@pytest.mark.parametrize("batch_rows", [1, 1000])
@pytest.mark.parametrize("indicator", ["feature_missing_x", "feature_observed_x"])
def test_declared_feature_companions_close_projection_and_keep_producer_domains(
    tmp_path, monkeypatch, batch_rows, indicator
):
    from quant_research_platform.research.evaluations import runner

    original = runner.iter_dataset_chunks
    monkeypatch.setattr(
        runner,
        "iter_dataset_chunks",
        lambda session: original(session, batch_rows=batch_rows),
    )
    frame = _canonical()
    case = fit_frame(tmp_path, frame, _linear(column=indicator, feature=True))
    assert case.result.metadata["rows_after_input_label_dropna"] == 2
    spec = _evaluation("feature_x")
    declared = case.config["input_spec"]
    declared["column_roles"]["feature_x"] = "feature"
    config = evaluation_config(
        case.config["inputs"],
        tmp_path / "evaluation",
        spec,
        start="2024-01-01",
        end="2024-01-02",
    )
    config["inputs"] = {
        "path": config["inputs"],
        "spec": declared,
        "columns": ["timestamp", "asset_id", "feature_x", "label"],
    }
    metrics = read_evaluation_outputs(evaluate(config))[1]["metrics"]
    assert metrics.n_obs.tolist() == [2, 0]
    with build_evaluation_result_from_frames(
        [frame.iloc[:2], frame.iloc[2:]],
        spec=spec,
        asset_column="asset_id",
        declared_columns=declared["column_roles"],
    ) as direct:
        direct_metrics = pd.concat(
            [chunk[0] for chunk in direct.output_chunks], ignore_index=True
        )
    pd.testing.assert_frame_equal(metrics, direct_metrics)
    future = frame.assign(timestamp=frame.timestamp + pd.Timedelta(days=3))
    path = tmp_path / "future.parquet"
    future.to_parquet(path, index=False)
    prediction = predict(
        dict(
            model=str(case.root),
            inputs={
                "path": str(path),
                "spec": declared,
                "columns": ["timestamp", "asset_id", indicator],
            },
            output=str(tmp_path / "predictions"),
            prediction_window={"start": "2024-01-04", "end": "2024-01-05"},
        )
    )
    assert len(read_predictions(prediction)[0]) == 2


@pytest.mark.parametrize("invalid", ["missing", "nonbinary", "null"])
def test_declared_observation_companion_cannot_disappear_or_be_invalid(
    tmp_path, invalid
):
    frame = _canonical()
    case = fit_frame(tmp_path, frame, _linear(column="feature_x", feature=True))
    if invalid == "missing":
        frame = frame.drop(columns="feature_observed_x")
    else:
        frame.loc[0, "feature_observed_x"] = 2 if invalid == "nonbinary" else np.nan
    frame.to_parquet(case.config["inputs"], index=False)
    config = evaluation_config(
        case.config["inputs"],
        tmp_path / "evaluation",
        _evaluation("feature_x"),
        start="2024-01-01",
        end="2024-01-02",
    )
    config["input_spec"] = case.config["input_spec"]
    with pytest.raises(
        ValueError, match="dataset_column_not_found|invalid_feature_observation_flag"
    ):
        evaluate(config)
    assert not (tmp_path / "evaluation").exists()


@pytest.mark.parametrize(
    "training_unit,prediction_unit",
    [
        ("meter", "centimeter"),
        ("meter", "meter"),
        ("meter", None),
        (None, "centimeter"),
        (None, None),
    ],
)
def test_model_freezes_only_consumed_declared_units_before_predicting(
    tmp_path, monkeypatch, training_unit, prediction_unit
):
    from quant_research_platform.research.predictions import runner

    case = fit_frame(tmp_path, _training().assign(unused=2.0), _linear())
    declared = {
        **case.config["input_spec"],
        "column_units": {
            "unused": "meter",
            **({"x": training_unit} if training_unit else {}),
        },
    }
    model = fit(
        {**case.config, "input_spec": declared, "output": str(tmp_path / "unit_model")}
    )
    manifest = json.loads((model / "model.json").read_text())
    assert manifest["facts"]["input_units"] == (
        {"x": training_unit} if training_unit else {}
    )
    # Prediction must use saved scientific facts, not the original training source.
    from pathlib import Path

    Path(case.config["inputs"]).unlink()
    future = (
        _training()
        .iloc[[0]]
        .assign(timestamp=pd.Timestamp("2024-01-04"), asset_id="eth", x=100.0)
    )
    path = tmp_path / "future.parquet"
    future.to_parquet(path, index=False)
    declared = {
        **declared,
        "data_id": "different_data",
        "column_units": {
            "unused": "centimeter",
            **({"x": prediction_unit} if prediction_unit else {}),
        },
    }
    request = dict(
        model=str(model),
        inputs=str(path),
        input_spec=declared,
        output=str(tmp_path / "prediction"),
        prediction_window={"start": "2024-01-04", "end": "2024-01-04"},
    )
    if training_unit and prediction_unit and training_unit != prediction_unit:

        def forbidden(*args):
            pytest.fail("Unit conflict reached estimator prediction")

        monkeypatch.setattr(runner, "prediction_callable", forbidden)
        with pytest.raises(ValueError, match="model_input_unit_mismatch: x"):
            predict(request)
        assert not (tmp_path / "prediction").exists()
    else:
        assert read_predictions(predict(request))[0].prediction_score.iloc[
            0
        ] == pytest.approx(100.0)


def test_ordinary_prefixed_rank_feature_preserves_missing_neutral_domain(tmp_path):
    frame = _training().assign(
        timestamp=pd.Timestamp("2024-01-01"),
        asset_id=list("abc"),
        label_available_at=pd.Timestamp("2024-01-01"),
    )
    frame = frame.rename(columns={"x": "feature_x"})
    frame.loc[1, "feature_x"] = np.nan
    case = fit_frame(
        tmp_path,
        frame,
        ModelSpec(
            model_id="rank",
            model_algorithm="equal_weight_rank_ensemble",
            input_feature_columns=("feature_x",),
            target_column="label",
            random_seed=1,
        ),
    )
    assert case.result.metadata["rows_after_input_label_dropna"] == 2
    source = tmp_path / "future.parquet"
    frame.assign(timestamp=pd.Timestamp("2024-01-02")).to_parquet(source, index=False)
    output = predict(
        dict(
            model=str(case.root),
            inputs=str(source),
            input_spec=case.config["input_spec"],
            output=str(tmp_path / "prediction"),
            prediction_window={"start": "2024-01-02", "end": "2024-01-02"},
        )
    )
    saved = read_predictions(output)[0].set_index("asset_id")
    assert len(saved) == 2 and "b" not in saved.index


def test_canonical_observation_projection_matches_factor_statistics(tmp_path):
    from quant_research_platform.artifacts.statistics import read_statistics_artifact
    from quant_research_platform.research.statistics_production import factor_statistics

    frame = _canonical()
    case = fit_frame(tmp_path, frame, _linear(column="feature_x", feature=True))
    declared = case.config["input_spec"]
    artifact = read_statistics_artifact(
        factor_statistics(
            dict(
                dataset={
                    "path": case.config["inputs"],
                    "spec": declared,
                    "columns": ["feature_x", "label"],
                },
                output=str(tmp_path / "statistics"),
                window={"start": "2024-01-01", "end": "2024-01-02"},
                analyses=["rank_ic"],
                evaluation={"factors": ["feature_x"], "primary_label": "label"},
                statistics={
                    "hac_lag": 0,
                    "factor_fdr_families": {"features": ["feature_x"]},
                    "fdr_primary": "BY_5pct",
                    "fdr_robustness": "BH_5pct",
                },
            )
        )
    )
    with artifact.table("factor_diagnostics.parquet") as statistics:
        assert statistics.input_date_count.eq(2).all()
        assert statistics.observation_count.eq(1).all()
    # An observation flag selected as its own signal stays a separate finite signal.
    spec = _evaluation("feature_x")
    from dataclasses import replace

    spec = replace(
        spec,
        signals=(
            *spec.signals,
            SignalSpec(
                signal_id="observed",
                source_kind="feature",
                source_column="feature_observed_x",
            ),
        ),
    )
    with build_evaluation_result_from_frames(
        [frame],
        spec=spec,
        asset_column="asset_id",
        declared_columns=declared["column_roles"],
    ) as result:
        metrics = pd.concat([chunk[0] for chunk in result.output_chunks])
    assert metrics.loc[metrics.signal.eq("signal"), "n_obs"].tolist() == [2, 0]
    assert metrics.loc[metrics.signal.eq("observed"), "n_obs"].tolist() == [3, 3]


def test_declared_outputs_resolve_prefixed_source_name_without_a_name_ban(tmp_path):
    frame = _training().assign(
        timestamp=pd.Timestamp("2024-01-01"),
        asset_id=list("abc"),
        label_available_at=pd.Timestamp("2024-01-01"),
    )
    frame = frame.rename(columns={"x": "missing_x"})
    frame.loc[1, "missing_x"] = np.nan
    recipe = FeatureTransformSpec(
        "a_share_canonical",
        ("missing_x",),
        clip="none",
        industry_neutralize=False,
        size_neutralize=False,
    )
    produced = apply_feature_transforms(frame, (recipe,))
    case = fit_frame(
        tmp_path, produced, _linear(column="feature_missing_missing_x", feature=True)
    )
    spec = _evaluation("feature_missing_x")
    declared = case.config["input_spec"]
    declared["column_roles"]["feature_missing_x"] = "feature"
    config = evaluation_config(
        case.config["inputs"],
        tmp_path / "evaluation",
        spec,
        start="2024-01-01",
        end="2024-01-01",
        cutoff="2024-01-03",
    )
    config["inputs"] = {
        "path": config["inputs"],
        "spec": declared,
        "columns": ["feature_missing_x", "label"],
    }
    metrics = read_evaluation_outputs(evaluate(config))[1]["metrics"]
    assert metrics.n_obs.tolist() == [2]
