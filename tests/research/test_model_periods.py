"""Frozen phase metrics use real fits, historical CV, and native future daily rows."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.data import DataSpec
from quant_research_platform.contracts.experiments import label_maturity_mask
from quant_research_platform.research.evaluations.time_series import (
    evaluate_time_series,
)
from quant_research_platform.research.models.fitting import prediction_callable
from quant_research_platform.research.models.periods import evaluate_model_periods
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.research.rolling import run_sessions
from quant_research_platform.storage.datasets import write_dataset
from tests.correctness.test_direct_model_operations import data, fit_config, input_spec


def _declaration():
    return {
        **input_spec(),
        "column_units": {"label": "decimal_return"},
        "label_availability_columns": {"label": "label_available_at"},
    }


def _future_daily(
    tmp_path,
    source,
    predictions,
    *,
    dates,
    assets,
    axis,
    baselines=("zero",),
    top_quantile=None,
):
    return (
        evaluate_time_series(
            {
                "inputs": source,
                "top_quantile": top_quantile,
                "predictions": predictions,
                "factor_columns": [],
                "label": "label",
                "window": {"start": dates[0], "end": dates[-1]},
                "cutoff": dates[-1] + "T23:59:59",
                "sessions": dates,
                "asset_ids": assets,
                "evaluation_axis": axis,
                "r2_baselines": list(baselines),
                "comparison_groups": {},
                "inference_groups": {},
                "statistics": {
                    "hac_lag": 1,
                    "block_size": 1,
                    "repetitions": 100,
                    "seed": 42,
                    "confidence": 0.95,
                    "fdr_method": "BY",
                    "alpha": 0.05,
                },
                "output": str(tmp_path / "future"),
            }
        )
        / "daily"
    )


def _tables(output):
    artifact = read_statistics_artifact(output)
    with artifact.table("model_period_metrics.parquet") as metrics:
        metrics = metrics.copy()
    with artifact.table("model_epoch_curves.parquet") as epochs:
        return metrics, epochs.copy(), artifact.metadata


def test_training_tails_select_predictions_before_missing_labels(tmp_path):
    from quant_research_platform.research.models.periods import _training_daily

    dates = pd.date_range("2020-01-01", periods=2)
    frame = pd.DataFrame(
        [
            (day, f"s{i:02d}", float(i), i / 100, True, day)
            for day in dates
            for i in range(11)
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
    frame.loc[frame.index[-1], "label"] = np.nan
    path = tmp_path / "input.parquet"
    frame.to_parquet(path, index=False)
    source = {"path": str(path), "spec": _declaration()}
    config = {
        **fit_config(tmp_path, path),
        "inputs": source,
        "fit_windows": [{"start": "2020-01-01", "end": "2020-01-02"}],
    }
    result, facts = load_model(fit(config))
    daily = _training_daily(
        source,
        result,
        facts,
        sessions=dates,
        evaluation_axis="cross_sectional",
        top_quantile=0.2,
    )
    second = daily.loc[daily.timestamp.eq(dates[1])].set_index("metric")
    # The tail contains s09/s10; s10's missing label must not admit s08.
    assert second.loc["long_mean_label", "value"] == pytest.approx(0.09)
    assert second.loc["long_ic", "observation_count"] == 1
    assert pd.isna(second.loc["long_ic", "value"])
    assert daily.loc[daily.metric.eq("mse"), "observation_count"].sum() == 21


def test_historical_candidates_export_only_their_saved_validation_facts():
    from quant_research_platform.research.models.periods import _validation_outputs

    selection = {
        "metric": "daily_mse",
        "validation_window": {"start": "2020-01-01", "end": "2020-01-02"},
        "validation_cutoff": "2020-01-03",
        "common_rows": 6,
        "common_sessions": 2,
        "candidates": [
            {"hyperparameters": {"alpha": 0.01}, "validation_daily_mse": 0.4},
            {"hyperparameters": {"alpha": 0.1}, "validation_daily_mse": 0.6},
        ],
        "selected_hyperparameters": {"alpha": 0.01},
    }
    metrics, daily, curves = _validation_outputs(
        selection,
        {
            "period_id": "2020Q1",
            "model_id": "lasso",
            "label_unit": "dimensionless",
        },
    )
    assert not daily and not curves
    mse = [row for row in metrics if row["metric"] == "mse"]
    assert [row["value"] for row in mse] == [0.4, 0.6]
    assert [row["selected"] for row in mse] == [True, False]
    assert [row["selection_score"] for row in mse] == [0.4, 0.6]
    assert len({row["parameters_json"] for row in mse}) == 2
    assert all(
        row["reason"] == "not_saved"
        for row in metrics
        if row["metric"] in {"mae", "ic", "rank_ic"}
    )


def test_native_phase_errors_match_hand_calculation_and_saved_cv(tmp_path, monkeypatch):
    path = data(tmp_path)
    source = {"path": str(path), "spec": _declaration()}
    results = run_sessions(
        {
            "inputs": source,
            "output": str(tmp_path / "rolling"),
            "market_id": "a_share",
            "models": [fit_config(tmp_path, path)["model"]],
            "training_start": "2020-01-01",
            "min_train_sessions": 2,
            "validation_sessions": 1,
            "prediction_sessions": 3,
            "sessions": pd.date_range("2020-01-01", periods=9)
            .strftime("%Y-%m-%d")
            .tolist(),
            "rolling_window": {"start": "2020-01-04", "end": "2020-01-09"},
            "evaluation_cutoff": "2020-01-09",
            "label_unit": "decimal_return",
            "evaluation": {
                "evaluation_id": "fold",
                "market_id": "a_share",
                "research_dataset_id": "synthetic",
                "signals": [
                    {
                        "signal_id": "score",
                        "source_kind": "prediction",
                        "source_column": "score",
                    }
                ],
                "label_columns": ["label"],
                "evaluation_axis": "cross_sectional",
                "top_quantile": 0.7,
            },
        }
    )
    future = _future_daily(
        tmp_path,
        source,
        {"linear": [str(r["predictions"]) for r in results]},
        dates=pd.date_range("2020-01-04", periods=6).strftime("%Y-%m-%d").tolist(),
        assets=["a", "b", "c"],
        axis="cross_sectional",
        top_quantile=1.0,
    )
    import quant_research_platform.research.models.fitting as fitting

    monkeypatch.setattr(
        fitting, "_fit_model_result", lambda *a, **kw: pytest.fail("statistics fitted")
    )
    output = evaluate_model_periods(
        {
            "inputs": source,
            "evaluation_axis": "cross_sectional",
            "top_quantile": 0.2,
            "sessions": pd.date_range("2020-01-01", periods=9)
            .strftime("%Y-%m-%d")
            .tolist(),
            "asset_ids": ["a", "b", "c"],
            "prediction_daily": {"linear": str(future)},
            "periods": [
                {
                    "period_id": r["fold_id"],
                    "prediction_window": {
                        "start": f"2020-01-0{4 + i * 3}",
                        "end": f"2020-01-0{6 + i * 3}",
                    },
                    "models": {"linear": str(r["model"])},
                }
                for i, r in enumerate(results)
            ],
            "output": str(tmp_path / "periods"),
        }
    )
    metrics, epochs, metadata = _tables(output)
    first = metrics.loc[metrics.period_id.eq(results[0]["fold_id"])].set_index(
        ["stage", "metric"]
    )
    assert metadata["models"][0]["top_quantile_by_stage"] == {
        "training": 0.2,
        "validation": 0.7,
        "prediction": 1.0,
    }
    # The three fractions retain 1, 2 and 3 of the three signals per day.
    for stage, count in (("training", 3), ("validation", 2), ("prediction", 9)):
        assert first.loc[(stage, "long_mean_label"), "observation_count"] == count
    # y=2*x+day: fit days1..3 has intercept2. Candidate fit days1..2 has1.5.
    for metric, expected in {
        "mse": 2 / 3,
        "mae": 2 / 3,
        "rmse": np.sqrt(2 / 3),
        "ic": 1,
        "rank_ic": 1,
        "r2_vs_zero": 58 / 59,
    }.items():
        assert first.loc[("training", metric), "value"] == pytest.approx(expected)
    assert first.loc[("training", "mse"), "observation_count"] == 9
    assert first.loc[("validation", "mse"), "value"] == pytest.approx(2.25)
    assert first.loc[("validation", "rmse"), "value"] == pytest.approx(1.5)
    assert first.loc[("validation", "mae"), "value"] == pytest.approx(1.5)
    for metric, expected in {
        "mse": 29 / 3,
        "mae": 3,
        "rmse": np.sqrt(29 / 3),
        "r2_vs_zero": 224 / 253,
    }.items():
        assert first.loc[("prediction", metric), "value"] == pytest.approx(expected)
    second = metrics.loc[
        metrics.period_id.eq(results[1]["fold_id"]) & metrics.metric.eq("mse")
    ].set_index("stage")
    assert second.loc["training", "observation_count"] == 18
    assert second.loc["training", "value"] == pytest.approx(35 / 12)
    assert second.loc["validation", "value"] == pytest.approx(9)
    assert epochs.empty and metadata["models"][0]["historical_cv_curves"] == "not_saved"
    assert (
        first.loc[("training", "mse"), "cutoff"]
        < first.loc[("prediction", "mse"), "window_start"]
    )
    with pytest.raises(ValueError, match="prediction_must_follow"):
        predict(
            {
                "model": str(results[0]["model"]),
                "inputs": source,
                "prediction_window": {"start": "2020-01-01", "end": "2020-01-03"},
                "output": str(tmp_path / "forbidden"),
            }
        )


@pytest.mark.parametrize(
    "algorithm",
    [
        "lasso_regression",
        "xgboost_regressor",
        "lstm_regressor",
        "transformer_regressor",
    ],
)
@pytest.mark.parametrize("axis", ["cross_sectional", "single_asset_time_series"])
def test_frozen_native_month_stream_and_four_regressors(
    tmp_path, monkeypatch, algorithm, axis
):
    raw = pd.read_parquet(data(tmp_path))
    dates = pd.date_range("2020-01-29", periods=9)
    raw["timestamp"] = raw.timestamp.map(
        dict(zip(pd.date_range("2020-01-01", periods=9), dates, strict=True))
    )
    declaration = _declaration()
    single = axis == "single_asset_time_series"
    if single:
        raw["timestamp"] += pd.to_timedelta(
            raw.asset_id.map({"a": 0, "b": 3, "c": 6}), unit="s"
        )
        raw["asset_id"] = "a"
        raw["session_date"] = raw.timestamp.dt.normalize()
        declaration["column_roles"]["session_date"] = "context"
    raw["label_available_at"] = raw.timestamp
    raw.loc[0, "label_available_at"] = pd.Timestamp("2030-01-01")
    raw.loc[4, "eligible"] = False
    raw.loc[5, "x"] = np.nan
    raw.loc[raw.timestamp.dt.normalize().eq(dates[-1]), "label_available_at"] = (
        pd.Timestamp("2030-01-01")
    )
    source = {
        "path": str(
            write_dataset(
                tmp_path / "native",
                raw.iloc[::-1],
                spec=DataSpec(
                    **{
                        **declaration,
                        "primary_key": tuple(declaration["primary_key"]),
                        "partition_columns": tuple(declaration["partition_columns"]),
                    }
                ),
            ).root
        )
    }
    name = "model"
    parameters = (
        {"alpha": 0.01, "max_iter": None}
        if algorithm == "lasso_regression"
        else (
            {
                "device": "cpu",
                "n_jobs": 1,
                "n_estimators": 3,
                "early_stopping_rounds": 1,
            }
            if algorithm == "xgboost_regressor"
            else {
                "device": "cpu",
                "epochs": 3,
                "early_stopping_rounds": 1,
                "hidden_size": 4,
                "batch_size": 9,
                "sequence_length": 3,
                "sequence_group_columns": ["session_date"] if single else [],
            }
        )
    )
    model = fit_config(tmp_path, source["path"])["model"]
    model.update(
        model_id=name,
        model_algorithm=algorithm,
        hyperparameters=parameters,
        sample_filter_mode="unfiltered" if single else "eligible",
    )
    sessions = dates.strftime("%Y-%m-%d").tolist()
    results = run_sessions(
        {
            "inputs": source,
            "output": str(tmp_path / "rolling"),
            "market_id": "a_share",
            "models": [model],
            **(
                {"parameter_candidates": {name: [{"alpha": 1.0}, {"alpha": 0.01}]}}
                if algorithm == "lasso_regression"
                else {}
            ),
            "training_start": sessions[0],
            "sessions": sessions,
            "min_train_sessions": 2,
            "validation_sessions": 1,
            "prediction_sessions": 3,
            "rolling_window": {"start": sessions[3], "end": sessions[-1]},
            "evaluation_cutoff": sessions[-1] + "T23:59:59",
            "label_unit": "decimal_return",
            "evaluation": {
                "evaluation_id": "fold",
                "market_id": "a_share",
                "research_dataset_id": "synthetic",
                "signals": [
                    {
                        "signal_id": "score",
                        "source_kind": "prediction",
                        "source_column": "score",
                    }
                ],
                "label_columns": ["label"],
                "evaluation_axis": axis,
                "top_quantile": None if single else 0.5,
            },
        }
    )
    assets = ["a"] if single else ["a", "b", "c"]
    future = _future_daily(
        tmp_path,
        source,
        {name: [str(r["predictions"]) for r in results]},
        dates=sessions[3:],
        assets=assets,
        axis=axis,
        top_quantile=None if single else 0.5,
    )
    import quant_research_platform.research.models.fitting as fitting

    monkeypatch.setattr(
        fitting, "_fit_model_result", lambda *a, **kw: pytest.fail("statistics fitted")
    )
    output = evaluate_model_periods(
        {
            "inputs": source,
            "evaluation_axis": axis,
            "top_quantile": None if single else 0.5,
            "sessions": sessions,
            "asset_ids": assets,
            "prediction_daily": {name: str(future)},
            "output": str(tmp_path / "periods"),
            "periods": [
                {
                    "period_id": r["fold_id"],
                    "prediction_window": {
                        "start": sessions[3 + i * 3],
                        "end": sessions[5 + i * 3],
                    },
                    "models": {name: str(r["model"])},
                }
                for i, r in enumerate(results)
            ],
        }
    )
    metrics, epochs, lineage = _tables(output)
    for item in results:
        fitted, facts = load_model(item["model"])
        # Independently replay the frozen predictor on the full small causal context.
        frame = (
            raw.loc[raw.timestamp.le(pd.Timestamp(facts["fit_cutoff"]))]
            .sort_values(["timestamp", "asset_id"])
            .reset_index(drop=True)
        )
        predicted = prediction_callable(
            tuple(fitted.metadata["input_columns"]), fitted.prediction_model
        )(frame)
        keep = (
            label_maturity_mask(
                frame,
                fitting_cutoff=facts["fit_cutoff"],
                availability_column="label_available_at",
                target_column="label",
            )
            & frame.x.notna()
        )
        if not single:
            keep &= frame.eligible
        errors = (
            pd.Series((predicted[keep] - frame.loc[keep, "label"]) ** 2)
            .groupby(frame.loc[keep, "timestamp"].dt.normalize())
            .mean()
        )
        selected = metrics.loc[
            metrics.period_id.eq(item["fold_id"])
            & metrics.stage.eq("training")
            & metrics.metric.eq("mse")
        ].iloc[0]
        assert selected.value == pytest.approx(errors.mean(), rel=2e-5, abs=1e-9)
        assert selected.observation_count == int(keep.sum())
        if algorithm == "lasso_regression":
            assert selected.iterations == int(fitted.prediction_model.estimator.n_iter_)
        elif algorithm == "xgboost_regressor":
            assert (
                selected.iterations
                == fitted.prediction_model.estimator.get_booster().num_boosted_rounds()
            )
        cv = facts["training_selection"]
        winner = next(
            c
            for c in cv["candidates"]
            if c["hyperparameters"] == cv["selected_hyperparameters"]
        )
        row = metrics.loc[
            metrics.period_id.eq(item["fold_id"])
            & metrics.stage.eq("validation")
            & metrics.selected
            & metrics.metric.eq("mse")
        ].iloc[0]
        assert row.value == pytest.approx(
            winner["validation_daily_mse"], rel=1e-12, abs=1e-14
        )
        assert row.observation_count == cv["common_rows"]
        assert row.evaluation_axis == axis
        if algorithm != "lasso_regression":
            assert row.iterations == winner["early_stopping"]["best_iteration"] + 1
    future_rows = metrics.loc[metrics.stage.eq("prediction") & metrics.metric.eq("mse")]
    assert future_rows.valid_session_count.tolist() == [3, 2]
    if algorithm in {"lstm_regressor", "transformer_regressor"}:
        final = epochs.loc[epochs.stage.eq("final_fit")]
        candidate = epochs.loc[epochs.stage.eq("candidate_fit")]
        assert not final.empty and final.validation_loss.isna().all()
        assert not candidate.empty and candidate.validation_loss.notna().all()
        assert epochs.training_loss_scope.eq("online_epoch_weighted").all()
    elif algorithm == "xgboost_regressor":
        assert epochs.epoch.isna().all()
        assert epochs.boosting_round.notna().all()
        assert epochs.validation_loss.notna().all()
        assert epochs.stage.eq("candidate_fit").all()
    else:
        assert epochs.empty
    assert all(
        item["historical_cv_curves"]
        == ("not_saved" if algorithm == "lasso_regression" else "saved")
        for item in lineage["models"]
    )
    artifact = read_statistics_artifact(output)
    with artifact.table("model_daily_metrics.parquet") as daily:
        assert set(daily.stage) == {"training", "validation", "prediction"}
        for item in results:
            candidates = daily.loc[
                daily.period_id.eq(item["fold_id"]) & daily.stage.eq("validation")
            ]
            assert candidates.candidate_id.nunique() == (
                2 if algorithm == "lasso_regression" else 1
            )
            assert candidates.loc[candidates.selected, "candidate_id"].nunique() == 1
        frozen = daily.loc[daily.stage.eq("training") & daily.metric.eq("mse")]
        for period, values in frozen.groupby("period_id"):
            expected = metrics.loc[
                metrics.period_id.eq(period)
                & metrics.stage.eq("training")
                & metrics.metric.eq("mse"),
                "value",
            ].iloc[0]
            assert values.value.mean() == pytest.approx(expected)


def test_explicit_fit_has_unsaved_cv_and_rejects_truncated_training_input(tmp_path):
    path = data(tmp_path)
    source = {"path": str(path), "spec": _declaration()}
    model = fit({**fit_config(tmp_path, path), "inputs": source})
    predictions = predict(
        {
            "model": str(model),
            "inputs": source,
            "prediction_window": {"start": "2020-01-04", "end": "2020-01-05"},
            "output": str(tmp_path / "pred"),
        }
    )
    continued = predict(
        {
            "model": str(model),
            "inputs": source,
            "prediction_window": {"start": "2020-01-06", "end": None},
            "output": str(tmp_path / "open-pred"),
        }
    )
    future = _future_daily(
        tmp_path,
        source,
        {"linear": [str(predictions), str(continued)]},
        dates=["2020-01-04", "2020-01-05", "2020-01-06"],
        assets=["a", "b", "c"],
        axis="cross_sectional",
    )
    config = {
        "inputs": source,
        "evaluation_axis": "cross_sectional",
        "sessions": pd.date_range("2020-01-01", periods=6)
        .strftime("%Y-%m-%d")
        .tolist(),
        "asset_ids": ["a", "b", "c"],
        "prediction_daily": {"linear": str(future)},
        "periods": [
            {
                "period_id": "one",
                "prediction_window": {"start": "2020-01-04", "end": "2020-01-06"},
                "models": {"linear": str(model)},
            }
        ],
        "output": str(tmp_path / "periods"),
    }
    metrics, _, _ = _tables(evaluate_model_periods(config))
    cv = metrics.loc[metrics.stage.eq("validation")]
    assert cv.reason.eq("not_saved").all() and cv.value.isna().all()
    assert cv.observation_count.isna().all()
    no_zero = _future_daily(
        tmp_path / "no-zero",
        source,
        {"linear": [str(predictions), str(continued)]},
        dates=["2020-01-04", "2020-01-05", "2020-01-06"],
        assets=["a", "b", "c"],
        axis="cross_sectional",
        baselines=(),
    )
    incomplete, _, _ = _tables(
        evaluate_model_periods(
            {
                **config,
                "prediction_daily": {"linear": str(no_zero)},
                "output": str(tmp_path / "no-zero-periods"),
            }
        )
    )
    forecast = incomplete.loc[incomplete.stage.eq("prediction")].set_index("metric")
    assert len(forecast) == 6
    assert forecast.loc["r2_vs_zero", "reason"] == "not_saved"
    assert pd.isna(forecast.loc["r2_vs_zero", "value"])
    assert forecast.loc["mse", "value"] == pytest.approx(29 / 3)
    by_asset = _future_daily(
        tmp_path / "by-asset",
        source,
        {"linear": [str(predictions), str(continued)]},
        dates=["2020-01-04", "2020-01-05", "2020-01-06"],
        assets=["a", "b", "c"],
        axis="single_asset_time_series",
    )
    subset, _, _ = _tables(
        evaluate_model_periods(
            {
                **config,
                "evaluation_axis": "single_asset_time_series",
                "asset_ids": ["a"],
                "prediction_daily": {"linear": str(by_asset)},
                "output": str(tmp_path / "subset-periods"),
            }
        )
    )
    assert set(subset.loc[subset.stage.eq("training"), "asset_id"]) == {"a", "b", "c"}
    assert set(subset.loc[subset.stage.eq("prediction"), "asset_id"]) == {"a"}
    mse = subset.loc[subset.stage.eq("prediction") & subset.metric.eq("mse")].iloc[0]
    assert mse.observation_count == 3 and mse.value == pytest.approx(29 / 3)
    empty_prediction = predict(
        {
            "model": str(model),
            "inputs": source,
            "prediction_window": {"start": "2020-01-04", "end": None},
            "asset_ids": ["unobserved"],
            "output": str(tmp_path / "empty-pred"),
        }
    )
    empty_daily = _future_daily(
        tmp_path / "empty",
        source,
        {"linear": [str(empty_prediction)]},
        dates=["2020-01-04", "2020-01-05", "2020-01-06"],
        assets=["a", "b", "c"],
        axis="cross_sectional",
    )
    empty_metrics, _, _ = _tables(
        evaluate_model_periods(
            {
                **config,
                "prediction_daily": {"linear": str(empty_daily)},
                "output": str(tmp_path / "empty-periods"),
            }
        )
    )
    forecast = empty_metrics.loc[empty_metrics.stage.eq("prediction")]
    assert {"mse", "rmse", "mae", "ic", "rank_ic", "r2_vs_zero"} <= set(forecast.metric)
    assert forecast.status.eq("UNDEFINED").all()
    assert forecast.observation_count.eq(0).all() and forecast.session_count.eq(3).all()
    with pytest.raises(ValueError, match="training_sample_mismatch"):
        evaluate_model_periods(
            {
                **config,
                "inputs": {
                    **source,
                    "time_window": {"start": "2020-01-02", "end": "2020-01-09"},
                },
                "output": str(tmp_path / "truncated"),
            }
        )
    with pytest.raises(ValueError, match="evaluation_input_session_not_declared"):
        evaluate_model_periods(
            {
                **config,
                "sessions": config["sessions"][1:],
                "output": str(tmp_path / "missing-clock"),
            }
        )
    with pytest.raises(ValueError, match="prediction_assets_mismatch"):
        evaluate_model_periods(
            {**config, "asset_ids": ["a"], "output": str(tmp_path / "wrong-domain")}
        )
    with pytest.raises(ValueError, match="must_cover_whole_days"):
        evaluate_model_periods(
            {
                **config,
                "periods": [
                    {
                        **config["periods"][0],
                        "prediction_window": {
                            "start": "2020-01-04T12:00:00",
                            "end": "2020-01-06T23:59:59",
                        },
                    }
                ],
                "output": str(tmp_path / "partial-day"),
            }
        )
    other = fit(
        {
            **fit_config(tmp_path, path, model_id="other"),
            "inputs": source,
            "model": fit_config(tmp_path, path)["model"],
            "fit_windows": [{"start": "2020-01-01", "end": "2020-01-02"}],
            "fit_cutoff": "2020-01-02",
        }
    )
    with pytest.raises(ValueError, match="prediction_fit_mismatch"):
        evaluate_model_periods(
            {
                **config,
                "periods": [{**config["periods"][0], "models": {"linear": str(other)}}],
                "output": str(tmp_path / "wrong-fit"),
            }
        )


@pytest.mark.parametrize("algorithm", ["linear_regression", "lstm_regressor"])
def test_utc_aware_fit_and_prediction_windows_cross_months(tmp_path, algorithm):
    path = data(tmp_path)
    frame = pd.read_parquet(path)
    dates = pd.date_range("2020-01-29", periods=9)
    frame["timestamp"] = frame.timestamp.map(
        dict(zip(pd.date_range("2020-01-01", periods=9), dates, strict=True))
    )
    frame["label_available_at"] = frame.timestamp
    gap = frame.timestamp.isin(dates[2:4])
    frame.loc[gap, "x"] *= 7
    frame.loc[gap, "label"] = np.nan
    frame.loc[gap, "label_available_at"] = pd.Timestamp("2030-01-01")
    frame.to_parquet(path, index=False, row_group_size=2)
    source = {"path": str(path), "spec": _declaration()}
    recipe = fit_config(tmp_path, path)["model"]
    recipe["model_algorithm"] = algorithm
    if algorithm == "lstm_regressor":
        recipe["hyperparameters"] = {
            "device": "cpu",
            "epochs": 2,
            "sequence_length": 4,
            "hidden_size": 4,
            "batch_size": 9,
        }
    model = fit(
        {
            **fit_config(tmp_path, path),
            "inputs": source,
            "model": recipe,
            "fit_windows": [
                {
                    "start": "2020-01-29T00:00:00Z",
                    "end": "2020-01-30T23:59:59Z",
                },
                {
                    "start": "2020-02-02T00:00:00Z",
                    "end": "2020-02-03T23:59:59Z",
                },
            ],
            "fit_cutoff": "2020-02-03T23:59:59Z",
        }
    )
    window = {"start": "2020-02-04T00:00:00Z", "end": "2020-02-06T23:59:59.999999999Z"}
    predictions = predict(
        {
            "inputs": source,
            "model": str(model),
            "prediction_window": window,
            "output": str(tmp_path / "pred"),
        }
    )
    future = _future_daily(
        tmp_path,
        source,
        {"linear": [str(predictions)]},
        dates=dates[6:].strftime("%Y-%m-%d").tolist(),
        assets=["a", "b", "c"],
        axis="cross_sectional",
    )
    predictions.rename(tmp_path / "moved_predictions")
    clock = dates if algorithm == "lstm_regressor" else dates.difference(dates[2:4])
    config = {
        "inputs": source,
        "evaluation_axis": "cross_sectional",
        "sessions": clock.strftime("%Y-%m-%d").tolist(),
        "asset_ids": ["a", "b", "c"],
        "periods": [
            {
                "period_id": "aware",
                "prediction_window": window,
                "models": {"linear": str(model)},
            }
        ],
        "prediction_daily": {"linear": str(future)},
        "output": str(tmp_path / "periods"),
    }
    metrics, _, _ = _tables(evaluate_model_periods(config))
    mse = metrics.loc[metrics.metric.eq("mse")].set_index("stage")
    if algorithm == "linear_regression":
        assert mse.loc["training", "value"] == pytest.approx(17 / 4)
    else:
        frozen, facts = load_model(model)
        context = (
            frame.loc[frame.timestamp.lt(dates[6])]
            .sort_values(["timestamp", "asset_id"])
            .reset_index(drop=True)
        )
        predicted = prediction_callable(
            tuple(frozen.metadata["input_columns"]), frozen.prediction_model
        )(context)
        sample = context.label.notna()
        errors = (
            ((context.loc[sample, "label"] - predicted[sample]) ** 2)
            .groupby(context.loc[sample, "timestamp"])
            .mean()
        )
        assert mse.loc["training", "value"] == pytest.approx(
            errors.mean(), rel=2e-5, abs=1e-9
        )
        with pytest.raises(ValueError, match="evaluation_input_session_not_declared"):
            evaluate_model_periods(
                {
                    **config,
                    "sessions": dates.difference(dates[2:4])
                    .strftime("%Y-%m-%d")
                    .tolist(),
                    "output": str(tmp_path / "missing-context-clock"),
                }
            )
    assert mse.loc["training", "observation_count"] == 12
    assert mse.loc["prediction", "observation_count"] == 9
    assert mse.loc["prediction", "window_start"] == pd.Timestamp("2020-02-04")
