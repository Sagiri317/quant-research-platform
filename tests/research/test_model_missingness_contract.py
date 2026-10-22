"""Approved model-stage missingness and calendar-domain regressions."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.explanations import read_explanation
from quant_research_platform.contracts.models import ModelSpec, TrainingWeighting
from quant_research_platform.research.models.explanations import explain
from quant_research_platform.research.models.fitting import _collect_training_frame
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit_model
from quant_research_platform.research.models.scaler import (
    equal_timestamp_sample_weights,
)
from tests.research.direct_model_test_support import fit_frame, predict_frame


def frame():
    dates = pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-03"] * 3)
    return pd.DataFrame(
        dict(
            timestamp=dates,
            asset_id=["a", "b", "c"] * 2,
            x=[1.0, np.nan, 3.0, 5.0, 7.0, 9.0],
            z=[2.0, 4.0, 6.0, 8.0, np.nan, 12.0],
            label=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            eligible=True,
            label_available_at=dates,
        )
    )


def spec(algorithm="ridge_regression"):
    return ModelSpec(
        model_id=algorithm,
        model_algorithm=algorithm,
        input_factor_columns=("x", "z"),
        target_column="label",
    )


@pytest.mark.parametrize(
    "algorithm",
    [
        "linear_regression",
        "ridge_regression",
        "decision_tree_regressor",
        "random_forest_regressor",
    ],
)
def test_train_only_missingness_roundtrip(tmp_path, algorithm):
    data = frame()
    case = fit_frame(tmp_path / "model", data, spec(algorithm))
    assert case.result.metadata["rows_after_input_label_dropna"] == 6
    scaler = case.result.prediction_model.scaler
    if algorithm in {"linear_regression", "ridge_regression"}:
        assert scaler.medians == (5.0, 6.0)
        assert scaler.transform(data[["x", "z"]]).shape == (6, 4)
    else:
        assert scaler.native_missing
        assert scaler.transform(data[["x", "z"]]).isna().sum().sum() == 2
    future = data.iloc[:3].copy().assign(timestamp=pd.Timestamp("2024-01-04"))
    future.loc[future.index[0], ["x", "z"]] = np.nan
    future.loc[future.index[1], "z"] = 10000.0
    predictions, _, _ = predict_frame(
        tmp_path / "future",
        future,
        case,
        window={"start": "2024-01-04", "end": "2024-01-04"},
    )
    assert predictions.asset_id.tolist() == ["b", "c"]
    assert scaler.medians == (() if scaler.native_missing else (5.0, 6.0))


@pytest.mark.parametrize(
    "algorithm",
    [
        "linear_regression",
        "ridge_regression",
        "lasso_regression",
        "huber_regression",
        "gradient_boosting_regressor",
    ],
)
def test_imputed_model_cannot_learn_an_all_missing_feature_median(algorithm):
    data = frame().assign(z=np.nan)
    with pytest.raises(ValueError, match="model_training_feature_all_missing"):
        fit_model(
            [data],
            spec(algorithm),
            fit_windows=[{"start": "2024-01-01", "end": "2024-01-03"}],
            fit_cutoff="2024-01-03",
        )


@pytest.mark.parametrize(
    "algorithm",
    [
        "decision_tree_regressor",
        "random_forest_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
    ],
)
def test_native_missing_training_column_roundtrips_without_imputation(
    tmp_path, algorithm
):
    parameters = {"max_depth": 2}
    if algorithm != "decision_tree_regressor":
        parameters.update(n_estimators=4, n_jobs=1)
    if algorithm == "lightgbm_regressor":
        parameters.update(min_child_samples=1, verbosity=-1)
    model = replace(spec(algorithm), hyperparameters=parameters)
    data = frame().assign(x=np.arange(1.0, 7.0) * 1000, z=np.nan)
    data = pd.concat(
        [data, data.iloc[[-1]].assign(asset_id="d", x=np.nan, label=999.0)],
        ignore_index=True,
    )
    case = fit_frame(tmp_path / "model", data, model)
    assert case.result.metadata["rows_input"] == 7
    assert case.result.metadata["rows_after_input_label_dropna"] == 6
    assert case.result.metadata["input_columns"] == ["x", "z"]
    assert case.result.metadata["feature_importance"]["z"] == 0.0
    scaler = case.result.prediction_model.scaler
    assert scaler.native_missing
    assert scaler.columns == ("x", "z")
    assert scaler.medians == ()
    pd.testing.assert_frame_equal(
        scaler.transform(data[["x", "z"]]), data[["x", "z"]], check_exact=True
    )
    frozen = scaler.to_payload()
    saved_model = (case.root / "predictor.pkl").read_bytes()

    future = data.iloc[:3].copy().assign(timestamp=pd.Timestamp("2024-01-04"))
    future.loc[future.index[:2], "x"] = np.nan
    future.loc[future.index[1], "z"] = 10000.0
    pd.testing.assert_frame_equal(
        scaler.transform(future[["x", "z"]]), future[["x", "z"]], check_exact=True
    )
    direct = case.result.prediction_model.predict(future[["x", "z"]])
    assert np.isnan(direct[0])
    assert np.isfinite(direct[1:]).all()
    predictions, _, _ = predict_frame(
        tmp_path / "future",
        future,
        case,
        window={"start": "2024-01-04", "end": "2024-01-04"},
    )
    assert predictions.asset_id.tolist() == ["b", "c"]
    np.testing.assert_allclose(
        predictions.prediction_score, direct[1:], rtol=1e-12, atol=1e-14
    )
    method = (
        "native_tree"
        if algorithm in {"xgboost_regressor", "lightgbm_regressor"}
        else "tree_shap_path"
    )
    _, tables = read_explanation(
        explain(
            {
                "model": str(case.root),
                "inputs": str(tmp_path / "future" / "input.parquet"),
                "input_spec": case.config["input_spec"],
                "output": str(tmp_path / "explanation"),
                "window": {"start": "2024-01-04", "end": "2024-01-04"},
                "methods": [method],
                "selection": {"max_rows": 3, "max_dates": 1},
                "sample_role": "validation",
                "output_space": "model_output",
            }
        )
    )
    assert tables["predictions"].status.eq("DEFINED").all()
    np.testing.assert_allclose(
        tables["predictions"].prediction_score, direct[1:], rtol=1e-12, atol=1e-14
    )
    missing_contributions = tables["contributions"].loc[
        lambda rows: rows.feature.eq("z")
    ]
    assert len(missing_contributions) == 2
    assert missing_contributions.contribution.eq(0.0).all()
    reloaded, _ = load_model(case.root)
    reloaded_predictions = reloaded.prediction_model.predict(future[["x", "z"]])
    np.testing.assert_array_equal(np.isnan(reloaded_predictions), np.isnan(direct))
    np.testing.assert_allclose(reloaded_predictions, direct, rtol=1e-12, atol=1e-14)
    assert reloaded.prediction_model.scaler.to_payload() == frozen
    assert (case.root / "predictor.pkl").read_bytes() == saved_model


@pytest.mark.parametrize("algorithm", ["ridge_regression", "decision_tree_regressor"])
@pytest.mark.parametrize("technical_fill", [False, True])
def test_all_original_training_inputs_missing_has_no_training_samples(
    algorithm, technical_fill
):
    data = frame().assign(x=np.nan, z=np.nan)
    model = spec(algorithm)
    if technical_fill:
        data = data.rename(columns={"x": "feature_x", "z": "feature_z"}).assign(
            feature_x=0.0,
            feature_z=0.0,
            feature_observed_x=0.0,
            feature_observed_z=0.0,
            feature_missing_z=1.0,
        )
        model = replace(
            model,
            input_factor_columns=(),
            input_feature_columns=("feature_x", "feature_z", "feature_missing_z"),
        )
    with pytest.raises(ValueError, match="model_training_sample_empty"):
        fit_model(
            [data],
            model,
            fit_windows=[{"start": "2024-01-01", "end": "2024-01-03"}],
            fit_cutoff="2024-01-03",
            declared_columns=data.columns,
        )


def test_rank_all_missing_column_keeps_declared_zero_weight(tmp_path):
    data = frame().assign(z=np.nan)
    case = fit_frame(tmp_path / "rank", data, spec("ic_weighted_rank_ensemble"))
    predictor = case.result.prediction_model
    weights = dict(zip(predictor.columns, predictor.weights, strict=True))
    assert weights["x"] == pytest.approx(1.0)
    assert weights["z"] == 0.0


def test_decay_uses_explicit_missing_session_not_observation_count():
    dates = pd.to_datetime(["2024-01-01", "2024-01-03"])
    weights = equal_timestamp_sample_weights(
        dates,
        TrainingWeighting(half_life_periods=1),
        trading_sessions=pd.date_range("2024-01-01", "2024-01-03"),
    )
    assert weights[1] / weights[0] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="training_decay_calendar_required"):
        equal_timestamp_sample_weights(dates, TrainingWeighting(half_life_periods=252))


def test_enabled_decay_requires_matching_complete_own_calendar():
    data = frame().assign(market_type=1)
    calendar = pd.DataFrame(
        dict(
            calendar_date=pd.date_range("2024-01-01", "2024-01-03"),
            source_market_type=1,
            is_trading_day=True,
        )
    )
    model = replace(spec(), training_weighting=TrainingWeighting(half_life_periods=1))
    prepared = _collect_training_frame(
        [data], model, fitting_cutoff="2024-01-03", trading_calendar=calendar
    )
    weights = prepared.preprocessing[0]
    assert weights[-1] / weights[0] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="calendar_coverage_missing"):
        _collect_training_frame(
            [data],
            model,
            fitting_cutoff="2024-01-03",
            trading_calendar=calendar.iloc[[0, 2]],
        )
    with pytest.raises(ValueError, match="calendar_coverage_missing"):
        _collect_training_frame(
            [data.assign(market_type=[1, 4, 1] * 2)],
            model,
            fitting_cutoff="2024-01-03",
            trading_calendar=calendar,
        )
    mixed = data.assign(market_type=[1, 4, 1] * 2)
    shared = pd.concat((calendar, calendar.assign(source_market_type=4)))
    prepared = _collect_training_frame(
        [mixed], model, fitting_cutoff="2024-01-03", trading_calendar=shared
    )
    assert prepared.preprocessing[0].tolist() == pytest.approx(weights.tolist())
    shared.loc[
        shared.source_market_type.eq(4) & shared.calendar_date.eq("2024-01-02"),
        "is_trading_day",
    ] = False
    with pytest.raises(ValueError, match="common_session_clock_required"):
        _collect_training_frame(
            [mixed], model, fitting_cutoff="2024-01-03", trading_calendar=shared
        )
