"""Approved model-stage missingness and calendar-domain regressions."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.models import ModelSpec, TrainingWeighting
from quant_research_platform.research.models.fitting import _collect_training_frame
from quant_research_platform.research.models.production import fit_model
from quant_research_platform.research.models.scaler import (
    equal_timestamp_sample_weights,
)
from quant_research_platform.research.models.trial_selection import (
    evaluate_model_trial_metrics_from_chunks,
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
    ["ridge_regression", "decision_tree_regressor"],
)
def test_all_training_missing_selected_feature_fails(algorithm):
    data = frame().assign(z=np.nan)
    with pytest.raises(ValueError, match="model_training_feature_all_missing"):
        fit_model(
            [data],
            spec(algorithm),
            fit_windows=[{"start": "2024-01-01", "end": "2024-01-03"}],
            fit_cutoff="2024-01-03",
        )


def test_rank_all_missing_column_keeps_declared_zero_weight(tmp_path):
    data = frame().assign(z=np.nan)
    case = fit_frame(tmp_path / "rank", data, spec("ic_weighted_rank_ensemble"))
    predictor = case.result.prediction_model
    weights = dict(zip(predictor.columns, predictor.weights, strict=True))
    assert weights["x"] == pytest.approx(1.0)
    assert weights["z"] == 0.0


def test_row_cap_keeps_complete_dates_and_never_reuses_old_date():
    data = frame()
    prepared = _collect_training_frame(
        [data.iloc[:2], data.iloc[2:4], data.iloc[4:]],
        replace(spec(), max_train_rows=4),
        fitting_cutoff="2024-01-03",
    )
    assert len(prepared.training) == 3
    assert prepared.training.timestamp.eq(pd.Timestamp("2024-01-03")).all()
    prepared = _collect_training_frame(
        [data], replace(spec(), max_train_rows=2), fitting_cutoff="2024-01-03"
    )
    assert prepared.training.empty


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


def test_mse_dates_are_equal_weight_not_rows():
    data = pd.DataFrame(
        dict(
            timestamp=pd.to_datetime(["2024-01-01"] + ["2024-01-02"] * 100),
            asset_id=[str(i) for i in range(101)],
            label=0.0,
            score=[1.0] + [0.0] * 100,
        )
    )
    metrics = evaluate_model_trial_metrics_from_chunks(
        [data.iloc[:50], data.iloc[50:]],
        model_id="model",
        label_column="label",
        prediction_column="score",
    )
    assert metrics["validation_mse"] == pytest.approx(0.5)


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
