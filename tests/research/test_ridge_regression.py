"""Weighted Ridge matches independent penalized normal equations end to end."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.models import ModelSpec
from tests.research.direct_model_test_support import fit_frame, predict_frame


@pytest.mark.parametrize("scaling", [True, False])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_ridge_weighted_math_prediction_and_persistence(
    tmp_path, scaling, fit_intercept
):
    dates = pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 5)
    frame = pd.DataFrame(
        dict(
            timestamp=dates,
            asset_id=["a", "b", "c", "a", "b", "ineligible", "immature", "missing"],
            x=[10.0, 20.0, np.nan, 31.0, 50.0, 10000.0, -10000.0, np.nan],
            z=[2.0, np.nan, 6.0, 8.0, 12.0, 10000.0, -10000.0, np.nan],
            constant=[7.0] * 7 + [np.nan],
            label=[0.1, 0.2, 0.15, 0.3, 0.5, 999.0, -999.0, 500.0],
            eligible=[True] * 5 + [False, True, True],
            label_available_at=dates,
        )
    )
    frame.loc[6, "label_available_at"] = pd.Timestamp("2024-01-03")
    original = frame.copy(deep=True)
    columns = ("z", "x", "constant")
    alpha = 2.5
    parameters = {"alpha": alpha, "fit_intercept": fit_intercept, "solver": "cholesky"}
    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=columns,
        target_column="label",
        hyperparameters=parameters,
        random_seed=19,
        **({} if scaling else {"training_scaling": False}),
    )
    assert spec.training_scaling is scaling
    case = fit_frame(tmp_path / "fit", frame, spec)
    metadata = case.result.metadata
    predictor = case.result.prediction_model
    assert metadata["rows_after_input_label_dropna"] == 5
    assert metadata["dependency_status"]["package"] == "sklearn"
    assert metadata["hyperparameters"] == parameters
    assert case.facts["model"]["training_scaling"] is scaling
    assert case.facts["candidate_parameters"] == parameters
    pd.testing.assert_frame_equal(frame, original)

    # The five legal rows have equal total mass per date and sum to N_train.
    weights = np.array([5 / 6] * 3 + [5 / 4] * 2)
    medians = np.array([7.0, 25.5, 7.0])
    matrix = np.array(
        [
            [2.0, 10.0, 7.0],
            [7.0, 20.0, 7.0],
            [6.0, 25.5, 7.0],
            [8.0, 31.0, 7.0],
            [12.0, 50.0, 7.0],
        ]
    )
    mean = np.average(matrix, axis=0, weights=weights) if scaling else np.zeros(3)
    scale = (
        np.sqrt(np.average((matrix - mean) ** 2, axis=0, weights=weights))
        if scaling
        else np.ones(3)
    )
    scale[scale == 0] = 1.0
    transformed = (matrix - mean) / scale
    design = (
        np.column_stack([np.ones(5), transformed]) if fit_intercept else transformed
    )
    penalty = np.eye(design.shape[1]) * alpha
    if fit_intercept:
        penalty[0, 0] = 0.0
    target = np.array([0.1, 0.2, 0.15, 0.3, 0.5])
    expected = np.linalg.solve(
        design.T @ (weights[:, None] * design) + penalty,
        design.T @ (weights * target),
    )
    intercept = expected[0] if fit_intercept else 0.0
    coefficients = expected[1:] if fit_intercept else expected
    np.testing.assert_allclose(predictor.coef_, coefficients, rtol=1e-11, atol=1e-13)
    assert predictor.intercept_ == pytest.approx(intercept, rel=1e-11, abs=1e-13)
    np.testing.assert_allclose(
        [metadata["feature_importance"][column] for column in columns],
        np.abs(coefficients),
        rtol=1e-11,
        atol=1e-13,
    )
    np.testing.assert_allclose(predictor.scaler.mean_, mean, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(predictor.scaler.scale_, scale, rtol=1e-13, atol=1e-13)
    assert predictor.scaler.medians == tuple(medians)
    assert not predictor.scaler.native_missing
    frozen = predictor.scaler.to_payload()

    future = frame.iloc[:3].copy().assign(timestamp=pd.Timestamp("2024-01-03"))
    future.loc[:, "x"] = [100.0, np.nan, np.nan]
    future.loc[:, "z"] = [2.0, 8.0, np.nan]
    future.loc[:, "constant"] = [7.0, 7.0, np.nan]
    query = np.array([[2.0, 100.0, 7.0], [8.0, 25.5, 7.0]])
    expected_predictions = intercept + ((query - mean) / scale) @ coefficients
    direct = predictor.predict(future.loc[:, columns])
    np.testing.assert_allclose(direct[:2], expected_predictions, rtol=1e-11, atol=1e-13)
    assert np.isnan(direct[2])
    predictions, _, _ = predict_frame(
        tmp_path / "predict",
        future,
        case,
        window={"start": "2024-01-03", "end": "2024-01-03"},
    )
    assert predictions.asset_id.tolist() == ["a", "b"]
    np.testing.assert_allclose(
        predictions.prediction_score,
        expected_predictions,
        rtol=1e-11,
        atol=1e-13,
    )
    assert predictor.scaler.to_payload() == frozen
