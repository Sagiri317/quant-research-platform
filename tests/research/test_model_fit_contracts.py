"""Model fitting, legal sample retention and owned preprocessing buffers."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models import fitting
from quant_research_platform.research.models.estimators import (
    build_estimator_from_model_spec,
)


def test_fitting_counts_raw_filter_and_label_rows_once() -> None:
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="model_target",
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"]
            ),
            "asset_id": ["a", "b", "c", "d"],
            "x": [1.0, 2.0, 3.0, 4.0],
            "model_target": [1.0, 2.0, 3.0, 4.0],
            "eligible": [True, True, True, False],
            "model_target_available_at": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2019-01-03", "2018-01-04"]
            ),
        }
    )
    result = fitting._fit_model_result(
        fitting._collect_training_frame((frame,), spec, fitting_cutoff="2018-12-31"),
        spec,
        estimator=build_estimator_from_model_spec(spec),
    )
    model_result = result
    assert model_result.metadata["rows_input"] == 4
    assert model_result.metadata["rows_after_filter_mode"] == 3
    assert model_result.metadata["rows_after_input_label_dropna"] == 2


def test_convergence_warning_is_a_model_fit_failure() -> None:
    phase = np.arange(40) * (2 * np.pi / 40)
    common = np.sqrt(2) * np.sin(phase)
    difference = 0.01 * np.sqrt(2) * np.cos(phase)
    spec = ModelSpec(
        model_id="linear_warning",
        model_algorithm="lasso_regression",
        input_factor_columns=("x", "z"),
        target_column="label",
        hyperparameters={"alpha": 0.01, "tol": 1e-12, "max_iter": 1},
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2018-01-01", periods=40, freq="s"),
            "x": common + difference,
            "z": common - difference,
            "label": 2 * common,
        }
    )
    with pytest.raises(ValueError, match="model_fit_non_converged"):
        fitting._fit_predictor(
            spec,
            prepared_training=fitting._collect_training_frame(
                [frame.assign(eligible=True, label_available_at=frame.timestamp)],
                spec,
                fitting_cutoff="2018-01-02",
            ),
            estimator=build_estimator_from_model_spec(spec),
        )


def test_training_retention_preserves_all_legal_rows_in_canonical_key_order():
    chunk_rows = 5
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.bdate_range("2017-01-02", periods=16).repeat(2),
            "label_available_at": pd.Timestamp("2017-02-01"),
            "asset_id": ["b", "a"] * 16,
            "x": np.arange(32.0),
            "label": np.arange(32.0) * 2 + 1,
            "eligible": True,
        }
    )
    frame.loc[frame.index % 7 == 0, "label_available_at"] = pd.Timestamp("2018-01-01")
    expected = frame.loc[
        frame.index % 7 != 0, ["timestamp", "asset_id", "x", "label"]
    ].copy()
    expected["timestamp"] = expected["timestamp"].dt.as_unit("ns")
    prepared = fitting._collect_training_frame(
        (frame.iloc[start : start + chunk_rows] for start in range(0, 32, chunk_rows)),
        spec,
        fitting_cutoff="2017-12-31",
    )
    pd.testing.assert_frame_equal(
        prepared.training,
        expected.sort_values(["timestamp", "asset_id"], ignore_index=True),
        check_exact=True,
    )


@pytest.mark.parametrize(
    "algorithm,parameters",
    [("lasso_regression", {"alpha": 0.01, "copy_X": copy}) for copy in (True, False)]
    + [("ridge_regression", {"alpha": 0.1, "copy_X": copy}) for copy in (True, False)]
    + [("xgboost_regressor", {"device": "cpu", "n_estimators": 3, "max_depth": 2})],
)
def test_real_estimators_consume_prepared_matrix_without_changing_source(
    algorithm, parameters
):
    spec = ModelSpec(
        model_id="candidate",
        model_algorithm=algorithm,
        input_factor_columns=("x", "z"),
        target_column="label",
        hyperparameters=parameters,
        random_seed=37,
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 2),
            "x": np.arange(6.0),
            "z": [2.0, 5.0, 1.0, 4.0, 9.0, 3.0],
            "label": np.sin(np.arange(6.0)),
            "eligible": True,
            "label_available_at": pd.Timestamp("2020-01-03"),
        }
    )
    source = frame.copy(deep=True)
    prepared = fitting._collect_training_frame(
        [frame], spec, fitting_cutoff="2020-01-03"
    )
    training = prepared.training.copy(deep=True)
    weights, scaler = prepared.preprocessing
    original_weights = weights.copy()
    reference = build_estimator_from_model_spec(spec)
    reference.fit(
        prepared.matrix.copy(deep=True),
        training.label.to_numpy(copy=True),
        sample_weight=weights.copy(),
    )
    actual = fitting._fit_model_result(
        prepared, spec, estimator=build_estimator_from_model_spec(spec)
    )
    expected = reference.predict(scaler.transform(frame[["x", "z"]]))
    np.testing.assert_allclose(
        actual.prediction_model.predict(frame[["x", "z"]]),
        expected,
        rtol=1e-12,
        atol=1e-14,
    )
    assert actual.metadata["training_target_mean"] == pytest.approx(
        np.average(training.label, weights=original_weights)
    )
    pd.testing.assert_frame_equal(frame, source)
    pd.testing.assert_frame_equal(prepared.training, training)
    np.testing.assert_array_equal(weights, original_weights)


def test_training_filter_handles_duplicate_dataframe_index():

    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["a", "b"],
            "x": [1.0, 2.0],
            "label": [3.0, 4.0],
            "eligible": [False, True],
            "label_available_at": pd.Timestamp("2020-01-02"),
        },
        index=[7, 7],
    )
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
    )
    prepared = fitting._collect_training_frame(
        [frame], spec, fitting_cutoff="2020-01-02"
    )
    assert prepared.training.asset_id.tolist() == ["b"]
    assert prepared.row_counts["rows_input"] == 2
    assert prepared.row_counts["rows_after_filter_mode"] == 1


@pytest.mark.parametrize(
    "column,value",
    (
        ("x", np.inf),
        ("label", np.inf),
        ("eligible", None),
        ("label_available_at", "not-a-date"),
    ),
)
def test_filtered_raw_training_rows_still_reject_invalid_values(column, value):

    frame = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": ["a", "b"],
            "x": [1.0, 2.0],
            "label": [3.0, 4.0],
            "eligible": [True, False],
            "label_available_at": pd.Timestamp("2020-01-02"),
        }
    )
    if column in {"eligible", "label_available_at"}:
        frame[column] = frame[column].astype(object)
    frame.loc[1, column] = value
    spec = ModelSpec(
        model_id="linear",
        model_algorithm="linear_regression",
        input_factor_columns=("x",),
        target_column="label",
    )
    with pytest.raises(ValueError):
        fitting._collect_training_frame([frame], spec, fitting_cutoff="2020-01-02")
