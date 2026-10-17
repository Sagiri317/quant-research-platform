"""Independent estimator-input and persistence checks for model-stage scaling."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.models import ModelSpec, model_spec_payload
from quant_research_platform.research.models import estimators
from quant_research_platform.research.models.explanations import _native_importance
from quant_research_platform.research.models.fitting import _collect_training_frame
from quant_research_platform.research.models.scaler import ModelMatrixScaler
from tests.research.direct_model_test_support import fit_frame

ALGORITHMS = (
    "linear_regression",
    "ridge_regression",
    "lasso_regression",
    "huber_regression",
    "decision_tree_regressor",
    "random_forest_regressor",
    "gradient_boosting_regressor",
    "xgboost_regressor",
    "lightgbm_regressor",
)


def _frame():
    dates = pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 2)
    return pd.DataFrame(
        dict(
            timestamp=dates,
            asset_id=list("abcab"),
            x=[10.0, 20.0, np.nan, 31.0, 50.0],
            z=[2.0, np.nan, 6.0, 8.0, 12.0],
            label=[1.0, 2.0, 1.5, 3.0, 5.0],
            eligible=True,
            label_available_at=dates,
        )
    )


def _spec(algorithm, choice=None):
    parameters = {}
    if algorithm == "huber_regression":
        parameters["max_iter"] = 1000
    if algorithm in ALGORITHMS[4:]:
        parameters["max_depth"] = 2
    if algorithm in ALGORITHMS[5:]:
        parameters["n_estimators"] = 4
    if algorithm in {
        "random_forest_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
    }:
        parameters["n_jobs"] = 1
    if algorithm == "lightgbm_regressor":
        parameters.update(min_child_samples=1, verbosity=-1)
    return ModelSpec(
        model_id=algorithm,
        model_algorithm=algorithm,
        input_factor_columns=("x", "z"),
        target_column="label",
        hyperparameters=parameters,
        **({} if choice is None else {"training_scaling": choice}),
    )


def _expected(data, native, scaling):
    matrix = data[["x", "z"]].astype("float64")
    weights = np.array([1 / 3] * 3 + [1 / 2] * 2)
    if not native:
        matrix = matrix.fillna(matrix.median())
        for column in ("x", "z"):
            matrix[column + "__missing"] = data[column].isna().astype(float)
    if scaling:
        for column in matrix:
            observed = matrix[column].notna()
            values = matrix.loc[observed, column].to_numpy()
            w = weights[observed]
            center = np.average(values, weights=w)
            std = np.sqrt(np.average((values - center) ** 2, weights=w))
            matrix.loc[observed, column] = (values - center) / (std or 1.0)
    return matrix


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("choice", [None, False, True])
def test_actual_estimator_inputs_and_native_roundtrip(
    tmp_path, monkeypatch, algorithm, choice
):
    spec = _spec(algorithm, choice)
    data = _frame().astype({"x": "Int64", "z": "Float64"})
    original_data = data.copy(deep=True)
    expected_scaling = algorithm in ALGORITHMS[:4] if choice is None else choice
    native = algorithm in {
        "decision_tree_regressor",
        "random_forest_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
    }
    expected = _expected(data, native, expected_scaling)
    captured = []
    if algorithm == "linear_regression":
        original = np.linalg.lstsq

        def spy(design, target, **kwargs):
            captured.append(design.copy())
            return original(design, target, **kwargs)

        monkeypatch.setattr(np.linalg, "lstsq", spy)
    else:
        factory = estimators._MODEL_ESTIMATOR_FACTORIES[algorithm]
        cls = type(factory(spec, dict(spec.hyperparameters)))
        original = cls.fit

        def spy(self, matrix, target, **kwargs):
            captured.append(matrix.copy())
            return original(self, matrix, target, **kwargs)

        monkeypatch.setattr(cls, "fit", spy)
    case = fit_frame(tmp_path / "fit", data, spec)
    assert len(captured) == 1
    actual = captured[0]
    if algorithm == "linear_regression":
        actual = actual[:, 1:] / actual[:, :1]
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13, equal_nan=True)
    assert spec.training_scaling is expected_scaling
    assert model_spec_payload(spec)["training_scaling"] is expected_scaling
    assert case.facts["model"]["training_scaling"] is expected_scaling
    assert case.result.metadata["training_scaling"] is expected_scaling
    scaler = case.result.prediction_model.scaler
    assert scaler.scaling is expected_scaling
    assert scaler.native_missing is native
    np.testing.assert_allclose(
        scaler.transform(data[["x", "z"]]),
        expected,
        rtol=2e-13,
        atol=2e-13,
        equal_nan=True,
    )
    pd.testing.assert_frame_equal(data, original_data, check_exact=True)
    if not native:
        assert scaler.medians == (25.5, 7.0)
        assert scaler.columns == ("x", "z", "x__missing", "z__missing")
    frozen = scaler.to_payload()
    future = data[["x", "z"]].copy()
    future.loc[0] = [np.nan, np.nan]
    future.loc[1] = [10000.0, np.nan]
    original_future = future.copy(deep=True)
    prediction = case.result.prediction_model.predict(future)
    assert np.isnan(prediction[0])
    assert np.isfinite(prediction[1:]).all()
    assert scaler.to_payload() == frozen
    pd.testing.assert_frame_equal(future, original_future, check_exact=True)


def test_scaling_independent_of_missingness_and_all_missing_columns():
    data = _frame().assign(z=np.nan)
    for scaling in (False, True):
        native = _collect_training_frame(
            [data],
            _spec("decision_tree_regressor", scaling),
            fitting_cutoff="2024-01-03",
        )
        matrix = native.preprocessing[2]
        assert matrix.z.isna().all()
        assert matrix.columns.tolist() == ["x", "z"]
        with pytest.raises(ValueError, match="model_training_feature_all_missing"):
            _ = _collect_training_frame(
                [data],
                _spec("gradient_boosting_regressor", scaling),
                fitting_cutoff="2024-01-03",
            ).preprocessing


@pytest.mark.parametrize("value", [0, 1, "true", []])
def test_scaling_flag_is_strict_boolean(value):
    with pytest.raises((ValueError, TypeError), match="training_scaling"):
        replace(_spec("ridge_regression"), training_scaling=value)


def test_rank_keeps_own_rules():
    spec = ModelSpec(
        model_id="rank",
        model_algorithm="equal_weight_rank_ensemble",
        input_factor_columns=("x",),
        target_column="label",
    )
    assert spec.training_scaling is False
    with pytest.raises(ValueError, match="training_scaling"):
        replace(spec, training_scaling=True)


@pytest.mark.parametrize("algorithm", ALGORITHMS[:4])
def test_unscaled_coefficients_do_not_sum_mixed_units(tmp_path, algorithm):
    case = fit_frame(tmp_path / "fit", _frame(), _spec(algorithm, False))
    predictor = case.result.prediction_model
    coefficients = np.asarray(predictor.coef_)
    np.testing.assert_allclose(
        list(case.result.metadata["feature_importance"].values()),
        np.abs(coefficients[:2]),
    )
    facts = dict(case.facts, input_units={"x": "CNY", "z": "shares"})
    table = _native_importance(case.result, facts)
    assert table.kind.str.contains("raw").all()
    assert not table.kind.str.contains("standardized|sum").any()
    assert table.unit.tolist() == ["decimal_return/CNY", "decimal_return/shares"]
    np.testing.assert_allclose(table.signed_value, coefficients[:2])

    from quant_research_platform.artifacts.explanations import read_explanation
    from quant_research_platform.research.models.explanations import explain

    future = _frame().assign(timestamp=pd.Timestamp("2024-01-04"))
    future["asset_id"] = [str(index) for index in range(len(future))]
    source = tmp_path / "future.parquet"
    future.to_parquet(source, index=False)
    metadata, tables = read_explanation(
        explain(
            {
                "model": str(case.root),
                "inputs": str(source),
                "input_spec": case.config["input_spec"],
                "output": str(tmp_path / "explanation"),
                "methods": ["native_importance", "affine"],
                "window": {"start": "2024-01-04", "end": "2024-01-04"},
                "sample_role": "validation",
                "selection": {"max_rows": 5, "max_dates": 1},
                "output_space": "model_output",
            }
        )
    )
    assert metadata["attribution"]["baseline"] == "unscaled_input_zero"
    assert tables["native_importance"].kind.str.contains("raw").all()
    assert tables["predictions"].status.eq("DEFINED").all()


@pytest.mark.parametrize("algorithm", ALGORITHMS[4:6] + ALGORITHMS[7:])
@pytest.mark.parametrize("scaling", [False, True])
def test_native_all_missing_column_scaling_roundtrip(tmp_path, algorithm, scaling):
    data = _frame().assign(z=np.nan)
    case = fit_frame(tmp_path / "fit", data, _spec(algorithm, scaling))
    scaler = case.result.prediction_model.scaler
    assert scaler.scaling is scaling
    assert scaler.native_missing
    assert scaler.medians == ()
    assert scaler.transform(data[["x", "z"]]).z.isna().all()
    assert scaler.origin_[1] == 0 and scaler.scale_[1] == 1
    assert case.result.metadata["rows_after_input_label_dropna"] == 4
    future = data[["x", "z"]].copy()
    future.loc[0] = [np.nan, np.nan]
    future.loc[1] = [np.nan, 1000.0]
    scores = case.result.prediction_model.predict(future)
    assert np.isnan(scores[[0, 2]]).all()
    assert np.isfinite(scores[[1, 3, 4]]).all()


@pytest.mark.parametrize(
    "algorithm", ["ridge_regression", "gradient_boosting_regressor"]
)
@pytest.mark.parametrize("choice", [None, False, True])
def test_cli_json_uses_same_resolved_scaling(tmp_path, algorithm, choice):
    import json

    from typer.testing import CliRunner

    from quant_research_platform.interfaces.cli.commands import app
    from quant_research_platform.research.models.persisted import load_model

    case = fit_frame(tmp_path / "api", _frame(), _spec(algorithm, choice))
    config = dict(case.config)
    config["output"] = str(tmp_path / "cli")
    if choice is None:
        config["model"].pop("training_scaling")
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(config))
    invocation = CliRunner().invoke(app, ["fit", "--config", str(path)])
    assert invocation.exit_code == 0, invocation.output
    result, facts = load_model(tmp_path / "cli")
    assert (
        result.metadata["training_scaling"] == case.result.metadata["training_scaling"]
    )
    assert (
        result.prediction_model.scaler.to_payload()
        == case.result.prediction_model.scaler.to_payload()
    )
    assert facts["model"]["training_scaling"] == case.facts["model"].get(
        "training_scaling", algorithm == "ridge_regression"
    )
    np.testing.assert_allclose(
        result.prediction_model.predict(_frame()[["x", "z"]]),
        case.result.prediction_model.predict(_frame()[["x", "z"]]),
        rtol=1e-13,
        atol=1e-13,
    )


@pytest.mark.parametrize(
    "value", [True, 1 + 2j, np.inf, "invalid", pd.Timestamp("2024-01-01")]
)
@pytest.mark.parametrize("stage", ["fit", "transform"])
def test_imputation_preserves_strict_numeric_validation(value, stage):
    valid = pd.DataFrame({"x": pd.array([1, None, 2], dtype="Int64")})
    invalid = pd.DataFrame({"x": pd.Series([1, None, value], dtype=object)})
    with pytest.raises(ValueError, match="invalid_model_scaler_column"):
        if stage == "fit":
            ModelMatrixScaler.fit(invalid, missing="impute")
        else:
            ModelMatrixScaler.fit(valid, missing="impute").transform(invalid)
