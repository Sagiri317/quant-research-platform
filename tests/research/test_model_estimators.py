"""Constructor defaults and dependency failures without importing unused backends."""

from dataclasses import replace

import pytest

from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.research.models import estimators


def _spec(algorithm):
    return ModelSpec(
        model_id="constructor",
        model_algorithm=algorithm,
        input_factor_columns=("z",),
        input_feature_columns=("x",),
        target_column="label",
        random_seed=731,
    )


@pytest.mark.parametrize(
    "algorithm,class_name",
    [
        ("lasso_regression", "Lasso"),
        ("ridge_regression", "Ridge"),
        ("xgboost_regressor", "XGBRegressor"),
    ],
)
def test_native_constructor_class_and_seed_defaults(algorithm, class_name):
    spec = _spec(algorithm)
    estimator = estimators.build_estimator_from_model_spec(spec)
    assert type(estimator).__name__ == class_name
    parameters = estimator.get_params(deep=False)
    assert parameters["random_state"] == 731
    explicit = replace(spec, random_seed=19, hyperparameters={"random_state": 19})
    assert estimators.build_estimator_from_model_spec(explicit).random_state == 19
    assert dict(explicit.hyperparameters) == {"random_state": 19}
    assert dict(spec.hyperparameters) == {}


def test_ols_needs_no_backend(monkeypatch):
    def unexpected_import(name):
        pytest.fail(f"internal model imported backend: {name}")

    monkeypatch.setattr(estimators, "import_module", unexpected_import)
    assert (
        estimators.build_estimator_from_model_spec(_spec("linear_regression")) is None
    )


@pytest.mark.parametrize(
    "algorithm,package",
    [
        ("xgboost_regressor", "xgboost"),
    ],
)
def test_only_selected_dependency_is_required(monkeypatch, algorithm, package):
    calls = []

    def missing(name):
        calls.append(name)
        raise ImportError("synthetic missing backend")

    monkeypatch.setattr(estimators, "import_module", missing)
    with pytest.raises(ImportError, match=f"^model_dependency_unavailable: {package}$"):
        estimators.build_estimator_from_model_spec(_spec(algorithm))
    assert calls == [package]
