"""Constructor defaults and dependency failures without importing unused backends."""

from dataclasses import replace
from types import SimpleNamespace

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
        ("ridge_regression", "Ridge"),
        ("lasso_regression", "Lasso"),
        ("huber_regression", "HuberRegressor"),
        ("decision_tree_regressor", "DecisionTreeRegressor"),
        ("random_forest_regressor", "RandomForestRegressor"),
        ("gradient_boosting_regressor", "GradientBoostingRegressor"),
        ("xgboost_regressor", "XGBRegressor"),
        ("lightgbm_regressor", "LGBMRegressor"),
    ],
)
def test_native_constructor_class_and_seed_defaults(algorithm, class_name):
    spec = _spec(algorithm)
    estimator = estimators.build_estimator_from_model_spec(spec)
    assert type(estimator).__name__ == class_name
    parameters = estimator.get_params(deep=False)
    if algorithm == "huber_regression":
        assert "random_state" not in parameters
    else:
        assert parameters["random_state"] == 731
        explicit = replace(spec, random_seed=19, hyperparameters={"random_state": 19})
        assert estimators.build_estimator_from_model_spec(explicit).random_state == 19
        assert dict(explicit.hyperparameters) == {"random_state": 19}
    assert dict(spec.hyperparameters) == {}


def test_internal_models_need_no_backend_and_keep_rank_column_order(monkeypatch):
    def unexpected_import(name):
        pytest.fail(f"internal model imported backend: {name}")

    monkeypatch.setattr(estimators, "import_module", unexpected_import)
    assert (
        estimators.build_estimator_from_model_spec(_spec("linear_regression")) is None
    )
    for algorithm in ("equal_weight_rank_ensemble", "ic_weighted_rank_ensemble"):
        estimator = estimators.build_estimator_from_model_spec(_spec(algorithm))
        assert estimator.columns == ("z", "x")
        assert estimator.label_column == "label"


@pytest.mark.parametrize(
    "algorithm,package",
    [
        ("ridge_regression", "sklearn"),
        ("xgboost_regressor", "xgboost"),
        ("lightgbm_regressor", "lightgbm"),
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
    monkeypatch.setattr(estimators, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(RuntimeError, match="^model_dependency_version_unavailable$"):
        estimators.build_estimator_from_model_spec(_spec(algorithm))
