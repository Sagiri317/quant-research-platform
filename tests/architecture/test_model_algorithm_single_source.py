from __future__ import annotations

import inspect
from pathlib import Path

PROJECT_SRC = Path("src/quant_research_platform")
MODELS_ROOT = PROJECT_SRC / "research" / "models"


def test_model_algorithm_ids_live_only_in_supervised_model_facts() -> None:
    import quant_research_platform.research.models.supervised as supervised

    names = supervised.model_algorithm_ids()
    assert names
    assert names == tuple(sorted(names))
    assert len(names) == len(set(names))
    for name in names:
        definition = supervised.model_algorithm_definition(name)
        assert definition.algorithm == name
        assert isinstance(supervised.model_default_hyperparameters(name), dict)
        assert isinstance(supervised.model_smoke_hyperparameters(name), dict)
        assert callable(definition.estimator_factory)
        assert definition.input_kind in {
            "feature_inputs",
            "single_factor_input",
            "rank_ensemble_inputs",
        }
        status = supervised.model_dependency_status(name)
        assert status["status"] in {"available", "missing"}


def test_supervised_model_root_delegates_prediction_and_rank_ensemble_owners() -> None:
    import quant_research_platform.research.models.supervised as supervised

    root_source = Path(supervised.__file__).read_text(encoding="utf-8")
    missing_modules = [
        path.as_posix()
        for path in (
            MODELS_ROOT / "prediction.py",
            MODELS_ROOT / "rank_ensemble.py",
        )
        if not path.is_file()
    ]
    root_owned_names = (
        "class _RankEnsemblePredictor",
        "class _ICWeightedRankEnsembleFactory",
        "class _ICWeightedRankEnsemble",
        "def _rank_ensemble_scores",
        "def _rank_frame_columns",
        "def _rank_series",
        "def _predict_scores",
        "def _prediction_model_input_frame",
    )

    assert missing_modules == []
    assert not [name for name in root_owned_names if name in root_source]


def test_supervised_model_root_delegates_validation_split_owner() -> None:
    import quant_research_platform.research.models.supervised as supervised

    root_source = Path(supervised.__file__).read_text(encoding="utf-8")
    owner_path = MODELS_ROOT / "validation_split.py"
    owner_source = owner_path.read_text(encoding="utf-8")

    assert owner_path.is_file()
    assert "def apply_training_controls" in owner_source
    assert "def trim_training_buffer" in owner_source
    assert "def _apply_training_controls" not in root_source
    assert "def _trim_training_buffer" not in root_source


def test_supervised_model_root_delegates_estimator_owner() -> None:
    import quant_research_platform.research.models.supervised as supervised

    root_source = Path(supervised.__file__).read_text(encoding="utf-8")
    owner_path = MODELS_ROOT / "estimators.py"
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_names = (
        "class _LinearRegressionEstimator",
        "class _SingleFactorBaseline",
        "def _linear_regression_estimator",
        "def _ridge_regression_estimator",
        "def _lasso_regression_estimator",
        "def _elastic_net_estimator",
        "def _huber_regression_estimator",
        "def _decision_tree_regressor",
        "def _random_forest_regressor",
        "def _extra_trees_regressor",
        "def _gradient_boosting_regressor",
        "def _hist_gradient_boosting_regressor",
        "def _xgboost_regressor",
        "def _lightgbm_regressor",
        "def _single_factor_baseline",
        "def _equal_weight_rank_ensemble",
        "def _ic_weighted_rank_ensemble",
    )

    assert owner_path.is_file()
    assert [name for name in owner_names if name not in owner_source] == []
    assert [name for name in owner_names if name in root_source] == []


def test_model_estimator_dispatch_uses_algorithm_facts_only() -> None:
    import quant_research_platform.research.models.supervised as supervised

    source = inspect.getsource(supervised)

    assert "def _build_regression_estimator" not in source
    assert "if spec.model_algorithm ==" not in inspect.getsource(
        supervised.build_estimator_from_model_spec
    )
    assert 'spec.model_algorithm != "linear_regression"' not in inspect.getsource(
        supervised._fit_model_result_from_frames
    )
    assert "_LinearRegressionEstimator" not in inspect.getsource(
        supervised._fit_model_result_from_frames
    )
    assert "isinstance(estimator," not in inspect.getsource(
        supervised._fit_model_result_from_frames
    )
    assert "_ICWeightedRankEnsembleFactory" not in inspect.getsource(
        supervised._fit_predictor
    )


def test_model_algorithm_ids_is_the_only_collection_helper() -> None:
    import quant_research_platform.contracts.models as contracts_models
    import quant_research_platform.research.models.supervised as supervised

    assert not hasattr(contracts_models, "MODEL_ALGORITHMS")
    assert not hasattr(supervised, "MODEL_ALGORITHMS")
    assert not hasattr(supervised, "model_algorithm_names")
    assert supervised.model_algorithm_ids()


def test_reports_do_not_branch_on_one_model_algorithm() -> None:
    from quant_research_platform.artifacts import outputs

    source = inspect.getsource(outputs)
    assert 'model_algorithm"] == "linear_regression"' not in source
    assert 'model_algorithm == "linear_regression"' not in source


def test_a_share_model_scenario_inputs_use_algorithm_facts_only() -> None:
    from quant_research_platform.markets.implementations.a_share import (
        research_scenarios as scenarios,
    )

    assert 'model_algorithm == "single_factor_baseline"' not in inspect.getsource(
        scenarios
    )
    assert "model_algorithm_definition" in inspect.getsource(
        scenarios._model_factor_inputs
    )
    assert "model_algorithm_definition" in inspect.getsource(
        scenarios._model_feature_inputs
    )
