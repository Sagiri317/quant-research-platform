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
        "class RankEnsemblePredictor",
        "class ICWeightedRankEnsembleFactory",
        "class ICWeightedRankEnsemble",
        "def _rank_ensemble_scores",
        "def _rank_frame_columns",
        "def _rank_series",
        "def _predict_scores",
        "def prediction_model_input_frame",
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
        "class SingleFactorBaseline",
        "def linear_regression_estimator",
        "def ridge_regression_estimator",
        "def lasso_regression_estimator",
        "def elastic_net_estimator",
        "def huber_regression_estimator",
        "def decision_tree_regressor",
        "def random_forest_regressor",
        "def extra_trees_regressor",
        "def gradient_boosting_regressor",
        "def hist_gradient_boosting_regressor",
        "def xgboost_regressor",
        "def lightgbm_regressor",
        "def single_factor_baseline",
        "def equal_weight_rank_ensemble",
        "def ic_weighted_rank_ensemble",
    )

    assert owner_path.is_file()
    assert [name for name in owner_names if name not in owner_source] == []
    assert [name for name in owner_names if name in root_source] == []


def test_supervised_model_root_delegates_feature_importance_owner() -> None:
    import quant_research_platform.research.models.supervised as supervised

    root_source = Path(supervised.__file__).read_text(encoding="utf-8")
    owner_path = MODELS_ROOT / "feature_importance.py"
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_names = (
        "def linear_feature_importance",
        "def estimator_feature_importance",
    )

    assert owner_path.is_file()
    assert [name for name in owner_names if name not in owner_source] == []
    assert "def _linear_feature_importance" not in root_source
    assert "def _estimator_feature_importance" not in root_source


def test_supervised_model_root_delegates_algorithm_facts_and_dependencies() -> None:
    import quant_research_platform.research.models.supervised as supervised

    root_source = Path(supervised.__file__).read_text(encoding="utf-8")
    facts_path = MODELS_ROOT / "algorithm_facts.py"
    dependencies_path = MODELS_ROOT / "dependencies.py"
    facts_source = facts_path.read_text(encoding="utf-8")
    dependencies_source = dependencies_path.read_text(encoding="utf-8")

    assert facts_path.is_file()
    assert dependencies_path.is_file()
    assert "class AlgorithmFacts" in facts_source
    assert "def algorithm_facts" in facts_source
    assert "def model_algorithm_ids" in facts_source
    assert "def model_dependency_status" in dependencies_source
    assert "def require_model_dependency" in dependencies_source
    assert "class _AlgorithmFacts" not in root_source
    assert "_MODEL_ALGORITHM_FACTS =" not in root_source
    assert "def _algorithm_facts" not in root_source
    assert "def _dependency_status" not in root_source


def test_supervised_model_root_delegates_fitting_owner() -> None:
    import quant_research_platform.research.models.supervised as supervised

    root_source = Path(supervised.__file__).read_text(encoding="utf-8")
    owner_path = MODELS_ROOT / "fitting.py"
    owner_source = owner_path.read_text(encoding="utf-8")
    owner_names = (
        "class ModelFitResult",
        "def fit_model_result_from_frames",
        "def fit_linear_model_result_from_frames",
        "def collect_training_frame",
        "def fit_predictor",
    )
    root_owned_names = (
        "class ModelFitResult",
        "def _fit_model_result_from_frames",
        "def _fit_linear_model_result_from_frames",
        "def _collect_training_frame",
        "def _fit_predictor",
    )

    assert owner_path.is_file()
    assert [name for name in owner_names if name not in owner_source] == []
    assert [name for name in root_owned_names if name in root_source] == []


def test_model_estimator_dispatch_uses_algorithm_facts_only() -> None:
    import quant_research_platform.research.models.fitting as fitting
    import quant_research_platform.research.models.supervised as supervised

    source = inspect.getsource(supervised)
    fitting_source = inspect.getsource(fitting)

    assert "def _build_regression_estimator" not in source
    assert "if spec.model_algorithm ==" not in inspect.getsource(
        supervised.build_estimator_from_model_spec
    )
    assert 'spec.model_algorithm != "linear_regression"' not in fitting_source
    assert "_LinearRegressionEstimator" not in fitting_source
    assert "isinstance(estimator," not in fitting_source
    assert "ICWeightedRankEnsembleFactory" not in fitting_source


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
