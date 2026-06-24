from __future__ import annotations

import inspect


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
    from quant_research_platform.artifacts import run_outputs

    source = inspect.getsource(run_outputs)
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
