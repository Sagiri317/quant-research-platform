from __future__ import annotations

import inspect


def test_model_algorithms_live_only_in_supervised_model_facts() -> None:
    import quant_research_platform.contracts.models as contracts_models
    import quant_research_platform.research.models.supervised as supervised

    assert not hasattr(contracts_models, "MODEL" + "_ALGORITHMS")
    names = supervised.model_algorithm_names()
    assert names == tuple(sorted(names))
    assert "linear_regression" in names
    for name in names:
        definition = supervised.model_algorithm_definition(name)
        assert definition.algorithm == name
        assert isinstance(supervised.model_default_hyperparameters(name), dict)
        assert isinstance(supervised.model_smoke_hyperparameters(name), dict)
        assert callable(definition.estimator_factory)
        status = supervised.model_dependency_status(name)
        assert status["status"] in {"available", "missing"}


def test_model_estimator_dispatch_uses_algorithm_facts_only() -> None:
    import quant_research_platform.research.models.supervised as supervised

    source = inspect.getsource(supervised)

    assert "def _build_regression_estimator" not in source
    assert "if spec.model_algorithm ==" not in inspect.getsource(
        supervised.build_estimator_from_model_spec
    )
    assert "spec.model_algorithm != \"linear_regression\"" not in inspect.getsource(
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
