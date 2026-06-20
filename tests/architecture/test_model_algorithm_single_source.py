from __future__ import annotations


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
        status = supervised.model_dependency_status(name)
        assert status["status"] in {"available", "missing"}
