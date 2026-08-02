from __future__ import annotations

from quant_research_platform.research.models.algorithm_facts import (
    algorithm_facts,
    model_algorithm_ids,
    model_default_hyperparameters,
    model_smoke_hyperparameters,
)
from quant_research_platform.research.models.dependencies import (
    model_dependency_status,
)


def test_model_algorithm_facts_are_complete_and_ordered() -> None:
    names = model_algorithm_ids()

    assert names == tuple(sorted(set(names)))
    assert names
    for name in names:
        definition = algorithm_facts(name)
        assert definition.algorithm == name
        assert callable(definition.estimator_factory)
        assert isinstance(model_default_hyperparameters(name), dict)
        assert isinstance(model_smoke_hyperparameters(name), dict)
        assert model_dependency_status(name)["status"] in {"available", "missing"}
