from __future__ import annotations

import pytest

from quant_research_platform.catalog.run_outputs import RunOutputCatalog
from quant_research_platform.contracts.run_outputs import (
    OutputScope,
    RunOutputDefinition,
)


def test_run_output_catalog_registers_by_output_definition_id() -> None:
    catalog = RunOutputCatalog()
    definition = RunOutputDefinition(
        output_definition_id="prediction_outputs",
        display_name="Prediction outputs",
        run_type="prediction",
        scope=OutputScope.LOCAL_ONLY,
    )

    catalog.register(definition)

    assert catalog.get("prediction_outputs") == definition
    assert catalog.list() == (definition,)


def test_run_output_catalog_rejects_duplicate_definition_id() -> None:
    catalog = RunOutputCatalog()
    definition = RunOutputDefinition(
        output_definition_id="evaluation_outputs",
        display_name="Evaluation outputs",
        run_type="evaluation",
        scope=OutputScope.LOCAL_ONLY,
    )
    catalog.register(definition)

    with pytest.raises(ValueError, match="duplicate_output_definition"):
        catalog.register(definition)
