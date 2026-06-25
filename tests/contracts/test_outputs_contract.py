from __future__ import annotations

import pytest

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.outputs import (
    OutputDefinition,
    OutputScope,
    OutputSpec,
)


def test_output_scope_is_local_or_review_safe_only() -> None:
    assert tuple(scope.value for scope in OutputScope) == (
        "LOCAL_ONLY",
        "REVIEW_SAFE",
    )


def test_output_spec_separates_instance_and_definition_identity() -> None:
    spec = OutputSpec(
        output_id="prediction_daily",
        output_definition_id="prediction_outputs",
        run_id="research_run_2026",
        market_id=MarketId("a_share"),
        run_type="prediction",
        scope=OutputScope.LOCAL_ONLY,
    )

    assert spec.output_id == "prediction_daily"
    assert spec.output_definition_id == "prediction_outputs"
    assert spec.run_id == "research_run_2026"
    assert spec.scope is OutputScope.LOCAL_ONLY


def test_output_definition_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError):
        OutputDefinition(
            output_definition_id="x",
            display_name="X",
            run_type="diagnostic",
            scope="UNKNOWN",
        )
