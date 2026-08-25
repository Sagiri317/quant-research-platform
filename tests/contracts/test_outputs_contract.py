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


def test_output_contract_preserves_provider_like_identifiers() -> None:
    definition = OutputDefinition(
        output_definition_id="provider_csmar_daily",
        display_name="Provider output",
        run_type="diagnostic",
        scope=OutputScope.LOCAL_ONLY,
    )
    spec = OutputSpec(
        output_id="provider_summary",
        output_definition_id=definition.output_definition_id,
        run_id="run_000001",
        market_id=MarketId("a_share"),
        run_type="diagnostic",
        scope=OutputScope.LOCAL_ONLY,
    )

    assert definition.output_definition_id == "provider_csmar_daily"
    assert spec.output_id == "provider_summary"


@pytest.mark.parametrize("identifier", ("../bad", "Upper", "with.dot", "con"))
def test_output_definition_rejects_invalid_identifiers(identifier: str) -> None:
    with pytest.raises(ValueError, match="invalid_output_id"):
        OutputDefinition(
            output_definition_id=identifier,
            display_name="Invalid",
            run_type="diagnostic",
            scope=OutputScope.LOCAL_ONLY,
        )


@pytest.mark.parametrize("identifier", ("../bad", "Upper", "with.dot", "con"))
def test_output_spec_rejects_invalid_definition_identifiers(identifier: str) -> None:
    with pytest.raises(ValueError, match="invalid_output_id"):
        OutputSpec(
            output_id="provider_summary",
            output_definition_id=identifier,
            run_id="run_000001",
            market_id=MarketId("a_share"),
            run_type="diagnostic",
            scope=OutputScope.LOCAL_ONLY,
        )


def test_output_definition_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError):
        OutputDefinition(
            output_definition_id="x",
            display_name="X",
            run_type="diagnostic",
            scope="UNKNOWN",
        )
