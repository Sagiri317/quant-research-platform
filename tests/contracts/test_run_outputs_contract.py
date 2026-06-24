from __future__ import annotations

import pytest

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.run_outputs import (
    OutputScope,
    RunOutputDefinition,
    RunOutputSpec,
)


def test_run_output_scope_is_local_or_review_safe_only() -> None:
    assert tuple(scope.value for scope in OutputScope) == (
        "LOCAL_ONLY",
        "REVIEW_SAFE",
    )


def test_run_output_spec_uses_output_ids_and_scope() -> None:
    spec = RunOutputSpec(
        output_id="prediction_outputs",
        output_definition_id="prediction_outputs",
        run_id="prediction_outputs_2026",
        market_id=MarketId("a_share"),
        run_type="prediction",
        scope=OutputScope.LOCAL_ONLY,
    )

    assert spec.output_id == "prediction_outputs"
    assert spec.output_definition_id == "prediction_outputs"
    assert spec.scope is OutputScope.LOCAL_ONLY


def test_run_output_definition_rejects_unknown_contract_version() -> None:
    with pytest.raises(ValueError, match="unsupported_run_output_contract_version"):
        RunOutputDefinition(
            output_definition_id="x",
            display_name="X",
            run_type="diagnostic",
            scope=OutputScope.LOCAL_ONLY,
            contract_version="9",
        )
