from __future__ import annotations

import pytest

from quant_research_platform.contracts.outputs import OutputSpec


@pytest.mark.parametrize("kind", ["prediction_outputs", "evaluation_outputs"])
def test_output_spec_separates_instance_and_kind(kind):
    spec = OutputSpec("provider_summary", kind, "run_000001", "a_share")
    assert spec.output_id == "provider_summary"
    assert spec.output_definition_id == kind
    with pytest.raises(ValueError, match="output_id_must_identify_instance"):
        OutputSpec(kind, kind, "run_000001", "a_share")


@pytest.mark.parametrize("kind", ["../bad", "Upper", "with.dot", "con", "custom"])
def test_output_spec_rejects_unknown_kind(kind):
    with pytest.raises(ValueError, match="invalid_output_definition_id"):
        OutputSpec("provider_summary", kind, "run_000001", "a_share")
