"""Proposal-only Factor Factory boundary."""

from quant_research_platform.agents.proposals import FactorProposal
from quant_research_platform.agents.safety import assert_proposal_is_public_safe
from quant_research_platform.catalog.defaults import default_field_catalog


def propose_factor(
    *,
    proposal_id: str,
    factor_id: str,
    rationale: str,
    input_columns: tuple[str, ...],
    output_columns: tuple[str, ...] | None = None,
    base_input_columns: tuple[str, ...] | None = None,
    derived_input_columns: tuple[str, ...] = (),
    derived_input_producers: dict[str, str] | None = None,
    required_input_data_id: str = "daily_panel",
) -> FactorProposal:
    _validate_input_columns(tuple(input_columns))
    proposal = FactorProposal(
        proposal_id=proposal_id,
        factor_id=factor_id,
        rationale=rationale,
        input_columns=tuple(input_columns),
        output_columns=tuple(output_columns or (factor_id,)),
        base_input_columns=tuple(base_input_columns or input_columns),
        derived_input_columns=tuple(derived_input_columns),
        derived_input_producers=dict(derived_input_producers or {}),
        required_input_data_id=required_input_data_id,
    )
    assert_proposal_is_public_safe(proposal)
    return proposal


def _validate_input_columns(input_columns: tuple[str, ...]) -> None:
    fields = default_field_catalog()
    unknown = sorted(set(input_columns).difference(fields.list_names()))
    if unknown:
        raise ValueError(f"unknown_factor_input_columns: {', '.join(unknown)}")


__all__ = ["FactorProposal", "propose_factor"]
