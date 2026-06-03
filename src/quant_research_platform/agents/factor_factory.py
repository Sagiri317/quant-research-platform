"""Proposal-only Factor Factory boundary."""

from quant_research_platform.agents.proposals import FactorProposal
from quant_research_platform.agents.safety import assert_proposal_is_public_safe
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_field_catalog,
)


def propose_factor(
    *,
    proposal_id: str,
    factor_id: str,
    rationale: str,
    input_columns: tuple[str, ...],
) -> FactorProposal:
    _validate_input_columns(tuple(input_columns))
    proposal = FactorProposal(
        proposal_id=proposal_id,
        factor_id=factor_id,
        rationale=rationale,
        input_columns=tuple(input_columns),
    )
    assert_proposal_is_public_safe(proposal)
    return proposal


def _validate_input_columns(input_columns: tuple[str, ...]) -> None:
    fields = a_share_field_catalog()
    unknown = sorted(set(input_columns).difference(fields.list_names()))
    if unknown:
        raise ValueError(f"unknown_factor_input_columns: {', '.join(unknown)}")


__all__ = ["FactorProposal", "propose_factor"]
