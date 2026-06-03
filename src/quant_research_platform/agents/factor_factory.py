"""Proposal-only Factor Factory boundary."""

from quant_research_platform.agents.proposals import FactorProposal
from quant_research_platform.agents.safety import assert_proposal_is_public_safe


def propose_factor(
    *,
    proposal_id: str,
    factor_id: str,
    rationale: str,
    input_columns: tuple[str, ...],
) -> FactorProposal:
    proposal = FactorProposal(
        proposal_id=proposal_id,
        factor_id=factor_id,
        rationale=rationale,
        input_columns=tuple(input_columns),
    )
    assert_proposal_is_public_safe(proposal)
    return proposal


__all__ = ["FactorProposal", "propose_factor"]
