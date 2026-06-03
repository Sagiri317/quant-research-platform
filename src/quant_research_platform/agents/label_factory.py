"""Proposal-only label factory boundary."""

from quant_research_platform.agents.proposals import LabelProposal
from quant_research_platform.agents.safety import assert_proposal_is_public_safe


def propose_label(
    *,
    proposal_id: str,
    label_id: str,
    rationale: str,
    horizon: int,
) -> LabelProposal:
    proposal = LabelProposal(
        proposal_id=proposal_id,
        label_id=label_id,
        rationale=rationale,
        horizon=horizon,
    )
    assert_proposal_is_public_safe(proposal)
    return proposal


__all__ = ["LabelProposal", "propose_label"]
