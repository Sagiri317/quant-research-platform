"""Proposal-only research dataset planner boundary."""

from quant_research_platform.agents.proposals import ResearchDatasetProposal
from quant_research_platform.agents.safety import assert_proposal_is_share_safe


def plan_research_dataset(
    *,
    proposal_id: str,
    data_id: str,
    input_data_ids: tuple[str, ...],
) -> ResearchDatasetProposal:
    proposal = ResearchDatasetProposal(
        proposal_id=proposal_id,
        data_id=data_id,
        input_data_ids=tuple(input_data_ids),
    )
    assert_proposal_is_share_safe(proposal)
    return proposal


__all__ = ["ResearchDatasetProposal", "plan_research_dataset"]
