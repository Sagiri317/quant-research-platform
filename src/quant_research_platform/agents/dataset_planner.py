"""Proposal-only research dataset planner boundary."""

from quant_research_platform.agents.proposals import ResearchDatasetProposal
from quant_research_platform.agents.safety import assert_proposal_is_public_safe


def plan_research_dataset(
    *,
    proposal_id: str,
    dataset_id: str,
    processed_data_ids: tuple[str, ...],
) -> ResearchDatasetProposal:
    proposal = ResearchDatasetProposal(
        proposal_id=proposal_id,
        dataset_id=dataset_id,
        processed_data_ids=tuple(processed_data_ids),
    )
    assert_proposal_is_public_safe(proposal)
    return proposal


__all__ = ["ResearchDatasetProposal", "plan_research_dataset"]
