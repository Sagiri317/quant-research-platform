"""Agent-assisted research dataset planning helpers."""

from quant_research_platform.agents.proposals import ResearchDatasetProposal
from quant_research_platform.agents.safety import assert_proposal_is_share_safe


def plan_research_dataset(
    *,
    proposal_id: str,
    data_id: str,
    input_data_ids: tuple[str, ...],
) -> ResearchDatasetProposal:
    if not isinstance(input_data_ids, tuple) or any(
        not isinstance(data_id, str) or not data_id for data_id in input_data_ids
    ):
        raise ValueError("invalid_proposal_tuple_field: input_data_ids")
    proposal = ResearchDatasetProposal(
        proposal_id=proposal_id,
        data_id=data_id,
        input_data_ids=input_data_ids,
    )
    assert_proposal_is_share_safe(proposal)
    return proposal


__all__ = ["ResearchDatasetProposal", "plan_research_dataset"]
