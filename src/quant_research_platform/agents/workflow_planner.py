"""Proposal-only workflow planner boundary."""

from quant_research_platform.agents.proposals import WorkflowProposal
from quant_research_platform.agents.safety import assert_proposal_is_public_safe


def plan_workflow(
    *,
    proposal_id: str,
    workflow_id: str,
    step_ids: tuple[str, ...] = (),
) -> WorkflowProposal:
    proposal = WorkflowProposal(
        proposal_id=proposal_id,
        workflow_id=workflow_id,
        step_ids=tuple(step_ids),
    )
    assert_proposal_is_public_safe(proposal)
    return proposal


__all__ = ["WorkflowProposal", "plan_workflow"]
