"""Agent-assisted workflow planning helpers."""

from quant_research_platform.agents.proposals import WorkflowProposal
from quant_research_platform.agents.safety import assert_proposal_is_share_safe


def plan_workflow(
    *,
    proposal_id: str,
    workflow_id: str,
    step_ids: tuple[str, ...] = (),
) -> WorkflowProposal:
    proposal = WorkflowProposal(
        proposal_id=proposal_id,
        workflow_id=workflow_id,
        step_ids=_validate_step_ids(step_ids),
    )
    assert_proposal_is_share_safe(proposal)
    return proposal


def _validate_step_ids(step_ids: object) -> tuple[str, ...]:
    if not isinstance(step_ids, tuple):
        raise ValueError("invalid_proposal_tuple_field: step_ids")
    if any(not isinstance(step_id, str) or not step_id for step_id in step_ids):
        raise ValueError("invalid_proposal_tuple_field: step_ids")
    return step_ids


__all__ = ["WorkflowProposal", "plan_workflow"]
