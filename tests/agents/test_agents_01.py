from __future__ import annotations

import pytest

from quant_research_platform.agents.dataset_planner import plan_research_dataset
from quant_research_platform.agents.factor_factory import propose_factor
from quant_research_platform.agents.label_factory import propose_label
from quant_research_platform.agents.proposals import (
    FactorProposal,
    LabelProposal,
    ResearchDatasetProposal,
    WorkflowProposal,
)
from quant_research_platform.agents.safety import (
    ProposalSafetyError,
    assert_proposal_is_public_safe,
    require_human_approval,
)
from quant_research_platform.agents.workflow_planner import plan_workflow


def test_agent_proposals_start_unapproved() -> None:
    proposals = [
        FactorProposal(
            proposal_id="factor-proposal-001",
            factor_id="price_volume_reversal",
            rationale="Mean reversion candidate from public field ids.",
            input_columns=("close", "volume"),
        ),
        LabelProposal(
            proposal_id="label-proposal-001",
            label_id="forward_return_20d",
            rationale="Forward return label for offline research review.",
            horizon=20,
        ),
        ResearchDatasetProposal(
            proposal_id="dataset-proposal-001",
            dataset_id="a_share_daily_research_dataset",
            processed_data_ids=("a_share_daily_0_1",),
        ),
        WorkflowProposal(
            proposal_id="workflow-proposal-001",
            workflow_id="factor_review_workflow",
            step_ids=("build_dataset_spec", "review_factor_spec"),
        ),
    ]

    assert all(not proposal.approved for proposal in proposals)
    for proposal in proposals:
        with pytest.raises(PermissionError, match="proposal_requires_human_approval"):
            require_human_approval(proposal)


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "private path data/raw/csmar/TRD_Dalyr.xlsx",
        "row sample: market_id=a_share asset_id=000001 timestamp=2024-01-02",
        "stock code 600519 has a high score",
        "source marker from CSMAR export table",
    ],
)
def test_public_safety_rejects_private_text_markers(unsafe_text: str) -> None:
    proposal = FactorProposal(
        proposal_id="factor-proposal-unsafe",
        factor_id="unsafe_factor",
        rationale=unsafe_text,
        input_columns=("close",),
    )

    with pytest.raises(ProposalSafetyError):
        assert_proposal_is_public_safe(proposal)


def test_public_safety_rejects_preapproved_proposals() -> None:
    proposal = WorkflowProposal(
        proposal_id="workflow-proposal-preapproved",
        workflow_id="unsafe_workflow",
        step_ids=("dataset_spec",),
        approved=True,
    )

    with pytest.raises(ProposalSafetyError, match="proposal_must_start_unapproved"):
        assert_proposal_is_public_safe(proposal)


def test_factories_return_only_public_safe_unapproved_proposals() -> None:
    factor = propose_factor(
        proposal_id="factor-proposal-002",
        factor_id="turnover_reversal",
        rationale="Candidate based on portable close and turnover fields.",
        input_columns=("close", "turnover"),
    )
    label = propose_label(
        proposal_id="label-proposal-002",
        label_id="forward_return_5d",
        rationale="Five session forward return target for review.",
        horizon=5,
    )
    dataset = plan_research_dataset(
        proposal_id="dataset-proposal-002",
        dataset_id="daily_factor_dataset",
        processed_data_ids=("a_share_daily_0_1",),
    )
    workflow = plan_workflow(
        proposal_id="workflow-proposal-002",
        workflow_id="factor_review_workflow",
        step_ids=("dataset_spec", "factor_spec", "human_review"),
    )

    assert isinstance(factor, FactorProposal)
    assert isinstance(label, LabelProposal)
    assert isinstance(dataset, ResearchDatasetProposal)
    assert isinstance(workflow, WorkflowProposal)
    assert not factor.approved
    assert not label.approved
    assert not dataset.approved
    assert not workflow.approved


def test_proposals_expose_spec_like_payloads_without_approval_state() -> None:
    proposal = WorkflowProposal(
        proposal_id="workflow-proposal-003",
        workflow_id="factor_review_workflow",
        step_ids=("dataset_spec", "human_review"),
    )

    payload = proposal.to_spec_payload()

    assert payload == {
        "proposal_id": "workflow-proposal-003",
        "workflow_id": "factor_review_workflow",
        "step_ids": ("dataset_spec", "human_review"),
        "contract_version": "0.1",
    }
    assert "approved" not in payload
