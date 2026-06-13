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
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.profiles import research_profile
from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.markets.implementations.a_share.fields import (
    a_share_field_catalog,
)


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
            data_id="a_share_daily_research_dataset",
            input_data_ids=("a_share_daily_0_1",),
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
        profile=research_profile(),
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
        data_id="daily_factor_dataset",
        input_data_ids=("a_share_daily_0_1",),
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


def test_factor_factory_rejects_unknown_input_columns() -> None:
    with pytest.raises(ValueError, match="unknown_factor_input_columns"):
        propose_factor(
            profile=research_profile(),
            proposal_id="factor-proposal-unknown-input",
            factor_id="bad_input_factor",
            rationale="Candidate with unregistered input.",
            input_columns=("unknown_input_column",),
        )


def test_unregistered_factor_proposal_has_no_execution_or_persistence_surface() -> None:
    proposal = FactorProposal(
        proposal_id="factor-proposal-unregistered",
        factor_id="candidate_factor",
        rationale="Candidate for catalog review only.",
        input_columns=("close",),
        output_columns=("candidate_factor",),
        required_input_data_id="daily_panel",
    )

    assert not hasattr(proposal, "execute")
    assert not hasattr(proposal, "persist")
    assert not hasattr(proposal, "to_workflow_node")


def test_factor_catalog_registers_valid_factor_proposal() -> None:
    proposal = propose_factor(
        profile=research_profile(),
        proposal_id="factor-proposal-register",
        factor_id="candidate_factor",
        rationale="Candidate based on registered close field.",
        input_columns=("close",),
        output_columns=("candidate_factor",),
        required_input_data_id="daily_panel",
    )
    catalog = FactorCatalog()

    definition = catalog.register_proposal(
        proposal,
        market_id=MarketId("a_share"),
        field_names=set(a_share_field_catalog().list_names()),
    )

    assert isinstance(definition, FactorDefinition)
    assert definition.factor_id == "candidate_factor"
    assert definition.output_column == "candidate_factor"
    assert catalog.get("candidate_factor") == definition


def test_factor_proposal_registration_rejects_duplicate_output_columns() -> None:
    proposal = FactorProposal(
        proposal_id="factor-proposal-duplicate-output",
        factor_id="candidate_factor",
        rationale="Invalid duplicate output candidate.",
        input_columns=("close",),
        output_columns=("candidate_factor", "candidate_factor"),
        required_input_data_id="daily_panel",
    )

    with pytest.raises(ValueError, match="duplicate_factor_output_column"):
        FactorCatalog().register_proposal(
            proposal,
            market_id=MarketId("a_share"),
            field_names=set(a_share_field_catalog().list_names()),
        )


def test_factor_proposal_registration_rejects_unresolved_derived_input() -> None:
    proposal = FactorProposal(
        proposal_id="factor-proposal-unresolved-derived",
        factor_id="candidate_factor",
        rationale="Invalid derived input candidate.",
        input_columns=("close", "candidate_derived"),
        base_input_columns=("close",),
        derived_input_columns=("candidate_derived",),
        output_columns=("candidate_factor",),
        required_input_data_id="daily_panel",
    )

    with pytest.raises(ValueError, match="unresolved_factor_derived_input"):
        FactorCatalog().register_proposal(
            proposal,
            market_id=MarketId("a_share"),
            field_names=set(a_share_field_catalog().list_names()),
        )


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
