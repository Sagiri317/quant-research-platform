from __future__ import annotations

from dataclasses import replace

import pytest

from quant_research_platform.agents.dataset_planner import plan_research_dataset
from quant_research_platform.agents.factor_factory import propose_factor
from quant_research_platform.agents.label_factory import propose_label
from quant_research_platform.agents.proposals import (
    FactorProposal,
    LabelProposal,
    ResearchDatasetProposal,
    WorkflowProposal,
    factor_definition_from_proposal,
)
from quant_research_platform.agents.safety import (
    ProposalSafetyError,
    assert_proposal_is_share_safe,
    require_human_approval,
)
from quant_research_platform.agents.workflow_planner import plan_workflow
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.contracts.factors import FactorDefinition
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.markets.implementations.a_share.fields import (
    field_catalog,
)
from quant_research_platform.profiles.research import research_profile
from tests.helpers import frequency_ids

pytestmark = pytest.mark.guardrails


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
def test_share_safety_rejects_local_text_markers(unsafe_text: str) -> None:
    proposal = FactorProposal(
        proposal_id="factor-proposal-unsafe",
        factor_id="unsafe_factor",
        rationale=unsafe_text,
        input_columns=("close",),
    )

    with pytest.raises(ProposalSafetyError):
        assert_proposal_is_share_safe(proposal)


def test_share_safety_rejects_preapproved_proposals() -> None:
    proposal = WorkflowProposal(
        proposal_id="workflow-proposal-preapproved",
        workflow_id="unsafe_workflow",
        step_ids=("dataset_spec",),
        approved=True,
    )

    with pytest.raises(ProposalSafetyError, match="proposal_must_start_unapproved"):
        assert_proposal_is_share_safe(proposal)


def test_factories_return_only_share_safe_unapproved_proposals() -> None:
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


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "C" + chr(58) + r"\private\dataset.csv",
        "source_id=csmar",
        "row sample: asset_id=000001 value=1.2",
        "stock code 600519 has a high score",
    ],
)
def test_factor_factory_rejects_share_unsafe_proposal_text(
    unsafe_text: str,
) -> None:
    with pytest.raises(ProposalSafetyError):
        propose_factor(
            profile=research_profile(),
            proposal_id="factor-proposal-unsafe-factory",
            factor_id="unsafe_candidate",
            rationale=unsafe_text,
            input_columns=("close",),
        )


def test_factor_factory_rejects_unknown_input_columns() -> None:
    with pytest.raises(ValueError, match="unknown_factor_input_columns"):
        propose_factor(
            profile=research_profile(),
            proposal_id="factor-proposal-unknown-input",
            factor_id="bad_input_factor",
            rationale="Candidate with unregistered input.",
            input_columns=("unknown_input_column",),
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"output_columns": "abc"}, "invalid_proposal_tuple_field: output_columns"),
        (
            {"base_input_columns": "close"},
            "invalid_proposal_tuple_field: base_input_columns",
        ),
        (
            {"derived_input_columns": "abc"},
            "invalid_proposal_tuple_field: derived_input_columns",
        ),
        (
            {"derived_input_producers": [("x", "y")]},
            "invalid_factor_derived_input_producers",
        ),
    ],
)
def test_factor_factory_rejects_bare_string_or_non_dict_contract_fields(
    kwargs: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        propose_factor(
            profile=research_profile(),
            proposal_id="factor-proposal-bad-contract-field",
            factor_id="bad_contract_field",
            rationale="Candidate with invalid factory contract inputs.",
            input_columns=("close",),
            **kwargs,  # type: ignore[arg-type]
        )


def test_workflow_planner_rejects_bare_str_step_ids() -> None:
    with pytest.raises(ValueError, match="invalid_proposal_tuple_field: step_ids"):
        plan_workflow(
            proposal_id="workflow-proposal-bad-steps",
            workflow_id="bad_workflow",
            step_ids="abc",  # type: ignore[arg-type]
        )


def test_workflow_planner_rejects_non_string_step_ids() -> None:
    with pytest.raises(ValueError, match="invalid_proposal_tuple_field: step_ids"):
        plan_workflow(
            proposal_id="workflow-proposal-bad-step-item",
            workflow_id="bad_workflow",
            step_ids=("a", 1),  # type: ignore[arg-type]
        )


def test_dataset_planner_rejects_bare_str_input_data_ids() -> None:
    with pytest.raises(
        ValueError,
        match="invalid_proposal_tuple_field: input_data_ids",
    ):
        plan_research_dataset(
            proposal_id="dataset-proposal-bad-inputs",
            data_id="bad_dataset",
            input_data_ids="features",  # type: ignore[arg-type]
        )


def test_factor_factory_derives_unique_required_input_data_id_from_profile() -> None:
    proposal = propose_factor(
        profile=research_profile(),
        proposal_id="factor-proposal-inferred-input",
        factor_id="candidate_factor",
        rationale="Candidate based on registered close field.",
        input_columns=("close",),
    )

    assert proposal.required_input_data_id == frequency_ids.MARKET_PANEL_DATA_ID


def test_factor_factory_requires_explicit_input_for_multiple_inputs() -> None:
    profile = research_profile()
    factors = profile.factors.list()
    catalog = FactorCatalog()
    catalog.register(factors[0])
    catalog.register(
        replace(
            factors[0],
            factor_id="alternate_input_factor",
            output_column="alternate_input_factor",
            required_input_data_id="alternate_processed_input",
        )
    )
    ambiguous_profile = replace(profile, factors=catalog)

    with pytest.raises(ValueError, match="factor_required_input_data_id_ambiguous"):
        propose_factor(
            profile=ambiguous_profile,
            proposal_id="factor-proposal-ambiguous-input",
            factor_id="candidate_factor",
            rationale="Candidate based on registered close field.",
            input_columns=("close",),
        )


def test_factor_factory_explicit_input_resolves_multiple_inputs() -> None:
    profile = research_profile()
    factors = profile.factors.list()
    catalog = FactorCatalog()
    catalog.register(factors[0])
    catalog.register(
        replace(
            factors[0],
            factor_id="alternate_input_factor",
            output_column="alternate_input_factor",
            required_input_data_id="alternate_processed_input",
        )
    )
    ambiguous_profile = replace(profile, factors=catalog)

    proposal = propose_factor(
        profile=ambiguous_profile,
        proposal_id="factor-proposal-explicit-input",
        factor_id="candidate_factor",
        rationale="Candidate based on registered close field.",
        input_columns=("close",),
        required_input_data_id="alternate_processed_input",
    )

    assert proposal.required_input_data_id == "alternate_processed_input"


def test_unregistered_factor_proposal_has_no_execution_or_persistence_surface() -> None:
    proposal = FactorProposal(
        proposal_id="factor-proposal-unregistered",
        factor_id="candidate_factor",
        rationale="Candidate for catalog review only.",
        input_columns=("close",),
        output_columns=("candidate_factor",),
        required_input_data_id=frequency_ids.MARKET_PANEL_DATA_ID,
    )

    assert not hasattr(proposal, "exec" + "ute")
    assert not hasattr(proposal, "persist")
    assert not hasattr(proposal, "to_workflow_node")
    assert not hasattr(proposal, "output_scope")
    assert not hasattr(proposal, "artifact_id")


def test_human_approved_factor_proposal_can_be_registered_by_catalog_owner() -> None:
    proposal = replace(
        propose_factor(
            profile=research_profile(),
            proposal_id="factor-proposal-register",
            factor_id="candidate_factor",
            rationale="Candidate based on registered close field.",
            input_columns=("close",),
            output_columns=("candidate_factor",),
            required_input_data_id=frequency_ids.MARKET_PANEL_DATA_ID,
        ),
        approved=True,
    )
    require_human_approval(proposal)
    assert proposal.approved

    catalog = FactorCatalog()

    definition = factor_definition_from_proposal(
        proposal,
        market_id=MarketId("a_share"),
        field_names=set(field_catalog().list_names()),
    )
    catalog.register(definition)

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
        required_input_data_id=frequency_ids.MARKET_PANEL_DATA_ID,
        approved=True,
    )

    with pytest.raises(ValueError, match="duplicate_factor_output_column"):
        factor_definition_from_proposal(
            proposal,
            market_id=MarketId("a_share"),
            field_names=set(field_catalog().list_names()),
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
        required_input_data_id=frequency_ids.MARKET_PANEL_DATA_ID,
        approved=True,
    )

    with pytest.raises(ValueError, match="unresolved_factor_derived_input"):
        factor_definition_from_proposal(
            proposal,
            market_id=MarketId("a_share"),
            field_names=set(field_catalog().list_names()),
        )


def test_proposals_expose_spec_like_payloads_without_approval_state() -> None:
    proposal = WorkflowProposal(
        proposal_id="workflow-proposal-003",
        workflow_id="factor_review_workflow",
        step_ids=("dataset_spec", "human_review"),
    )

    payload = proposal.to_spec_payload()

    assert {
        "proposal_id": "workflow-proposal-003",
        "workflow_id": "factor_review_workflow",
        "step_ids": ("dataset_spec", "human_review"),
        "contract_version": "0.1",
        "proposal_type": "workflow",
        "title": "factor_review_workflow",
        "rationale": "workflow proposal",
        "expected_inputs": ("dataset_spec", "human_review"),
        "expected_outputs": ("factor_review_workflow",),
        "required_contracts": (),
        "status": "proposed",
    }.items() <= payload.items()
    assert "approved" not in payload
