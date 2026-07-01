from __future__ import annotations

import ast
from pathlib import Path

import pytest

from quant_research_platform.agents.proposals import (
    PROPOSAL_STATUSES,
    FactorProposal,
    ResearchDatasetProposal,
)
from quant_research_platform.agents.safety import (
    ProposalSafetyError,
    assert_proposal_is_share_safe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_SRC = PROJECT_ROOT / "src" / "quant_research_platform" / "agents"


def test_agent_proposal_payload_has_required_shape_and_status() -> None:
    proposal = FactorProposal(
        proposal_id="factor-proposal-001",
        factor_id="factor_alpha",
        rationale="Evaluate a synthetic factor idea.",
        input_columns=("close",),
        required_input_data_id="price_volume",
    )

    payload = proposal.to_spec_payload()

    assert {
        "proposal_id",
        "proposal_type",
        "title",
        "rationale",
        "expected_inputs",
        "expected_outputs",
        "required_contracts",
        "status",
    }.issubset(payload)
    assert payload["status"] in PROPOSAL_STATUSES
    assert payload["expected_inputs"] == ("close",)
    assert payload["expected_outputs"] == ("factor_alpha",)
    assert "approved" not in payload


def test_agent_proposal_rejects_unapproved_status() -> None:
    with pytest.raises(ValueError, match="invalid_proposal_status"):
        ResearchDatasetProposal(
            proposal_id="dataset-proposal-001",
            data_id="research_dataset",
            input_data_ids=("features",),
            status="approved",
        )


@pytest.mark.parametrize(
    "unsafe_text",
    (
        "C:" + "\\Users\\name\\private.txt",
        "row" + "_samples=[{'asset': 'sample'}]",
        "provider" + "_marker=" + "csmar",
        "outputs/predictions.parquet",
        "private/raw_frame.csv",
        "payload.pkl",
        "tensor.arrow",
    ),
)
def test_agent_safety_rejects_private_or_row_level_text(unsafe_text: str) -> None:
    proposal = ResearchDatasetProposal(
        proposal_id="dataset-proposal-unsafe",
        data_id="research_dataset",
        input_data_ids=("features",),
        rationale=unsafe_text,
    )

    with pytest.raises(
        ProposalSafetyError,
        match="proposal_contains_share_unsafe_text",
    ):
        assert_proposal_is_share_safe(proposal)


def test_agents_do_not_import_storage_sources_or_a_share_implementation() -> None:
    forbidden_roots = {
        "quant_research_platform.storage",
        "quant_research_platform.sources",
        "quant_research_platform.markets.implementations.a_share",
    }
    imported: list[str] = []
    for path in sorted(AGENT_SRC.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)

    violations = sorted(
        module
        for module in imported
        if any(
            module == root or module.startswith(root + ".")
            for root in forbidden_roots
        )
    )
    assert violations == []
