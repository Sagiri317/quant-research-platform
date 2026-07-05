from __future__ import annotations

import ast
from pathlib import Path

import pytest

from quant_research_platform.agents.factor_factory import propose_factor
from quant_research_platform.agents.proposals import (
    PROPOSAL_STATUSES,
    FactorProposal,
    ResearchDatasetProposal,
    factor_definition_from_proposal,
)
from quant_research_platform.agents.safety import (
    ProposalSafetyError,
    assert_proposal_is_share_safe,
)
from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.factors import FactorSet
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.profiles.research import research_profile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_SRC = PROJECT_ROOT / "src" / "quant_research_platform" / "agents"


def _sample_windows_absolute_path() -> str:
    return "".join(
        ("C", chr(58), chr(92), "Users", chr(92), "name", chr(92), "private.txt")
    )


def _sample_provider_marker() -> str:
    marker_key = "".join(("provider", "_", "mark", "er="))
    provider_name = "".join(("cs", "mar"))
    return marker_key + provider_name


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


def test_crypto_factor_proposal_is_validation_only_and_unapproved() -> None:
    profile = research_profile("crypto_binance_1m")

    proposal = propose_factor(
        profile=profile,
        proposal_id="crypto-factor-proposal-001",
        factor_id="depth_pressure_demo",
        rationale="Evaluate aggregate quote pressure using review-safe fields.",
        input_columns=("bid_depth", "ask_depth"),
        family="microstructure",
        grain=DataGrain.ASSET_TIMESTAMP,
        lookback_window=5,
        timing_semantics="uses_past_and_current_ticks",
        null_policy="preserve_nulls",
    )

    with pytest.raises(PermissionError, match="proposal_requires_human_approval"):
        factor_definition_from_proposal(
            proposal,
            market_id=MarketId("crypto"),
            field_names=set(profile.fields.list_names()),
        )

    approved_proposal = FactorProposal(
        **{**proposal.__dict__, "approved": True},
    )
    definition = factor_definition_from_proposal(
        approved_proposal,
        market_id=MarketId("crypto"),
        field_names=set(profile.fields.list_names()),
    )

    assert proposal.approved is False
    assert proposal.status == "proposed"
    assert proposal.required_input_data_id == "market_panel_1m"
    assert definition.factor_set is FactorSet.EXPERIMENTAL
    assert definition.required_input_data_id == "market_panel_1m"
    assert definition.family == "microstructure"
    assert definition.grain is DataGrain.ASSET_TIMESTAMP
    assert definition.lookback_window == 5
    assert definition.timing_semantics == "uses_past_and_current_ticks"
    assert definition.null_policy == "preserve_nulls"


@pytest.mark.parametrize(
    "unsafe_text",
    (
        _sample_windows_absolute_path(),
        "row" + "_samples=[{'asset': 'sample'}]",
        _sample_provider_marker(),
        "outputs/predictions.parquet",
        "private/raw_frame.csv",
        "payload.pkl",
        "tensor.arrow",
        "source_asset_id",
        "source_id",
        "api_key=hidden",
        "access_token=hidden",
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
        "quant_research_platform.markets",
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
