"""Human-approved proposal objects for agent-driven research planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId


class ProposalPayloadMixin:
    def to_spec_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("approved", None)
        return payload


@dataclass(frozen=True)
class FactorProposal(ProposalPayloadMixin):
    proposal_id: str
    factor_id: str
    rationale: str
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...] = ()
    base_input_columns: tuple[str, ...] = ()
    derived_input_columns: tuple[str, ...] = ()
    derived_input_producers: dict[str, str] = field(default_factory=dict)
    required_input_data_id: str = ""
    contract_version: str = "0.1"
    approved: bool = False

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_factor_proposal_contract_version")
        if not self.output_columns:
            object.__setattr__(self, "output_columns", (self.factor_id,))
        if not self.base_input_columns and not self.derived_input_columns:
            object.__setattr__(self, "base_input_columns", self.input_columns)


@dataclass(frozen=True)
class LabelProposal(ProposalPayloadMixin):
    proposal_id: str
    label_id: str
    rationale: str
    horizon: int
    contract_version: str = "0.1"
    approved: bool = False


@dataclass(frozen=True)
class ResearchDatasetProposal(ProposalPayloadMixin):
    proposal_id: str
    data_id: str
    input_data_ids: tuple[str, ...]
    contract_version: str = "0.1"
    approved: bool = False


@dataclass(frozen=True)
class WorkflowProposal(ProposalPayloadMixin):
    proposal_id: str
    workflow_id: str
    step_ids: tuple[str, ...] = field(default_factory=tuple)
    contract_version: str = "0.1"
    approved: bool = False


def factor_definition_from_proposal(
    proposal: FactorProposal,
    *,
    market_id: MarketId,
    field_names: set[str],
) -> FactorDefinition:
    output_columns = tuple(proposal.output_columns or (proposal.factor_id,))
    if len(set(output_columns)) != len(output_columns):
        raise ValueError("duplicate_factor_output_column")
    if len(output_columns) != 1:
        raise ValueError("factor_proposal_single_output_required")

    base_inputs = tuple(proposal.base_input_columns or proposal.input_columns)
    derived_inputs = tuple(proposal.derived_input_columns)
    expected_inputs = tuple(dict.fromkeys((*base_inputs, *derived_inputs)))
    if expected_inputs != tuple(proposal.input_columns):
        raise ValueError("factor_input_columns_must_match_base_and_derived")

    unknown_base = sorted(set(base_inputs).difference(field_names))
    if unknown_base:
        raise ValueError(f"unknown_factor_input_columns: {', '.join(unknown_base)}")
    producers = dict(proposal.derived_input_producers or {})
    unresolved_derived = [
        column
        for column in derived_inputs
        if column not in field_names and not producers.get(column)
    ]
    if unresolved_derived:
        raise ValueError(
            "unresolved_factor_derived_input: "
            + ", ".join(sorted(unresolved_derived))
        )
    if not proposal.required_input_data_id:
        raise ValueError("factor_required_input_data_id_required")

    return FactorDefinition(
        factor_id=proposal.factor_id,
        market_id=market_id,
        factor_set=FactorSet.EXPERIMENTAL,
        input_columns=tuple(proposal.input_columns),
        base_input_columns=base_inputs,
        derived_input_columns=derived_inputs,
        output_column=output_columns[0],
        display_name=proposal.factor_id,
        description=proposal.rationale,
        family="proposal",
        required_input_data_id=proposal.required_input_data_id,
    )


__all__ = [
    "FactorProposal",
    "LabelProposal",
    "ProposalPayloadMixin",
    "ResearchDatasetProposal",
    "WorkflowProposal",
    "factor_definition_from_proposal",
]
