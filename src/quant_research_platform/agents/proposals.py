"""Human-approved proposal objects for agent-driven research planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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


__all__ = [
    "FactorProposal",
    "LabelProposal",
    "ProposalPayloadMixin",
    "ResearchDatasetProposal",
    "WorkflowProposal",
]
