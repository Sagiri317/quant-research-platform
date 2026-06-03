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
    contract_version: str = "0.1"
    approved: bool = False


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
    dataset_id: str
    processed_data_ids: tuple[str, ...]
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
