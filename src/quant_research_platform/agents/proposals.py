"""Agent-assisted research payload contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from quant_research_platform.agents.safety import (
    assert_proposal_text_is_share_safe,
    require_human_approval,
)
from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId

PROPOSAL_STATUSES = frozenset({"proposed", "needs_human_approval", "rejected"})


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
    family: str = "proposal"
    grain: DataGrain = DataGrain.ASSET_TIMESTAMP
    lookback_window: int | None = None
    timing_semantics: str = "as_of_timestamp"
    null_policy: str = "preserve_nulls"
    contract_version: str = "0.1"
    proposal_type: str = "factor"
    title: str = ""
    expected_inputs: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    required_contracts: tuple[str, ...] = ("FactorDefinition",)
    status: str = "proposed"
    approved: bool = False

    def __post_init__(self) -> None:
        _validate_common_proposal(
            self,
            id_fields=("proposal_id", "factor_id"),
            proposal_type="factor",
        )
        _validate_tuple_field(self.input_columns, "input_columns")
        _validate_tuple_field(self.output_columns, "output_columns", required=False)
        _validate_tuple_field(
            self.base_input_columns,
            "base_input_columns",
            required=False,
        )
        _validate_tuple_field(
            self.derived_input_columns,
            "derived_input_columns",
            required=False,
        )
        _validate_tuple_field(self.expected_inputs, "expected_inputs", required=False)
        _validate_tuple_field(self.expected_outputs, "expected_outputs", required=False)
        _validate_derived_input_producers(self.derived_input_producers)
        if not isinstance(self.grain, DataGrain):
            raise ValueError("invalid_factor_proposal_grain")
        for field_name in (
            "required_input_data_id",
            "family",
            "timing_semantics",
            "null_policy",
        ):
            _validate_non_empty_string(getattr(self, field_name), field_name)
        if self.lookback_window is not None and (
            not isinstance(self.lookback_window, int)
            or isinstance(self.lookback_window, bool)
            or self.lookback_window < 0
        ):
            raise ValueError("invalid_factor_proposal_lookback_window")
        _validate_proposal_status(self.status)
        if self.contract_version != "0.1":
            raise ValueError("unsupported_factor_proposal_contract_version")
        if not self.output_columns:
            object.__setattr__(self, "output_columns", (self.factor_id,))
        if not self.base_input_columns and not self.derived_input_columns:
            object.__setattr__(self, "base_input_columns", self.input_columns)
        if not self.title:
            object.__setattr__(self, "title", self.factor_id)
        if not self.expected_inputs:
            object.__setattr__(self, "expected_inputs", self.input_columns)
        if not self.expected_outputs:
            object.__setattr__(self, "expected_outputs", self.output_columns)


@dataclass(frozen=True)
class LabelProposal(ProposalPayloadMixin):
    proposal_id: str
    label_id: str
    rationale: str
    horizon: int
    contract_version: str = "0.1"
    proposal_type: str = "label"
    title: str = ""
    expected_inputs: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    required_contracts: tuple[str, ...] = ("LabelDefinition",)
    status: str = "proposed"
    approved: bool = False

    def __post_init__(self) -> None:
        _validate_common_proposal(
            self,
            id_fields=("proposal_id", "label_id"),
            proposal_type="label",
        )
        _validate_positive_integer(
            self.horizon,
            error="invalid_label_proposal_horizon",
        )
        _validate_tuple_field(self.expected_inputs, "expected_inputs", required=False)
        _validate_tuple_field(self.expected_outputs, "expected_outputs", required=False)
        _validate_proposal_status(self.status)
        if self.contract_version != "0.1":
            raise ValueError("unsupported_label_proposal_contract_version")
        if not self.title:
            object.__setattr__(self, "title", self.label_id)
        if not self.expected_outputs:
            object.__setattr__(self, "expected_outputs", (self.label_id,))


@dataclass(frozen=True)
class ResearchDatasetProposal(ProposalPayloadMixin):
    proposal_id: str
    data_id: str
    input_data_ids: tuple[str, ...]
    contract_version: str = "0.1"
    proposal_type: str = "research_dataset"
    title: str = ""
    rationale: str = "research dataset proposal"
    expected_inputs: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    required_contracts: tuple[str, ...] = ("ResearchDatasetSpec",)
    status: str = "proposed"
    approved: bool = False

    def __post_init__(self) -> None:
        _validate_common_proposal(
            self,
            id_fields=("proposal_id", "data_id"),
            proposal_type="research_dataset",
        )
        _validate_tuple_field(self.input_data_ids, "input_data_ids")
        _validate_tuple_field(self.expected_inputs, "expected_inputs", required=False)
        _validate_tuple_field(self.expected_outputs, "expected_outputs", required=False)
        _validate_proposal_status(self.status)
        if self.contract_version != "0.1":
            raise ValueError("unsupported_research_dataset_proposal_contract_version")
        if not self.title:
            object.__setattr__(self, "title", self.data_id)
        if not self.expected_inputs:
            object.__setattr__(self, "expected_inputs", self.input_data_ids)
        if not self.expected_outputs:
            object.__setattr__(self, "expected_outputs", (self.data_id,))


@dataclass(frozen=True)
class WorkflowProposal(ProposalPayloadMixin):
    proposal_id: str
    workflow_id: str
    step_ids: tuple[str, ...] = field(default_factory=tuple)
    contract_version: str = "0.1"
    proposal_type: str = "workflow"
    title: str = ""
    rationale: str = "workflow proposal"
    expected_inputs: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    required_contracts: tuple[str, ...] = ()
    status: str = "proposed"
    approved: bool = False

    def __post_init__(self) -> None:
        _validate_common_proposal(
            self,
            id_fields=("proposal_id", "workflow_id"),
            proposal_type="workflow",
        )
        _validate_tuple_field(self.step_ids, "step_ids", required=False)
        _validate_tuple_field(self.expected_inputs, "expected_inputs", required=False)
        _validate_tuple_field(self.expected_outputs, "expected_outputs", required=False)
        _validate_proposal_status(self.status)
        if self.contract_version != "0.1":
            raise ValueError("unsupported_workflow_proposal_contract_version")
        if not self.title:
            object.__setattr__(self, "title", self.workflow_id)
        if not self.expected_inputs:
            object.__setattr__(self, "expected_inputs", self.step_ids)
        if not self.expected_outputs:
            object.__setattr__(self, "expected_outputs", (self.workflow_id,))


def factor_definition_from_proposal(
    proposal: FactorProposal,
    *,
    market_id: MarketId,
    field_names: set[str],
) -> FactorDefinition:
    require_human_approval(proposal)
    assert_proposal_text_is_share_safe(proposal)
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
        family=proposal.family,
        lookback_window=proposal.lookback_window,
        grain=proposal.grain,
        timing_semantics=proposal.timing_semantics,
        null_policy=proposal.null_policy,
        required_input_data_id=proposal.required_input_data_id,
    )


def _validate_proposal_status(status: object) -> None:
    if not isinstance(status, str):
        raise ValueError("invalid_proposal_status")
    if status not in PROPOSAL_STATUSES:
        raise ValueError("invalid_proposal_status")


def _validate_common_proposal(
    proposal: object,
    *,
    id_fields: tuple[str, ...],
    proposal_type: str,
) -> None:
    if not isinstance(proposal.approved, bool):
        raise ValueError("invalid_proposal_approved")
    if proposal.proposal_type != proposal_type:
        raise ValueError("invalid_proposal_type")
    if not isinstance(proposal.rationale, str) or not proposal.rationale:
        raise ValueError("invalid_proposal_rationale")
    if not isinstance(proposal.title, str):
        raise ValueError("invalid_proposal_title")
    _validate_tuple_field(
        proposal.required_contracts,
        "required_contracts",
        required=False,
    )
    for field_name in id_fields:
        value = getattr(proposal, field_name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"invalid_proposal_id_field: {field_name}")


def _validate_tuple_field(
    value: object,
    field_name: str,
    *,
    required: bool = True,
) -> None:
    if not isinstance(value, tuple):
        raise ValueError(f"invalid_proposal_tuple_field: {field_name}")
    if required and not value:
        raise ValueError(f"invalid_proposal_tuple_field: {field_name}")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"invalid_proposal_tuple_field: {field_name}")


def _validate_non_empty_string(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid_factor_proposal_semantic_field: {field_name}")


def _validate_positive_integer(value: object, *, error: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(error)
    return value


def _validate_derived_input_producers(value: object) -> None:
    if not isinstance(value, dict) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(producer, str)
        or not producer
        for key, producer in value.items()
    ):
        raise ValueError("invalid_factor_derived_input_producers")


__all__ = [
    "FactorProposal",
    "LabelProposal",
    "PROPOSAL_STATUSES",
    "ProposalPayloadMixin",
    "ResearchDatasetProposal",
    "WorkflowProposal",
    "factor_definition_from_proposal",
]
