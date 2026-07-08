"""Agent-assisted factor research helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from quant_research_platform.agents.proposals import FactorProposal
from quant_research_platform.agents.safety import assert_proposal_is_share_safe
from quant_research_platform.contracts.data import DataGrain

if TYPE_CHECKING:
    from quant_research_platform.catalog.factors import FactorCatalog
    from quant_research_platform.catalog.fields import FieldCatalog


class ResearchProfileLike(Protocol):
    fields: FieldCatalog
    factors: FactorCatalog


def propose_factor(
    *,
    profile: ResearchProfileLike,
    proposal_id: str,
    factor_id: str,
    rationale: str,
    input_columns: tuple[str, ...],
    output_columns: tuple[str, ...] | None = None,
    base_input_columns: tuple[str, ...] | None = None,
    derived_input_columns: tuple[str, ...] = (),
    derived_input_producers: dict[str, str] | None = None,
    required_input_data_id: str | None = None,
    family: str = "proposal",
    grain: DataGrain = DataGrain.ASSET_TIMESTAMP,
    lookback_window: int | None = None,
    timing_semantics: str = "as_of_timestamp",
    null_policy: str = "preserve_nulls",
) -> FactorProposal:
    _validate_input_columns(input_columns, profile=profile)
    resolved_output_columns = _validate_optional_tuple(
        output_columns,
        "output_columns",
        default=(factor_id,),
    )
    resolved_base_input_columns = _validate_optional_tuple(
        base_input_columns,
        "base_input_columns",
        default=input_columns,
    )
    resolved_derived_input_columns = _validate_tuple_field(
        derived_input_columns,
        "derived_input_columns",
        required=False,
    )
    resolved_derived_input_producers = _validate_derived_input_producers(
        derived_input_producers,
    )
    resolved_input_data_id = required_input_data_id or _unique_factor_input_data_id(
        profile
    )
    proposal = FactorProposal(
        proposal_id=proposal_id,
        factor_id=factor_id,
        rationale=rationale,
        input_columns=input_columns,
        output_columns=resolved_output_columns,
        base_input_columns=resolved_base_input_columns,
        derived_input_columns=resolved_derived_input_columns,
        derived_input_producers=resolved_derived_input_producers,
        required_input_data_id=resolved_input_data_id,
        family=family,
        grain=grain,
        lookback_window=lookback_window,
        timing_semantics=timing_semantics,
        null_policy=null_policy,
    )
    assert_proposal_is_share_safe(proposal)
    return proposal


def _validate_input_columns(
    input_columns: tuple[str, ...],
    *,
    profile: ResearchProfileLike,
) -> None:
    if not isinstance(input_columns, tuple) or any(
        not isinstance(column, str) or not column for column in input_columns
    ):
        raise ValueError("invalid_proposal_tuple_field: input_columns")
    fields = profile.fields
    unknown = sorted(set(input_columns).difference(fields.list_names()))
    if unknown:
        raise ValueError(f"unknown_factor_input_columns: {', '.join(unknown)}")


def _validate_optional_tuple(
    value: tuple[str, ...] | None,
    field_name: str,
    *,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if value is None:
        return default
    return _validate_tuple_field(value, field_name, required=False)


def _validate_tuple_field(
    value: object,
    field_name: str,
    *,
    required: bool,
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"invalid_proposal_tuple_field: {field_name}")
    if required and not value:
        raise ValueError(f"invalid_proposal_tuple_field: {field_name}")
    if any(not isinstance(column, str) or not column for column in value):
        raise ValueError(f"invalid_proposal_tuple_field: {field_name}")
    return value


def _validate_derived_input_producers(
    value: dict[str, str] | None,
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or any(
        not isinstance(column, str)
        or not column
        or not isinstance(producer, str)
        or not producer
        for column, producer in value.items()
    ):
        raise ValueError("invalid_factor_derived_input_producers")
    return value


def _unique_factor_input_data_id(profile: ResearchProfileLike) -> str:
    candidates = sorted(
        {
            definition.required_input_data_id
            for definition in profile.factors.list()
            if definition.required_input_data_id
        }
    )
    if not candidates:
        raise ValueError("factor_required_input_data_id_not_inferable")
    if len(candidates) > 1:
        raise ValueError(
            "factor_required_input_data_id_ambiguous: " + ", ".join(candidates)
        )
    return candidates[0]


__all__ = ["FactorProposal", "propose_factor"]
