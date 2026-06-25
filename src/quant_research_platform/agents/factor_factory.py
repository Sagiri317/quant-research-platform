"""Proposal-only Factor Factory boundary."""

from quant_research_platform.agents.proposals import FactorProposal
from quant_research_platform.agents.safety import assert_proposal_is_share_safe
from quant_research_platform.profiles.research import ResearchProfile


def propose_factor(
    *,
    profile: ResearchProfile,
    proposal_id: str,
    factor_id: str,
    rationale: str,
    input_columns: tuple[str, ...],
    output_columns: tuple[str, ...] | None = None,
    base_input_columns: tuple[str, ...] | None = None,
    derived_input_columns: tuple[str, ...] = (),
    derived_input_producers: dict[str, str] | None = None,
    required_input_data_id: str | None = None,
) -> FactorProposal:
    _validate_input_columns(tuple(input_columns), profile=profile)
    resolved_input_data_id = required_input_data_id or _unique_factor_input_data_id(
        profile
    )
    proposal = FactorProposal(
        proposal_id=proposal_id,
        factor_id=factor_id,
        rationale=rationale,
        input_columns=tuple(input_columns),
        output_columns=tuple(output_columns or (factor_id,)),
        base_input_columns=tuple(base_input_columns or input_columns),
        derived_input_columns=tuple(derived_input_columns),
        derived_input_producers=dict(derived_input_producers or {}),
        required_input_data_id=resolved_input_data_id,
    )
    assert_proposal_is_share_safe(proposal)
    return proposal


def _validate_input_columns(
    input_columns: tuple[str, ...],
    *,
    profile: ResearchProfile,
) -> None:
    fields = profile.fields
    unknown = sorted(set(input_columns).difference(fields.list_names()))
    if unknown:
        raise ValueError(f"unknown_factor_input_columns: {', '.join(unknown)}")


def _unique_factor_input_data_id(profile: ResearchProfile) -> str:
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
