"""Factor catalog registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId


@dataclass
class FactorCatalog:
    _definitions: dict[str, FactorDefinition] = field(default_factory=dict)

    def register(self, definition: FactorDefinition) -> None:
        if definition.factor_id in self._definitions:
            raise ValueError(f"duplicate_factor_definition: {definition.factor_id}")
        if any(
            item.output_column == definition.output_column
            for item in self._definitions.values()
        ):
            raise ValueError(
                f"duplicate_factor_output_column: {definition.output_column}"
            )
        self._definitions[definition.factor_id] = definition

    def register_proposal(
        self,
        proposal: Any,
        *,
        market_id: MarketId,
        field_names: set[str],
    ) -> FactorDefinition:
        output_columns = tuple(proposal.output_columns or (proposal.factor_id,))
        if len(set(output_columns)) != len(output_columns):
            raise ValueError("duplicate_factor_output_column")
        if len(output_columns) != 1:
            raise ValueError("factor_proposal_single_output_required")
        if any(
            item.output_column == output_columns[0]
            for item in self._definitions.values()
        ):
            raise ValueError(f"duplicate_factor_output_column: {output_columns[0]}")

        base_inputs = tuple(proposal.base_input_columns or proposal.input_columns)
        derived_inputs = tuple(proposal.derived_input_columns)
        expected_inputs = tuple(dict.fromkeys((*base_inputs, *derived_inputs)))
        if expected_inputs != tuple(proposal.input_columns):
            raise ValueError("factor_input_columns_must_match_base_and_derived")

        unknown_base = sorted(set(base_inputs).difference(field_names))
        if unknown_base:
            raise ValueError(f"unknown_factor_input_columns: {', '.join(unknown_base)}")
        producers = dict(getattr(proposal, "derived_input_producers", {}) or {})
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
        if not proposal.required_processed_data_id:
            raise ValueError("factor_required_processed_data_id_required")

        definition = FactorDefinition(
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
            required_processed_data_id=proposal.required_processed_data_id,
        )
        self.register(definition)
        return definition

    def get(self, factor_id: str) -> FactorDefinition:
        return self._definitions[factor_id]

    def list(self) -> tuple[FactorDefinition, ...]:
        return tuple(self._definitions.values())

    def resolve_factor_set(self, value: str | FactorSet) -> FactorSet:
        if isinstance(value, FactorSet):
            return value
        try:
            return FactorSet(value)
        except ValueError as exc:
            raise ValueError("invalid_factor_set") from exc


__all__ = ["FactorCatalog"]
