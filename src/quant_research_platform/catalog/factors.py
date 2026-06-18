"""Factor catalog registry."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.factors import FactorDefinition


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

    def get(self, factor_id: str) -> FactorDefinition:
        return self._definitions[factor_id]

    def list(self) -> tuple[FactorDefinition, ...]:
        return tuple(self._definitions.values())


__all__ = ["FactorCatalog"]
