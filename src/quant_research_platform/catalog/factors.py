"""Factor catalog registry."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId


@dataclass
class FactorCatalog:
    _definitions: dict[str, FactorDefinition] = field(default_factory=dict)

    def register(self, definition: FactorDefinition) -> None:
        if definition.factor_id in self._definitions:
            raise ValueError(f"duplicate_factor_definition: {definition.factor_id}")
        self._definitions[definition.factor_id] = definition

    def get(self, factor_id: str) -> FactorDefinition:
        return self._definitions[factor_id]

    def list(self) -> tuple[FactorDefinition, ...]:
        return tuple(self._definitions[key] for key in sorted(self._definitions))

    def resolve_factor_set(self, value: str | FactorSet) -> FactorSet:
        if isinstance(value, FactorSet):
            return value
        try:
            return FactorSet(value)
        except ValueError as exc:
            raise ValueError("invalid_factor_set") from exc

    @classmethod
    def bootstrap_default(cls) -> FactorCatalog:
        from quant_research_platform.markets.implementations.a_share.factors.literature_price_volume import (  # noqa: E501
            expected_literature_price_volume_factor_columns,
        )

        catalog = cls()
        for factor_id in expected_literature_price_volume_factor_columns():
            catalog.register(
                FactorDefinition(
                    factor_id=factor_id,
                    market_id=MarketId("a_share"),
                    factor_set=FactorSet.FULL,
                    input_columns=(),
                    output_column=factor_id,
                )
            )
        return catalog


__all__ = ["FactorCatalog"]
