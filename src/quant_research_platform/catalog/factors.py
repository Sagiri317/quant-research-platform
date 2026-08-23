"""Factor catalog registry."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.catalog._immutable import ImmutableCatalog
from quant_research_platform.contracts.factors import FactorDefinition


class FactorCatalog(ImmutableCatalog[FactorDefinition]):
    def __init__(self, definitions: Iterable[FactorDefinition] = ()) -> None:
        values = tuple(definitions)
        seen: set[str] = set()
        output_columns: set[str] = set()
        for definition in values:
            if definition.factor_id in seen:
                raise ValueError(f"duplicate_factor_definition: {definition.factor_id}")
            if definition.output_column in output_columns:
                raise ValueError(
                    f"duplicate_factor_output_column: {definition.output_column}"
                )
            seen.add(definition.factor_id)
            output_columns.add(definition.output_column)
        super().__init__(values)

    def get(self, factor_id: str) -> FactorDefinition:
        try:
            return next(
                definition
                for definition in self._items
                if definition.factor_id == factor_id
            )
        except StopIteration as exc:
            raise KeyError(factor_id) from exc


__all__ = ["FactorCatalog"]
