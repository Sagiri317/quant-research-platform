"""Output definition catalog."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.catalog._immutable import ImmutableCatalog
from quant_research_platform.contracts.outputs import OutputDefinition


class OutputCatalog(ImmutableCatalog[OutputDefinition]):
    def __init__(self, items: Iterable[OutputDefinition] = ()) -> None:
        values = tuple(items)
        seen: set[str] = set()
        for definition in values:
            if definition.output_definition_id in seen:
                raise ValueError(
                    f"duplicate_output_definition: {definition.output_definition_id}"
                )
            seen.add(definition.output_definition_id)
        super().__init__(sorted(values, key=lambda item: item.output_definition_id))

    def get(self, output_definition_id: str) -> OutputDefinition:
        try:
            return next(
                definition
                for definition in self._items
                if definition.output_definition_id == output_definition_id
            )
        except StopIteration as exc:
            raise KeyError(
                f"unknown_output_definition: {output_definition_id}"
            ) from exc
