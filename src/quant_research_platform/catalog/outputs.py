"""Output definition catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.outputs import OutputDefinition


@dataclass
class OutputCatalog:
    _items: dict[str, OutputDefinition] = field(default_factory=dict)

    def register(self, definition: OutputDefinition) -> None:
        if definition.output_definition_id in self._items:
            raise ValueError(
                f"duplicate_output_definition: {definition.output_definition_id}"
            )
        self._items[definition.output_definition_id] = definition

    def get(self, output_definition_id: str) -> OutputDefinition:
        try:
            return self._items[output_definition_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown_output_definition: {output_definition_id}"
            ) from exc

    def list(self) -> tuple[OutputDefinition, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["OutputCatalog"]
