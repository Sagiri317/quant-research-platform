"""Run output definition catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.run_outputs import RunOutputDefinition


@dataclass
class RunOutputCatalog:
    _items: dict[str, RunOutputDefinition] = field(default_factory=dict)

    def register(self, definition: RunOutputDefinition) -> None:
        if definition.output_definition_id in self._items:
            raise ValueError(
                f"duplicate_output_definition: {definition.output_definition_id}"
            )
        self._items[definition.output_definition_id] = definition

    def get(self, output_definition_id: str) -> RunOutputDefinition:
        try:
            return self._items[output_definition_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown_output_definition: {output_definition_id}"
            ) from exc

    def list(self) -> tuple[RunOutputDefinition, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["RunOutputCatalog"]
