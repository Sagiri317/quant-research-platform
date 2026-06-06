"""Universe catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.universes import UniverseDefinition


@dataclass
class UniverseCatalog:
    _items: dict[str, UniverseDefinition] = field(default_factory=dict)

    def register(self, definition: UniverseDefinition) -> None:
        self._items[definition.universe_id] = definition

    def get(self, universe_id: str) -> UniverseDefinition:
        try:
            return self._items[universe_id]
        except KeyError as exc:
            raise KeyError(f"unknown_universe_definition: {universe_id}") from exc

    def list(self) -> tuple[UniverseDefinition, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["UniverseCatalog"]
