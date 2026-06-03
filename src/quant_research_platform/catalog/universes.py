"""Universe catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.universes import UniverseDefinition


@dataclass
class UniverseCatalog:
    _items: dict[str, UniverseDefinition] = field(default_factory=dict)

    def register(self, definition: UniverseDefinition) -> None:
        self._items[definition.universe_id] = definition


__all__ = ["UniverseCatalog"]
