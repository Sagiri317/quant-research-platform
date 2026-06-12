"""Label catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.labels import LabelDefinition


@dataclass
class LabelCatalog:
    _items: dict[str, LabelDefinition] = field(default_factory=dict)

    def register(self, definition: LabelDefinition, *, replace: bool = False) -> None:
        if definition.label_id in self._items and not replace:
            raise ValueError(f"duplicate_label_definition: {definition.label_id}")
        self._items[definition.label_id] = definition

    def list(self) -> tuple[LabelDefinition, ...]:
        return tuple(self._items.values())


__all__ = ["LabelCatalog"]
