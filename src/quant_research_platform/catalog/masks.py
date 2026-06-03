"""Mask catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.masks import MaskDefinition


@dataclass
class MaskCatalog:
    _items: dict[str, MaskDefinition] = field(default_factory=dict)

    def register(self, definition: MaskDefinition) -> None:
        self._items[definition.mask_id] = definition


__all__ = ["MaskCatalog"]
