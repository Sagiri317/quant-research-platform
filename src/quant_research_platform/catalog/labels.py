"""Label catalog."""

from __future__ import annotations

from collections.abc import Iterable

from quant_research_platform.catalog._immutable import ImmutableCatalog
from quant_research_platform.contracts.labels import LabelDefinition


class LabelCatalog(ImmutableCatalog[LabelDefinition]):
    def __init__(self, items: Iterable[LabelDefinition] = ()) -> None:
        values = tuple(items)
        seen: set[str] = set()
        for definition in values:
            if definition.label_id in seen:
                raise ValueError(f"duplicate_label_definition: {definition.label_id}")
            seen.add(definition.label_id)
        super().__init__(values)

    def get(self, label_id: str) -> LabelDefinition:
        try:
            return next(
                definition
                for definition in self._items
                if definition.label_id == label_id
            )
        except StopIteration as exc:
            raise KeyError(label_id) from exc
