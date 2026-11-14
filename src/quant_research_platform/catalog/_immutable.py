"""Store ordered catalog entries; subclasses validate their identities."""

from __future__ import annotations

from collections.abc import Iterable


class ImmutableCatalog[CatalogItem]:
    def __init__(self, items: Iterable[CatalogItem] = ()) -> None:
        self._items = tuple(items)

    def list(self) -> tuple[CatalogItem, ...]:
        return self._items
