"""Small immutable tuple-backed catalog base."""

from __future__ import annotations

from collections.abc import Iterable


class ImmutableCatalog[CatalogItem]:
    __slots__ = ("_items",)

    def __init__(self, items: Iterable[CatalogItem] = ()) -> None:
        object.__setattr__(self, "_items", tuple(items))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("catalog facts are immutable")

    def list(self) -> tuple[CatalogItem, ...]:
        return self._items
