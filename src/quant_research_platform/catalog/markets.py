"""Market implementation registry."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from quant_research_platform.catalog._immutable import ImmutableCatalog
from quant_research_platform.contracts.ids import MarketId


@dataclass(frozen=True)
class MarketDefinition:
    market_id: MarketId
    display_name: str
    processed_spec_ids: tuple[str, ...]
    definition_version: str = "0.1"


class MarketRegistry(ImmutableCatalog[MarketDefinition]):
    def __init__(self, items: Iterable[MarketDefinition] = ()) -> None:
        values = tuple(items)
        seen: set[str] = set()
        for definition in values:
            key = str(definition.market_id)
            if key in seen:
                raise ValueError(f"duplicate_market_definition: {key}")
            seen.add(key)
        super().__init__(sorted(values, key=lambda item: str(item.market_id)))

    def get(self, market_id: str) -> MarketDefinition:
        try:
            return next(
                definition
                for definition in self._items
                if str(definition.market_id) == market_id
            )
        except StopIteration as exc:
            raise KeyError(f"unknown_market_id: {market_id}") from exc


__all__ = ["MarketDefinition", "MarketRegistry"]
