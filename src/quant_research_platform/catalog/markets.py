"""Market implementation registry."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_AS_OF_MASK_ID,
    A_SHARE_DEFAULT_UNIVERSE_ID,
    A_SHARE_EVALUATION_MASK_ID,
    A_SHARE_EXECUTION_MASK_ID,
    A_SHARE_MARKET_ID,
    DAILY_CONTEXT_PROCESSED_DATA_ID,
    DAILY_PANEL_PROCESSED_DATA_ID,
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)


@dataclass(frozen=True)
class MarketDefinition:
    market_id: MarketId
    display_name: str
    processed_spec_ids: tuple[str, ...]
    definition_version: str = "0.1"


@dataclass
class MarketRegistry:
    _items: dict[str, MarketDefinition] = field(default_factory=dict)

    def register(self, definition: MarketDefinition) -> None:
        key = str(definition.market_id)
        if key in self._items:
            raise ValueError(f"duplicate_market_definition: {key}")
        self._items[key] = definition

    def get(self, market_id: str) -> MarketDefinition:
        try:
            return self._items[market_id]
        except KeyError as exc:
            raise KeyError(f"unknown_market_id: {market_id}") from exc

    def list(self) -> tuple[MarketDefinition, ...]:
        return tuple(self._items[key] for key in sorted(self._items))

    @classmethod
    def bootstrap_default(cls) -> MarketRegistry:
        registry = cls()
        registry.register(
            MarketDefinition(
                market_id=A_SHARE_MARKET_ID,
                display_name="A-share market implementation",
                processed_spec_ids=(
                    DAILY_PANEL_PROCESSED_DATA_ID,
                    DAILY_CONTEXT_PROCESSED_DATA_ID,
                    A_SHARE_DEFAULT_UNIVERSE_ID,
                    A_SHARE_AS_OF_MASK_ID,
                    A_SHARE_EXECUTION_MASK_ID,
                    A_SHARE_EVALUATION_MASK_ID,
                    PRICE_VOLUME_PROCESSED_DATA_ID,
                    FORWARD_RETURNS_PROCESSED_DATA_ID,
                ),
            )
        )
        return registry


__all__ = ["MarketDefinition", "MarketRegistry"]
