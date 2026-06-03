"""Market implementation registry."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.ids import MarketId


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
                market_id=MarketId("a_share"),
                display_name="A-share market implementation",
                processed_spec_ids=(
                    "daily_panel",
                    "daily_context",
                    "a_share_default",
                    "a_share_as_of",
                    "a_share_execution",
                    "a_share_evaluation",
                    "price_volume_56",
                    "forward_returns",
                ),
            )
        )
        return registry


__all__ = ["MarketDefinition", "MarketRegistry"]
