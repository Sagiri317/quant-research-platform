"""Universe definition contracts."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.contracts.ids import MarketId


@dataclass(frozen=True)
class UniverseDefinition:
    universe_id: str
    market_id: MarketId
    membership_column: str
    definition_version: str = "0.1"


__all__ = ["UniverseDefinition"]
