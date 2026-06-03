"""Market implementation base contracts."""

from __future__ import annotations

from dataclasses import dataclass

from quant_research_platform.contracts.ids import MarketId


@dataclass(frozen=True)
class MarketImplementation:
    market_id: MarketId
    enabled: bool = True


__all__ = ["MarketImplementation"]
