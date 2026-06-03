"""Mask definition contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quant_research_platform.contracts.ids import MarketId


@dataclass(frozen=True)
class MaskDefinition:
    mask_id: str
    market_id: MarketId
    output_column: str
    mask_type: str
    uses_future_data: bool
    definition_version: str = "0.1"

    def to_manifest(self) -> dict[str, Any]:
        return {
            "definition_version": self.definition_version,
            "mask_id": self.mask_id,
            "market_id": str(self.market_id),
            "output_column": self.output_column,
            "mask_type": self.mask_type,
            "uses_future_data": self.uses_future_data,
        }


__all__ = ["MaskDefinition"]
