"""Factor definition contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from quant_research_platform.contracts.ids import MarketId


class FactorSet(StrEnum):
    FULL = "full"
    CORE = "core"
    SMOKE = "smoke"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"


@dataclass(frozen=True)
class FactorDefinition:
    factor_id: str
    market_id: MarketId
    factor_set: FactorSet
    input_columns: tuple[str, ...]
    output_column: str
    window: int | None = None
    definition_version: str = "0.1"

    def to_manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "definition_version": self.definition_version,
            "factor_id": self.factor_id,
            "market_id": str(self.market_id),
            "factor_set": self.factor_set.value,
            "input_columns": list(self.input_columns),
            "output_column": self.output_column,
        }
        if self.window is not None:
            payload["window"] = self.window
        return payload


__all__ = ["FactorDefinition", "FactorSet"]
