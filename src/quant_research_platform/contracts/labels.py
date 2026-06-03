"""Label definition contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quant_research_platform.contracts.ids import MarketId


@dataclass(frozen=True)
class LabelDefinition:
    label_id: str
    market_id: MarketId
    output_column: str
    label_timing: str
    horizon: int
    lookahead: int
    entry_rule: str
    exit_rule: str
    definition_version: str = "0.1"

    def to_manifest(self) -> dict[str, Any]:
        return {
            "definition_version": self.definition_version,
            "label_id": self.label_id,
            "market_id": str(self.market_id),
            "output_column": self.output_column,
            "label_timing": self.label_timing,
            "horizon": self.horizon,
            "lookahead": self.lookahead,
            "entry_rule": self.entry_rule,
            "exit_rule": self.exit_rule,
        }


__all__ = ["LabelDefinition"]
