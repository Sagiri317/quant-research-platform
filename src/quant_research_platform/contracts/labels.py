"""Label definition contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quant_research_platform.contracts.ids import MarketId


@dataclass(frozen=True)
class LabelDefinition:
    label_id: str
    market_id: MarketId
    family: str
    input_columns: tuple[str, ...]
    output_column: str
    label_timing: str
    horizon: int
    lookahead: int
    entry_rule: str
    exit_rule: str
    definition_version: str = "0.1"

    def __post_init__(self) -> None:
        if not self.label_id:
            raise ValueError("label_id_required")
        if not self.family:
            raise ValueError("label_family_required")
        if not self.input_columns:
            raise ValueError("label_input_columns_required")
        if any(not column for column in self.input_columns):
            raise ValueError("label_input_column_required")
        if not self.output_column:
            raise ValueError("label_output_column_required")
        if not self.label_timing:
            raise ValueError("label_timing_required")
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool):
            raise ValueError("label_horizon_must_be_positive")
        if self.horizon <= 0:
            raise ValueError("label_horizon_must_be_positive")
        if not isinstance(self.lookahead, int) or isinstance(self.lookahead, bool):
            raise ValueError("label_lookahead_must_be_positive")
        if self.lookahead <= 0:
            raise ValueError("label_lookahead_must_be_positive")
        if not self.entry_rule:
            raise ValueError("label_entry_rule_required")
        if not self.exit_rule:
            raise ValueError("label_exit_rule_required")
        if not self.definition_version:
            raise ValueError("label_definition_version_required")
        if self.definition_version != "0.1":
            raise ValueError("unsupported_label_definition_version")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "definition_version": self.definition_version,
            "label_id": self.label_id,
            "market_id": str(self.market_id),
            "family": self.family,
            "input_columns": list(self.input_columns),
            "output_column": self.output_column,
            "label_timing": self.label_timing,
            "horizon": self.horizon,
            "lookahead": self.lookahead,
            "entry_rule": self.entry_rule,
            "exit_rule": self.exit_rule,
        }
