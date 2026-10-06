"""Label definition contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quant_research_platform.contracts.ids import MarketId, validate_market_id


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
    horizon_unit: str = "market_sessions"

    def __post_init__(self) -> None:
        if type(self.label_id) is not str or not self.label_id:
            raise ValueError("label_id_required")
        object.__setattr__(self, "market_id", validate_market_id(self.market_id))
        if type(self.family) is not str or not self.family:
            raise ValueError("label_family_required")
        if type(self.input_columns) is not tuple or not self.input_columns:
            raise ValueError("label_input_columns_required")
        if any(type(column) is not str or not column for column in self.input_columns):
            raise ValueError("label_input_column_required")
        if len(set(self.input_columns)) != len(self.input_columns):
            raise ValueError("label_input_columns_must_not_contain_duplicates")
        if type(self.output_column) is not str or not self.output_column:
            raise ValueError("label_output_column_required")
        if type(self.label_timing) is not str or not self.label_timing:
            raise ValueError("label_timing_required")
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool):
            raise ValueError("label_horizon_must_be_positive")
        if self.horizon_unit not in {"market_sessions", "milliseconds"}:
            raise ValueError("label_horizon_unit_invalid")
        if self.horizon <= 0:
            raise ValueError("label_horizon_must_be_positive")
        if not isinstance(self.lookahead, int) or isinstance(self.lookahead, bool):
            raise ValueError("label_lookahead_must_be_positive")
        if self.lookahead <= 0:
            raise ValueError("label_lookahead_must_be_positive")
        if type(self.entry_rule) is not str or not self.entry_rule:
            raise ValueError("label_entry_rule_required")
        if type(self.exit_rule) is not str or not self.exit_rule:
            raise ValueError("label_exit_rule_required")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "label_id": self.label_id,
            "market_id": str(self.market_id),
            "family": self.family,
            "input_columns": list(self.input_columns),
            "output_column": self.output_column,
            "available_at_column": f"{self.output_column}_available_at",
            "label_timing": self.label_timing,
            "horizon": self.horizon,
            "horizon_unit": self.horizon_unit,
            "lookahead": self.lookahead,
            "entry_rule": self.entry_rule,
            "exit_rule": self.exit_rule,
        }

    @property
    def available_at_column(self) -> str:
        return f"{self.output_column}_available_at"
