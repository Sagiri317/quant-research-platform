"""Factor definition contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.ids import MarketId


class FactorSet(StrEnum):
    STANDARD = "standard"
    CORE = "core"
    SMOKE = "smoke"
    EXPERIMENTAL = "experimental"


@dataclass(frozen=True)
class FactorDefinition:
    factor_id: str
    market_id: MarketId
    factor_set: FactorSet
    input_columns: tuple[str, ...]
    output_column: str
    base_input_columns: tuple[str, ...] = ()
    derived_input_columns: tuple[str, ...] = ()
    display_name: str = ""
    description: str = ""
    family: str = "unknown"
    required_windows: tuple[int, ...] = ()
    lookback_window: int | None = None
    grain: DataGrain = DataGrain.ASSET_TIMESTAMP
    timing_semantics: str = "as_of_timestamp"
    null_policy: str = "preserve_nulls"
    source_layer: str = "processed"
    required_input_data_id: str = ""
    window: int | None = None
    definition_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.definition_version != "0.1":
            raise ValueError("unsupported_factor_definition_version")
        if not self.input_columns:
            raise ValueError("factor_input_columns_required")
        base_inputs = self.base_input_columns or self.input_columns
        derived_inputs = self.derived_input_columns
        expected_inputs = tuple(dict.fromkeys((*base_inputs, *derived_inputs)))
        if expected_inputs != self.input_columns:
            raise ValueError("factor_input_columns_must_match_base_and_derived")
        object.__setattr__(self, "base_input_columns", base_inputs)
        object.__setattr__(self, "derived_input_columns", derived_inputs)
        if not self.required_input_data_id:
            raise ValueError("factor_required_input_data_id_required")

    def to_manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "definition_version": self.definition_version,
            "factor_id": self.factor_id,
            "market_id": str(self.market_id),
            "factor_set": self.factor_set.value,
            "display_name": self.display_name or self.factor_id,
            "description": self.description,
            "family": self.family,
            "input_columns": list(self.input_columns),
            "base_input_columns": list(self.base_input_columns),
            "derived_input_columns": list(self.derived_input_columns),
            "output_column": self.output_column,
            "required_windows": list(self.required_windows),
            "lookback_window": self.lookback_window,
            "grain": self.grain.value,
            "timing_semantics": self.timing_semantics,
            "null_policy": self.null_policy,
            "source_layer": self.source_layer,
            "required_input_data_id": self.required_input_data_id,
        }
        if self.window is not None:
            payload["window"] = self.window
        return payload


__all__ = ["FactorDefinition", "FactorSet"]
