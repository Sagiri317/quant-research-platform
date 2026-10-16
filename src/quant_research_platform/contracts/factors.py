"""Factor definition contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from quant_research_platform.contracts.data import DataGrain, normalize_data_grain
from quant_research_platform.contracts.ids import MarketId, validate_market_id
from quant_research_platform.contracts.validation import (
    _validate_column_tuple,
    _validate_non_empty_string,
)


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

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.factor_id, "factor_id_required")
        object.__setattr__(self, "market_id", validate_market_id(self.market_id))
        try:
            factor_set = (
                self.factor_set
                if isinstance(self.factor_set, FactorSet)
                else FactorSet(self.factor_set)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported_factor_set") from exc
        object.__setattr__(self, "factor_set", factor_set)
        _validate_column_tuple(self.input_columns, "factor_input_columns")
        _validate_non_empty_string(self.output_column, "factor_output_column_required")
        _validate_non_empty_string(
            self.required_input_data_id,
            "factor_required_input_data_id_required",
        )
        for field_name in (
            "display_name",
            "description",
            "family",
            "timing_semantics",
            "null_policy",
            "source_layer",
        ):
            if type(getattr(self, field_name)) is not str:
                raise ValueError(f"factor_{field_name}_must_be_string")
        object.__setattr__(self, "grain", normalize_data_grain(self.grain))
        if (
            type(self.required_windows) is not tuple
            or any(type(window) is not int for window in self.required_windows)
            or len(set(self.required_windows)) != len(self.required_windows)
        ):
            raise ValueError("factor_required_windows_invalid")
        if any(window <= 0 for window in self.required_windows):
            raise ValueError("factor_required_windows_must_be_positive")
        if self.lookback_window is not None and (
            type(self.lookback_window) is not int or self.lookback_window <= 0
        ):
            raise ValueError("factor_lookback_window_must_be_positive")

    def to_manifest(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "factor_id": self.factor_id,
            "market_id": str(self.market_id),
            "factor_set": self.factor_set.value,
            "display_name": self.display_name or self.factor_id,
            "description": self.description,
            "family": self.family,
            "input_columns": list(self.input_columns),
            "output_column": self.output_column,
            "required_windows": list(self.required_windows),
            "lookback_window": self.lookback_window,
            "grain": self.grain.value,
            "timing_semantics": self.timing_semantics,
            "null_policy": self.null_policy,
            "source_layer": self.source_layer,
            "required_input_data_id": self.required_input_data_id,
        }
        return payload
