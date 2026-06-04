"""Factor catalog registry."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId

_WINDOW_RE = re.compile(r"_(\d+)d\b")


@dataclass
class FactorCatalog:
    _definitions: dict[str, FactorDefinition] = field(default_factory=dict)

    def register(self, definition: FactorDefinition) -> None:
        if definition.factor_id in self._definitions:
            raise ValueError(f"duplicate_factor_definition: {definition.factor_id}")
        self._definitions[definition.factor_id] = definition

    def get(self, factor_id: str) -> FactorDefinition:
        return self._definitions[factor_id]

    def list(self) -> tuple[FactorDefinition, ...]:
        return tuple(self._definitions[key] for key in sorted(self._definitions))

    def resolve_factor_set(self, value: str | FactorSet) -> FactorSet:
        if isinstance(value, FactorSet):
            return value
        try:
            return FactorSet(value)
        except ValueError as exc:
            raise ValueError("invalid_factor_set") from exc

    @classmethod
    def bootstrap_default(cls) -> FactorCatalog:
        from quant_research_platform.markets.implementations.a_share.factors.literature_price_volume import (  # noqa: E501
            exact_literature_price_volume_factor_input_columns,
            expected_literature_price_volume_factor_columns,
        )

        catalog = cls()
        for factor_id in expected_literature_price_volume_factor_columns():
            base_inputs = exact_literature_price_volume_factor_input_columns(factor_id)
            catalog.register(
                FactorDefinition(
                    factor_id=factor_id,
                    market_id=MarketId("a_share"),
                    factor_set=FactorSet.FULL,
                    input_columns=base_inputs,
                    base_input_columns=base_inputs,
                    derived_input_columns=(),
                    output_column=factor_id,
                    display_name=factor_id,
                    description=(
                        "A-share daily price-volume factor registered by the "
                        "QRP 0.1 factor catalog."
                    ),
                    family="price_volume",
                    required_windows=_required_windows(factor_id),
                    lookback_window=_lookback_window(factor_id),
                    grain=DataGrain.ASSET_TIMESTAMP,
                    timing_semantics="same_session_close_available_at_next_session",
                    null_policy="preserve_nulls_after_min_periods",
                    source_layer="processed",
                    required_processed_data_id="daily_panel",
                    window=_lookback_window(factor_id),
                )
            )
        return catalog


def _required_windows(factor_id: str) -> tuple[int, ...]:
    windows = tuple(int(match) for match in _WINDOW_RE.findall(factor_id))
    return windows


def _lookback_window(factor_id: str) -> int | None:
    windows = _required_windows(factor_id)
    if not windows:
        return None
    return max(windows)


__all__ = ["FactorCatalog"]
