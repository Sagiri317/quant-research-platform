"""Label catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.labels import LabelDefinition


@dataclass
class LabelCatalog:
    _items: dict[str, LabelDefinition] = field(default_factory=dict)

    def register(self, definition: LabelDefinition, *, replace: bool = False) -> None:
        if definition.label_id in self._items and not replace:
            raise ValueError(f"duplicate_label_definition: {definition.label_id}")
        self._items[definition.label_id] = definition

    def list(self) -> tuple[LabelDefinition, ...]:
        return tuple(self._items[key] for key in sorted(self._items))

    @classmethod
    def bootstrap_default(cls) -> LabelCatalog:
        catalog = cls()
        for horizon in (1, 5, 10, 20, 60, 120):
            catalog.register(
                LabelDefinition(
                    label_id=f"forward_return_{horizon}d",
                    market_id=MarketId("a_share"),
                    output_column=f"forward_return_{horizon}d",
                    label_timing="next_open_to_horizon_open",
                    horizon=horizon,
                    lookahead=horizon + 1,
                    entry_rule="next_session_open",
                    exit_rule="horizon_session_open",
                )
            )
        return catalog


__all__ = ["LabelCatalog"]
