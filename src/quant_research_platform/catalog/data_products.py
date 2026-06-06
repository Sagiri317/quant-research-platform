"""Data product registry."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.data_products import DataProductDefinition


@dataclass
class DataProductRegistry:
    _items: dict[str, DataProductDefinition] = field(default_factory=dict)

    def register(self, definition: DataProductDefinition) -> None:
        if definition.data_product_id in self._items:
            raise ValueError(
                f"duplicate_data_product_definition: {definition.data_product_id}"
            )
        self._items[definition.data_product_id] = definition

    def get(self, data_product_id: str) -> DataProductDefinition:
        try:
            return self._items[data_product_id]
        except KeyError as exc:
            raise KeyError(f"unknown_data_product_id: {data_product_id}") from exc

    def list(self) -> tuple[DataProductDefinition, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["DataProductRegistry"]
