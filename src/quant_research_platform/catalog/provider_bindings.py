"""Internal provider implementation bindings for public data products."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class BindingVisibility(StrEnum):
    INTERNAL_PRIVATE = "internal_private"


@dataclass(frozen=True)
class ProviderBinding:
    data_product_id: str
    provider_id: str
    source_id: str
    implementation_package: str
    required_source_dataset_ids: tuple[str, ...]
    visibility: BindingVisibility = BindingVisibility.INTERNAL_PRIVATE

    def __post_init__(self) -> None:
        if self.visibility is not BindingVisibility.INTERNAL_PRIVATE:
            raise ValueError("provider_binding_must_be_internal")


@dataclass
class ProviderBindingRegistry:
    _items: dict[str, ProviderBinding] = field(default_factory=dict)

    def register(self, binding: ProviderBinding) -> None:
        if binding.data_product_id in self._items:
            raise ValueError(f"duplicate_provider_binding: {binding.data_product_id}")
        self._items[binding.data_product_id] = binding

    def get(self, data_product_id: str) -> ProviderBinding:
        try:
            return self._items[data_product_id]
        except KeyError as exc:
            raise KeyError(f"unknown_provider_binding: {data_product_id}") from exc


__all__ = [
    "BindingVisibility",
    "ProviderBinding",
    "ProviderBindingRegistry",
]
