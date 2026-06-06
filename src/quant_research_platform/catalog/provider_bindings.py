"""Internal provider implementation bindings for public data products."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


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


def private_interim_root_for_data_product(
    project_root: Path,
    data_product_id: str,
) -> Path:
    """Return the local private interim root for an internal binding."""

    from quant_research_platform.catalog.defaults import (
        default_provider_binding_registry,
    )

    binding = default_provider_binding_registry().get(data_product_id)
    return Path(project_root) / "data" / "interim" / binding.provider_id


__all__ = [
    "BindingVisibility",
    "ProviderBinding",
    "ProviderBindingRegistry",
    "private_interim_root_for_data_product",
]
