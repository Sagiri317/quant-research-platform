"""Public data product contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.ids import MarketId


class BindingVisibility(StrEnum):
    INTERNAL_PRIVATE = "internal_private"


@dataclass(frozen=True)
class ProviderBinding:
    provider_id: str
    implementation_package: str
    visibility: BindingVisibility = BindingVisibility.INTERNAL_PRIVATE


@dataclass(frozen=True)
class DataProductDefinition:
    data_product_id: str
    market_id: MarketId
    grain: DataGrain
    interim_spec_ids: tuple[str, ...]
    provider_binding: ProviderBinding
    public_identifier: bool = True
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_data_product_contract_version")
        if not self.public_identifier:
            raise ValueError("data_product_must_be_public_identifier")
        if self.provider_binding.visibility != BindingVisibility.INTERNAL_PRIVATE:
            raise ValueError("data_product_provider_binding_must_be_internal")

    def to_public_record(self) -> dict[str, Any]:
        return {
            "data_product_id": self.data_product_id,
            "market_id": str(self.market_id),
            "contract_version": self.contract_version,
            "grain": self.grain.value,
            "public_identifier": self.public_identifier,
        }


__all__ = [
    "BindingVisibility",
    "DataProductDefinition",
    "ProviderBinding",
]
