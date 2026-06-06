"""Internal catalog graph consistency validation."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass

from quant_research_platform.catalog.data_products import DataProductRegistry
from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.catalog.provider_bindings import (
    ProviderBindingRegistry,
)
from quant_research_platform.catalog.sources import SourceRegistry


class CatalogConsistencyError(ValueError):
    error_code = "catalog_registry_graph_invalid"

    def __init__(self) -> None:
        super().__init__(self.error_code)


@dataclass(frozen=True)
class RegistryGraphValidationResult:
    data_product_id: str
    product_interim_output_ids: tuple[str, ...]
    source_id: str
    required_source_dataset_ids: tuple[str, ...]


def validate_data_product_source_graph(
    data_product_id: str,
    *,
    data_products: DataProductRegistry | None = None,
    provider_bindings: ProviderBindingRegistry | None = None,
    sources: SourceRegistry | None = None,
    source_catalogs: dict[str, InterimDataCatalog] | None = None,
) -> RegistryGraphValidationResult:
    """Validate the internal product -> source dataset graph."""

    data_product_registry = data_products or _default_data_product_registry()
    binding_registry = provider_bindings or _default_provider_binding_registry()
    source_registry = sources or _default_source_registry()
    catalogs = source_catalogs if source_catalogs is not None else _default_catalogs()

    try:
        data_product = data_product_registry.get(data_product_id)
        binding = binding_registry.get(data_product.data_product_id)
        source = source_registry.get(binding.source_id)
    except (KeyError, ValueError) as exc:
        raise CatalogConsistencyError() from exc

    if binding.data_product_id != data_product.data_product_id:
        raise CatalogConsistencyError()
    if binding.provider_id != source.provider_id:
        raise CatalogConsistencyError()
    if binding.implementation_package != source.implementation_package:
        raise CatalogConsistencyError()
    if not _is_importable(source.implementation_package):
        raise CatalogConsistencyError()
    if not binding.required_source_dataset_ids:
        raise CatalogConsistencyError()

    source_catalog = catalogs.get(binding.source_id) or _load_interim_catalog(
        binding.source_id,
        source.implementation_package,
    )
    source_dataset_ids = {spec.interim_data_id for spec in source_catalog.list()}
    if not set(binding.required_source_dataset_ids).issubset(source_dataset_ids):
        raise CatalogConsistencyError()
    if set(data_product.interim_spec_ids).intersection(
        binding.required_source_dataset_ids
    ):
        raise CatalogConsistencyError()

    return RegistryGraphValidationResult(
        data_product_id=data_product.data_product_id,
        product_interim_output_ids=data_product.interim_spec_ids,
        source_id=binding.source_id,
        required_source_dataset_ids=binding.required_source_dataset_ids,
    )


def _is_importable(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def _load_interim_catalog(
    source_id: str,
    implementation_package: str,
) -> InterimDataCatalog:
    try:
        module = importlib.import_module(f"{implementation_package}.catalog")
        factory = getattr(module, f"{source_id}_interim_catalog")
        catalog = factory()
    except (AttributeError, ModuleNotFoundError) as exc:
        raise CatalogConsistencyError() from exc
    if not isinstance(catalog, InterimDataCatalog):
        raise CatalogConsistencyError()
    return catalog


def _default_data_product_registry() -> DataProductRegistry:
    from quant_research_platform.catalog.defaults import (
        default_data_product_registry,
    )

    return default_data_product_registry()


def _default_provider_binding_registry() -> ProviderBindingRegistry:
    from quant_research_platform.catalog.defaults import (
        default_provider_binding_registry,
    )

    return default_provider_binding_registry()


def _default_source_registry() -> SourceRegistry:
    from quant_research_platform.catalog.defaults import default_source_registry

    return default_source_registry()


def _default_catalogs() -> dict[str, InterimDataCatalog]:
    from quant_research_platform.catalog.defaults import default_source_interim_catalogs

    return default_source_interim_catalogs()


__all__ = [
    "CatalogConsistencyError",
    "RegistryGraphValidationResult",
    "validate_data_product_source_graph",
]
