"""Queryable catalogs for QRP 0.1 platform objects."""

from quant_research_platform.catalog.data_products import DataProductRegistry
from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.catalog.fields import FieldCatalog

_DEFAULT_EXPORTS = {
    "default_catalogs",
    "default_data_product_registry",
    "default_factor_catalog",
    "default_label_catalog",
}


def __getattr__(name: str) -> object:
    if name in _DEFAULT_EXPORTS:
        from quant_research_platform.catalog import defaults

        return getattr(defaults, name)
    raise AttributeError(name)


__all__ = [
    "DataProductRegistry",
    "FactorCatalog",
    "FieldCatalog",
    "default_catalogs",
    "default_data_product_registry",
    "default_factor_catalog",
    "default_label_catalog",
]
