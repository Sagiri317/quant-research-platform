from __future__ import annotations

import pytest

from quant_research_platform.catalog.data_products import DataProductRegistry
from quant_research_platform.catalog.provider_bindings import (
    BindingVisibility,
    ProviderBinding,
    ProviderBindingRegistry,
)
from quant_research_platform.catalog.sources import SourceDefinition, SourceRegistry
from quant_research_platform.contracts.ids import SourceId
from quant_research_platform.sources.csmar.catalog import csmar_interim_catalog

EXPECTED_CSMAR_SOURCE_DATASET_IDS = (
    "daily_market",
    "stock_master",
    "market_cap",
    "adjustment",
    "index_market",
    "index_info",
    "benchmark_index_market",
)


def test_data_product_provider_source_registry_graph_closes() -> None:
    data_product = DataProductRegistry.bootstrap_default().get("a_share_daily_bar")
    binding = ProviderBindingRegistry.bootstrap_default().get(
        data_product.data_product_id
    )
    source = SourceRegistry.bootstrap_default().get(binding.source_id)

    assert data_product.interim_spec_ids == ("a_share_daily_bar_interim",)
    assert binding.provider_id == "csmar"
    assert binding.source_id == "csmar"
    assert binding.visibility is BindingVisibility.INTERNAL_PRIVATE
    assert binding.required_source_dataset_ids == EXPECTED_CSMAR_SOURCE_DATASET_IDS
    assert str(source.source_id) == "csmar"
    assert source.provider_id == "csmar"
    assert source.implementation_package == "quant_research_platform.sources.csmar"
    assert not source.public_identifier

    csmar_catalog_ids = _csmar_catalog_ids()
    assert set(binding.required_source_dataset_ids).issubset(csmar_catalog_ids)
    assert set(data_product.interim_spec_ids).isdisjoint(
        binding.required_source_dataset_ids
    )

    from quant_research_platform.catalog.registry_validation import (
        validate_data_product_source_graph,
    )

    result = validate_data_product_source_graph(data_product.data_product_id)

    assert result.data_product_id == "a_share_daily_bar"
    assert result.product_interim_output_ids == ("a_share_daily_bar_interim",)
    assert result.source_id == "csmar"
    assert result.required_source_dataset_ids == EXPECTED_CSMAR_SOURCE_DATASET_IDS


def test_registry_graph_validation_rejects_unknown_source_dataset_id() -> None:
    from quant_research_platform.catalog.registry_validation import (
        CatalogConsistencyError,
        validate_data_product_source_graph,
    )

    bindings = ProviderBindingRegistry()
    bindings.register(
        ProviderBinding(
            data_product_id="a_share_daily_bar",
            provider_id="csmar",
            source_id="csmar",
            implementation_package="quant_research_platform.sources.csmar",
            required_source_dataset_ids=("daily_market", "fake_table"),
        )
    )
    sources = SourceRegistry()
    sources.register(
        SourceDefinition(
            source_id=SourceId("csmar"),
            provider_id="csmar",
            display_name="CSMAR source adapter",
            implementation_package="quant_research_platform.sources.csmar",
        )
    )

    with pytest.raises(CatalogConsistencyError) as exc_info:
        validate_data_product_source_graph(
            "a_share_daily_bar",
            data_products=DataProductRegistry.bootstrap_default(),
            provider_bindings=bindings,
            sources=sources,
            source_catalogs={"csmar": csmar_interim_catalog()},
        )

    assert exc_info.value.error_code == "catalog_registry_graph_invalid"
    assert str(exc_info.value) == "catalog_registry_graph_invalid"


def test_source_registry_declares_source_identity_not_internal_dataset_list() -> None:
    source = SourceRegistry.bootstrap_default().get("csmar")

    assert str(source.source_id) == "csmar"
    assert source.provider_id == "csmar"
    assert source.implementation_package == "quant_research_platform.sources.csmar"
    assert source.visibility.value == "internal_private"
    assert not source.public_identifier
    assert not hasattr(source, "interim_spec_ids")


def test_registry_graph_does_not_depend_on_stale_source_dataset_id() -> None:
    stale_id = "_".join(("csmar", "daily", "market", "interim"))
    binding = ProviderBindingRegistry.bootstrap_default().get("a_share_daily_bar")
    source = SourceRegistry.bootstrap_default().get("csmar")

    assert stale_id not in binding.required_source_dataset_ids
    assert not hasattr(source, "interim_spec_ids")
    assert stale_id not in _csmar_catalog_ids()


def _csmar_catalog_ids() -> set[str]:
    return {spec.interim_data_id for spec in csmar_interim_catalog().list()}
