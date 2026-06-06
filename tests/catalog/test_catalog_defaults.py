from __future__ import annotations

from quant_research_platform.catalog.defaults import (
    PlatformCatalogs,
    default_catalogs,
    default_data_product_registry,
    default_factor_catalog,
    default_label_catalog,
    default_market_registry,
    default_provider_binding_registry,
    default_report_catalog,
    default_research_dataset_catalog,
    default_source_interim_catalogs,
    default_source_registry,
    default_universe_catalog,
)
from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.catalog.registry_validation import (
    validate_data_product_source_graph,
)
from quant_research_platform.markets.implementations.a_share.catalog import (
    a_share_daily_bar_data_product_definition,
    a_share_default_universe_definition,
    a_share_factor_catalog,
    a_share_label_catalog,
    a_share_market_definition,
    a_share_research_dataset_catalog,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_DAILY_BAR_DATA_PRODUCT_ID,
    A_SHARE_DEFAULT_UNIVERSE_ID,
    A_SHARE_MARKET_ID,
    A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
)


def test_default_catalogs_returns_platform_catalog_bundle() -> None:
    catalogs = default_catalogs()

    assert isinstance(catalogs, PlatformCatalogs)
    assert catalogs.factors is not None
    assert catalogs.labels is not None
    assert catalogs.data_products is not None
    assert catalogs.source_interim_catalogs["csmar"] is not None
    assert catalogs.universes.get("a_share_default").membership_column == (
        "eligible"
    )


def test_default_market_catalogs_delegate_to_a_share_owner_factories() -> None:
    default_factors = default_factor_catalog().list()
    owner_factors = a_share_factor_catalog().list()
    default_labels = default_label_catalog().list()
    owner_labels = a_share_label_catalog().list()

    assert [(item.factor_id, item.output_column) for item in default_factors] == [
        (item.factor_id, item.output_column) for item in owner_factors
    ]
    assert [(item.label_id, item.output_column) for item in default_labels] == [
        (item.label_id, item.output_column) for item in owner_labels
    ]


def test_default_research_dataset_catalog_delegates_to_a_share_owner() -> None:
    default_spec = default_research_dataset_catalog().get(
        A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID
    )
    owner_spec = a_share_research_dataset_catalog().get(
        A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID
    )

    assert default_spec.dataset_id == owner_spec.dataset_id


def test_default_internal_registries_are_composed_by_platform_defaults() -> None:
    data_product = default_data_product_registry().get(
        A_SHARE_DAILY_BAR_DATA_PRODUCT_ID
    )
    owner_data_product = a_share_daily_bar_data_product_definition()
    market = default_market_registry().get(A_SHARE_MARKET_ID)
    owner_market = a_share_market_definition()
    universe = default_universe_catalog().get(A_SHARE_DEFAULT_UNIVERSE_ID)
    owner_universe = a_share_default_universe_definition()

    assert data_product.data_product_id == owner_data_product.data_product_id
    assert data_product.market_id == owner_data_product.market_id
    assert data_product.grain == owner_data_product.grain
    assert data_product.interim_spec_ids == owner_data_product.interim_spec_ids
    assert default_provider_binding_registry().get(
        A_SHARE_DAILY_BAR_DATA_PRODUCT_ID
    ).source_id == "csmar"
    assert default_source_registry().get("csmar").provider_id == "csmar"
    assert market.market_id == owner_market.market_id
    assert market.display_name == owner_market.display_name
    assert market.processed_spec_ids == owner_market.processed_spec_ids
    assert universe.universe_id == owner_universe.universe_id
    assert universe.market_id == owner_universe.market_id
    assert universe.membership_column == owner_universe.membership_column
    assert default_report_catalog().get(
        "evaluation_report_private"
    ).run_type == "evaluation"


def test_default_source_and_universe_catalogs_are_platform_composed() -> None:
    source_catalogs = default_source_interim_catalogs()
    universe = default_universe_catalog().get("a_share_default")

    assert set(source_catalogs) == {"csmar"}
    assert isinstance(source_catalogs["csmar"], InterimDataCatalog)
    assert universe.membership_column == "eligible"


def test_registry_graph_validation_uses_default_catalogs() -> None:
    result = validate_data_product_source_graph("a_share_daily_bar")

    assert result.data_product_id == "a_share_daily_bar"
    assert result.source_id == "csmar"
