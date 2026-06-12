from __future__ import annotations

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.catalog.profiles import (
    CatalogBundle,
    local_a_share_csmar_catalogs,
)
from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.markets.implementations.a_share.definition import (
    a_share_market_definition,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_MARKET_ID,
    A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    definitions as label_definitions,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    catalog as processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.research_profiles import (
    a_share_research_dataset_catalog,
)


def test_local_profile_returns_boundary_sealed_bundle() -> None:
    catalogs = local_a_share_csmar_catalogs()

    assert isinstance(catalogs, CatalogBundle)
    assert catalogs.factors is not None
    assert catalogs.labels is not None
    assert catalogs.processed is not None
    assert catalogs.research_datasets is not None
    assert catalogs.source_interim_catalogs["csmar"] is not None
    assert not hasattr(catalogs, "data_products")
    assert not hasattr(catalogs, "provider_bindings")
    assert not hasattr(catalogs, "universes")


def test_local_profile_market_catalogs_delegate_to_a_share_owner_factories() -> None:
    catalogs = local_a_share_csmar_catalogs()
    default_factors = catalogs.factors.list()
    owner_factors = factor_definitions.a_share_factor_catalog().list()
    default_labels = catalogs.labels.list()
    owner_labels = label_definitions.a_share_label_catalog().list()

    assert [(item.factor_id, item.output_column) for item in default_factors] == [
        (item.factor_id, item.output_column) for item in owner_factors
    ]
    assert [(item.label_id, item.output_column) for item in default_labels] == [
        (item.label_id, item.output_column) for item in owner_labels
    ]


def test_local_profile_research_dataset_catalog_delegates_to_a_share_owner() -> None:
    catalogs = local_a_share_csmar_catalogs()
    default_spec = catalogs.research_datasets.get(
        A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID
    )
    owner_spec = a_share_research_dataset_catalog().get(
        A_SHARE_PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID
    )

    assert default_spec.data_id == owner_spec.data_id
    assert default_spec.input_data_ids == owner_spec.input_data_ids


def test_local_profile_processed_catalog_uses_kind_not_category() -> None:
    catalogs = local_a_share_csmar_catalogs()
    specs = catalogs.processed.list()
    kinds_by_id = {spec.data_id: spec.kind for spec in specs}

    assert kinds_by_id["daily_panel"] is ProcessedDataKind.PANEL
    assert kinds_by_id["daily_context"] is ProcessedDataKind.ANNOTATION
    assert kinds_by_id["price_volume"] is ProcessedDataKind.FACTOR
    assert kinds_by_id["forward_returns"] is ProcessedDataKind.LABEL
    assert all(not hasattr(spec, "category") for spec in specs)


def test_local_profile_internal_registries_are_platform_composed() -> None:
    catalogs = local_a_share_csmar_catalogs()
    market = catalogs.markets.get(A_SHARE_MARKET_ID)
    owner_market = a_share_market_definition()
    source = catalogs.sources.get("csmar")

    assert market.market_id == owner_market.market_id
    assert market.display_name == owner_market.display_name
    assert market.processed_spec_ids == owner_market.processed_spec_ids
    assert source.provider_id == "csmar"
    assert source.implementation_package == "quant_research_platform.sources.csmar"
    assert catalogs.reports.get(
        "evaluation_report_private"
    ).run_type == "evaluation"


def test_local_profile_source_interim_catalogs_are_platform_composed() -> None:
    source_catalogs = local_a_share_csmar_catalogs().source_interim_catalogs

    assert set(source_catalogs) == {"csmar"}
    assert isinstance(source_catalogs["csmar"], InterimDataCatalog)
    assert {
        spec.data_id for spec in source_catalogs["csmar"].list()
    } == {
        "daily_market",
        "stock_master",
        "market_cap",
        "adjustment",
        "index_info",
        "trading_calendar",
    }
    assert all(
        not hasattr(spec, "market_id")
        for spec in source_catalogs["csmar"].list()
    )


def test_a_share_processed_catalog_owner_matches_profile() -> None:
    profile_ids = {
        spec.data_id for spec in local_a_share_csmar_catalogs().processed.list()
    }
    owner_ids = {
        spec.data_id for spec in processed_catalog.a_share_processed_catalog().list()
    }

    assert profile_ids == owner_ids
