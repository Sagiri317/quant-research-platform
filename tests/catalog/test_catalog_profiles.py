from __future__ import annotations

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.contracts.data import ProcessedDataKind
from quant_research_platform.markets.implementations.a_share.definition import (
    market_definition,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    MARKET_ID,
    PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
)
from quant_research_platform.markets.implementations.a_share.labels import (
    definitions as label_definitions,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    catalog as processed_catalog,
)
from quant_research_platform.markets.implementations.a_share.research_profiles import (
    build_research_dataset_catalog,
)
from quant_research_platform.profiles.research import (
    ResearchProfile,
    research_profile,
)
from tests.helpers import frequency_ids


def test_local_profile_returns_boundary_sealed_bundle() -> None:
    catalogs = research_profile()

    assert isinstance(catalogs, ResearchProfile)
    assert catalogs.factors is not None
    assert catalogs.labels is not None
    assert catalogs.processed is not None
    assert catalogs.research_datasets is not None
    assert catalogs.source_interim_catalogs["csmar"] is not None


def test_local_profile_market_catalogs_delegate_to_a_share_owner_factories() -> None:
    catalogs = research_profile()
    profile_factors = catalogs.factors.list()
    owner_factors = factor_definitions.factor_catalog().list()
    profile_labels = catalogs.labels.list()
    owner_labels = label_definitions.label_catalog().list()

    assert [(item.factor_id, item.output_column) for item in profile_factors] == [
        (item.factor_id, item.output_column) for item in owner_factors
    ]
    assert [(item.label_id, item.output_column) for item in profile_labels] == [
        (item.label_id, item.output_column) for item in owner_labels
    ]


def test_local_profile_research_dataset_catalog_delegates_to_a_share_owner() -> None:
    catalogs = research_profile()
    profile_spec = catalogs.research_datasets.get(
        PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID
    )
    owner_spec = build_research_dataset_catalog().get(
        PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID
    )

    assert profile_spec.data_id == owner_spec.data_id
    assert profile_spec.input_data_ids == owner_spec.input_data_ids


def test_local_profile_processed_catalog_uses_kind_not_category() -> None:
    catalogs = research_profile()
    specs = catalogs.processed.list()
    kinds_by_id = {spec.data_id: spec.kind for spec in specs}

    assert kinds_by_id[frequency_ids.MARKET_PANEL_DATA_ID] is ProcessedDataKind.PANEL
    assert (
        kinds_by_id[frequency_ids.MARKET_CONTEXT_DATA_ID]
        is ProcessedDataKind.ANNOTATION
    )
    assert kinds_by_id["price_volume"] is ProcessedDataKind.FACTOR
    assert kinds_by_id["forward_returns"] is ProcessedDataKind.LABEL


def test_local_profile_internal_registries_are_platform_composed() -> None:
    catalogs = research_profile()
    market = catalogs.markets.get(MARKET_ID)
    owner_market = market_definition()
    source = catalogs.sources.get("csmar")

    assert market.market_id == owner_market.market_id
    assert market.display_name == owner_market.display_name
    assert market.processed_spec_ids == owner_market.processed_spec_ids
    assert source.provider_id == "csmar"
    assert source.implementation_package == "quant_research_platform.sources.csmar"
    assert catalogs.outputs.get("evaluation_outputs").run_type == "evaluation"


def test_local_profile_source_interim_catalogs_are_platform_composed() -> None:
    source_catalogs = research_profile().source_interim_catalogs

    assert set(source_catalogs) == {"csmar"}
    assert isinstance(source_catalogs["csmar"], InterimDataCatalog)
    assert {spec.data_id for spec in source_catalogs["csmar"].list()} == {
        frequency_ids.MARKET_BARS_DATA_ID,
        "stock_master",
        "market_cap",
        "adjustment",
        "index_info",
        "trading_calendar",
    }


def test_processed_catalog_owner_matches_profile() -> None:
    profile_ids = {spec.data_id for spec in research_profile().processed.list()}
    owner_ids = {spec.data_id for spec in processed_catalog.processed_catalog().list()}

    assert profile_ids == owner_ids
