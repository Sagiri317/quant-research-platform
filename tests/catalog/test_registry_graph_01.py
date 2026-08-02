from __future__ import annotations

import pytest

from quant_research_platform.catalog.processed import InterimDataCatalog
from quant_research_platform.profiles.research import research_profile
from tests.helpers import frequency_ids

EXPECTED_CSMAR_INTERIM_DATA_IDS = (
    "adjustment",
    "index_info",
    frequency_ids.MARKET_BARS_DATA_ID,
    "market_cap",
    "stock_master",
    "trading_calendar",
)


def test_csmar_source_interim_catalog_graph_closes_without_provider_binding() -> None:
    catalogs = research_profile()
    source = catalogs.sources.get("csmar")
    catalog = catalogs.source_interim_catalogs["csmar"]

    assert isinstance(catalog, InterimDataCatalog)
    assert str(source.source_id) == "csmar"
    assert source.provider_id == "csmar"
    assert source.implementation_package == "quant_research_platform.sources.csmar"
    assert source.visibility.value == "internal_private"
    assert not source.public_identifier
    assert tuple(spec.data_id for spec in catalog.list()) == (
        EXPECTED_CSMAR_INTERIM_DATA_IDS
    )
    assert {str(spec.source_id) for spec in catalog.list()} == {"csmar"}


def test_csmar_catalog_rejects_unknown_data_id() -> None:
    catalog = research_profile().source_interim_catalogs["csmar"]

    with pytest.raises(KeyError, match="unknown_interim_data_spec"):
        catalog.get("unknown")


def test_csmar_catalog_exposes_current_source_owned_data_ids() -> None:
    csmar_catalog_ids = {
        spec.data_id
        for spec in research_profile().source_interim_catalogs["csmar"].list()
    }

    assert csmar_catalog_ids == set(EXPECTED_CSMAR_INTERIM_DATA_IDS)
