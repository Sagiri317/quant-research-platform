from __future__ import annotations

import pytest

from quant_research_platform.catalog.defaults import (
    default_source_interim_catalogs,
    default_source_registry,
)
from quant_research_platform.catalog.processed import InterimDataCatalog

EXPECTED_CSMAR_INTERIM_DATA_IDS = (
    "adjustment",
    "daily_market",
    "index_info",
    "market_cap",
    "stock_master",
    "trading_calendar",
)


def test_csmar_source_interim_catalog_graph_closes_without_provider_binding() -> None:
    source = default_source_registry().get("csmar")
    catalog = default_source_interim_catalogs()["csmar"]

    assert isinstance(catalog, InterimDataCatalog)
    assert str(source.source_id) == "csmar"
    assert source.provider_id == "csmar"
    assert source.implementation_package == "quant_research_platform.sources.csmar"
    assert source.visibility.value == "internal_private"
    assert not source.public_identifier
    assert not hasattr(source, "interim_spec_ids")
    assert tuple(spec.data_id for spec in catalog.list()) == (
        EXPECTED_CSMAR_INTERIM_DATA_IDS
    )
    assert {str(spec.source_id) for spec in catalog.list()} == {"csmar"}


def test_csmar_source_interim_specs_have_source_not_market_identity() -> None:
    for spec in default_source_interim_catalogs()["csmar"].list():
        assert hasattr(spec, "data_id")
        assert hasattr(spec, "source_id")
        assert not hasattr(spec, "market_id")
        assert not hasattr(spec, "interim_data_id")


def test_csmar_catalog_rejects_retired_interim_data_id() -> None:
    catalog = default_source_interim_catalogs()["csmar"]

    with pytest.raises(KeyError, match="unknown_interim_data_spec"):
        catalog.get("_".join(("csmar", "daily", "market", "interim")))


def test_csmar_catalog_exposes_current_source_owned_data_ids() -> None:
    csmar_catalog_ids = {
        spec.data_id for spec in default_source_interim_catalogs()["csmar"].list()
    }

    assert csmar_catalog_ids == set(EXPECTED_CSMAR_INTERIM_DATA_IDS)
    assert "index_market" not in csmar_catalog_ids
    assert "a_share_daily_bar_interim" not in csmar_catalog_ids
