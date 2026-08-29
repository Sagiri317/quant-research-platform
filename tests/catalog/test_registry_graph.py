from __future__ import annotations

import pytest

from quant_research_platform.sources.csmar.catalog import source_definition
from quant_research_platform.sources.csmar.interim import interim_catalog
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
    source = source_definition()
    catalog = interim_catalog()

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
    catalog = interim_catalog()

    with pytest.raises(KeyError, match="unknown_interim_data_spec"):
        catalog.get("unknown")
