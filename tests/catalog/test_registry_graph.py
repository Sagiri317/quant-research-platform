from __future__ import annotations

import pytest

from quant_research_platform.sources.csmar.interim import interim_catalog
from tests.helpers import frequency_ids

EXPECTED_CSMAR_INTERIM_DATA_IDS = (
    "adjustment",
    "financial_balance",
    "financial_cashflow",
    "financial_income",
    "index_info",
    "index_market",
    "index_membership",
    "index_weights",
    frequency_ids.MARKET_BARS_DATA_ID,
    "market_cap",
    "stock_master",
    "trading_calendar",
)


def test_csmar_interim_catalog_uses_single_source_id() -> None:
    catalog = interim_catalog()

    assert tuple(spec.data_id for spec in catalog.list()) == (
        EXPECTED_CSMAR_INTERIM_DATA_IDS
    )
    assert {str(spec.source_id) for spec in catalog.list()} == {"csmar"}


def test_csmar_catalog_rejects_unknown_data_id() -> None:
    catalog = interim_catalog()

    with pytest.raises(KeyError, match="unknown_interim_data_spec"):
        catalog.get("unknown")
