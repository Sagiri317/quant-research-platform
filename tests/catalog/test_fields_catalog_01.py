from __future__ import annotations

import quant_research_platform.contracts.data as data_contracts
from quant_research_platform.catalog.fields import (
    canonical_field_names,
    platform_field_catalog,
    platform_interim_column_names,
)
from quant_research_platform.contracts.columns import ColumnRole, FieldSpec


def test_contracts_do_not_define_concrete_canonical_fields() -> None:
    assert not hasattr(data_contracts, "CANONICAL_INTERIM_COLUMNS")
    assert not hasattr(data_contracts, "CANONICAL_ENTITY_TIME_KEY")


def test_catalog_fields_owns_platform_canonical_fields_only() -> None:
    names = canonical_field_names()
    catalog = platform_field_catalog()

    assert platform_interim_column_names() == (
        "market_id",
        "asset_id",
        "timestamp",
        "session_date",
        "available_at",
        "source_id",
        "source_asset_id",
    )
    assert catalog.get(names.market_id) == FieldSpec(
        name="market_id",
        role=ColumnRole.KEY,
        dtype="string",
        nullable=False,
        description="Platform market identifier.",
    )
    assert catalog.get(names.timestamp).role is ColumnRole.TIMESTAMP
    assert catalog.get(names.source_id).role is ColumnRole.SOURCE
    assert catalog.get(names.source_asset_id).role is ColumnRole.SOURCE

    for market_specific_field in ("is_st", "is_pt", "limit_status", "can_enter"):
        assert market_specific_field not in catalog.list_names()
