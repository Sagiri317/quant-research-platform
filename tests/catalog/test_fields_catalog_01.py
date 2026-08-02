from __future__ import annotations

from quant_research_platform.catalog.fields import (
    canonical_field_names,
    platform_field_catalog,
    platform_interim_column_names,
)
from quant_research_platform.contracts.columns import ColumnRole, FieldSpec, LogicalType


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
        logical_type=LogicalType.STRING_ID,
        nullable=False,
        description="Platform market identifier.",
    )
    assert catalog.get(names.timestamp).role is ColumnRole.TIMESTAMP
    assert catalog.get(names.timestamp).logical_type is LogicalType.TIMESTAMP
    assert catalog.get(names.source_id).role is ColumnRole.SOURCE
    assert catalog.get(names.source_asset_id).role is ColumnRole.SOURCE

    for market_specific_field in ("is_st", "is_pt", "limit_status", "can_enter"):
        assert market_specific_field not in catalog.list_names()
