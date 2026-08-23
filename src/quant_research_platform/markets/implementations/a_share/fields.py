"""A-share market-specific field catalog."""

from __future__ import annotations

from quant_research_platform.catalog.fields import FieldCatalog, platform_field_catalog
from quant_research_platform.contracts.columns import ColumnRole, FieldSpec

FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(name="stock_name", role=ColumnRole.CONTEXT, dtype="string"),
    FieldSpec(name="market_type", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="trading_status_code", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="filling_flag", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="open", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="high", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="low", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="close", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="adj_open", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="adj_close", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="prev_close", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="volume", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="amount", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="turnover", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="period_return", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="limit_status", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="is_limit_up", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="is_limit_down", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="is_trading", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="is_st", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="is_pt", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="is_a_share_scope", role=ColumnRole.UNIVERSE, dtype="bool"),
    FieldSpec(name="eligible_base", role=ColumnRole.UNIVERSE, dtype="bool"),
    FieldSpec(name="eligible", role=ColumnRole.UNIVERSE, dtype="bool"),
    FieldSpec(name="can_enter", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="can_exit", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="listing_date", role=ColumnRole.TIMESTAMP, dtype="date"),
    FieldSpec(name="delisting_date", role=ColumnRole.TIMESTAMP, dtype="date"),
    FieldSpec(name="company_status", role=ColumnRole.CONTEXT, dtype="string"),
    FieldSpec(name="status_change_date", role=ColumnRole.TIMESTAMP, dtype="date"),
    FieldSpec(name="listing_age_days", role=ColumnRole.CONTEXT, dtype="int64"),
    FieldSpec(
        name="listing_age_trading_days",
        role=ColumnRole.CONTEXT,
        dtype="int64",
    ),
    FieldSpec(
        name="eligible_after_listing_age",
        role=ColumnRole.UNIVERSE,
        dtype="bool",
    ),
    FieldSpec(name="eligible_liquidity", role=ColumnRole.UNIVERSE, dtype="bool"),
    FieldSpec(
        name="csrc_2012_industry_code",
        role=ColumnRole.CONTEXT,
        dtype="string",
        nullable=True,
    ),
    FieldSpec(
        name="csrc_2012_industry_name",
        role=ColumnRole.CONTEXT,
        dtype="string",
        nullable=True,
    ),
    FieldSpec(name="share_change_date", role=ColumnRole.TIMESTAMP, dtype="date"),
    FieldSpec(name="total_shares", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="a_float_shares", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="total_market_cap", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="float_market_cap", role=ColumnRole.CONTEXT, dtype="float64"),
    FieldSpec(name="next_bar_is_trading", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(
        name="next_bar_limit_status",
        role=ColumnRole.CONTEXT,
        dtype="float64",
    ),
)


def field_catalog() -> FieldCatalog:
    """Return platform canonical fields plus A-share-only processed fields."""

    return FieldCatalog((*platform_field_catalog().list(), *FIELD_SPECS))
