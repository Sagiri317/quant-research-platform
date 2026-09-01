"""A-share market facts and execution-binding fields."""

from __future__ import annotations

from quant_research_platform.catalog.fields import FieldCatalog, platform_field_catalog
from quant_research_platform.contracts.columns import ColumnRole, FieldSpec

_CONTEXT_FLOATS = (
    "market_type",
    "trading_status_code",
    "filling_flag",
    "open",
    "high",
    "low",
    "close",
    "adj_open",
    "adj_close",
    "prev_close",
    "volume",
    "amount",
    "after_hours_volume",
    "after_hours_amount",
    "period_return",
    "turnover",
    "limit_status",
    "total_shares",
    "a_float_shares",
    "total_market_cap",
    "float_market_cap",
)
_FINANCIAL_FLOATS = (
    "total_assets",
    "total_liabilities",
    "parent_equity",
    "cash",
    "short_term_borrowings",
    "current_portion_noncurrent_liabilities",
    "long_term_borrowings",
    "bonds_payable",
    "operating_revenue",
    "operating_cost",
    "consolidated_net_income",
    "parent_net_income",
    "operating_cash_flow",
    "capex_cash_paid",
    "beginning_total_assets",
    "beginning_parent_equity",
    "prior_year_total_assets",
    "prior_year_parent_equity",
    "prior_year_parent_net_income_ttm",
    "prior_year_operating_revenue_ttm",
    "average_total_assets",
    "average_parent_equity",
    "operating_revenue_ttm",
    "operating_cost_ttm",
    "consolidated_net_income_ttm",
    "parent_net_income_ttm",
    "operating_cash_flow_ttm",
    "capex_cash_paid_ttm",
)

FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(name="stock_name", role=ColumnRole.CONTEXT, dtype="string"),
    FieldSpec(
        name="industry_standard_id",
        role=ColumnRole.CONTEXT,
        dtype="string",
        nullable=True,
    ),
    FieldSpec(
        name="industry_level1_code",
        role=ColumnRole.CONTEXT,
        dtype="string",
        nullable=True,
    ),
    FieldSpec(
        name="industry_level1_name",
        role=ColumnRole.CONTEXT,
        dtype="string",
        nullable=True,
    ),
    *(
        FieldSpec(name=name, role=ColumnRole.CONTEXT, dtype="float64", nullable=True)
        for name in _CONTEXT_FLOATS
    ),
    FieldSpec(name="is_trading", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="is_a_share_scope", role=ColumnRole.UNIVERSE, dtype="bool"),
    FieldSpec(name="is_st", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="is_pt", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="can_enter", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(name="can_exit", role=ColumnRole.MASK, dtype="bool"),
    FieldSpec(
        name="listing_date", role=ColumnRole.TIMESTAMP, dtype="date", nullable=True
    ),
    FieldSpec(
        name="delisting_date", role=ColumnRole.TIMESTAMP, dtype="date", nullable=True
    ),
    FieldSpec(
        name="share_change_date", role=ColumnRole.TIMESTAMP, dtype="date", nullable=True
    ),
    FieldSpec(
        name="listing_age_trading_days",
        role=ColumnRole.CONTEXT,
        dtype="int64",
        nullable=True,
    ),
    *(
        FieldSpec(name=name, role=ColumnRole.CONTEXT, dtype="float64", nullable=True)
        for name in _FINANCIAL_FLOATS
    ),
)


def field_catalog() -> FieldCatalog:
    return FieldCatalog((*platform_field_catalog().list(), *FIELD_SPECS))
