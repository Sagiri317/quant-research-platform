"""The frozen 19-factor fundamental library."""

from __future__ import annotations

from quant_research_platform.catalog.factors import FactorCatalog
from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId

FUNDAMENTAL_FACTORS = (
    "earnings_yield",
    "book_to_market",
    "sales_to_price",
    "cashflow_to_price",
    "roe",
    "roa",
    "gross_profitability",
    "gross_margin",
    "cfo_to_assets",
    "accruals",
    "asset_growth_yoy",
    "revenue_growth_ttm_yoy",
    "earnings_change_to_assets",
    "parent_equity_growth_yoy",
    "capex_to_assets",
    "liabilities_to_assets",
    "debt_to_assets",
    "cash_to_assets",
    "net_debt_to_assets",
)
CORE_FUNDAMENTAL_FACTORS = (
    "earnings_yield",
    "book_to_market",
    "roe",
    "accruals",
    "asset_growth_yoy",
    "net_debt_to_assets",
)
FUNDAMENTAL_INPUT_COLUMNS = (
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
    "total_market_cap",
    "previous_total_assets",
    "previous_parent_equity",
    "previous_parent_net_income",
    "previous_operating_revenue",
)


def fundamental_factor_definitions(
    *,
    market_id: MarketId,
    required_input_data_id: str,
    factor_set: FactorSet = FactorSet.STANDARD,
) -> tuple[FactorDefinition, ...]:
    definitions = tuple(
        FactorDefinition(
            factor_id=f"fundamental_{factor}",
            market_id=market_id,
            factor_set=FactorSet.CORE
            if factor in CORE_FUNDAMENTAL_FACTORS
            else factor_set,
            input_columns=FUNDAMENTAL_INPUT_COLUMNS,
            base_input_columns=FUNDAMENTAL_INPUT_COLUMNS,
            derived_input_columns=(),
            output_column=factor,
            display_name=factor,
            description="Protocol-defined PIT fundamental factor.",
            family="fundamental",
            grain=DataGrain.ASSET_TIMESTAMP,
            timing_semantics="point_in_time_available_at",
            null_policy="preserve_nulls",
            source_layer="processed",
            required_input_data_id=required_input_data_id,
        )
        for factor in FUNDAMENTAL_FACTORS
    )
    return definitions


def fundamental_factor_catalog(
    *, market_id: MarketId, required_input_data_id: str
) -> FactorCatalog:
    return FactorCatalog(
        fundamental_factor_definitions(
            market_id=market_id,
            required_input_data_id=required_input_data_id,
        )
    )
