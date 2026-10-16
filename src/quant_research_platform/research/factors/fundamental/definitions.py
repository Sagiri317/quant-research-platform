"""The frozen 19-factor fundamental library."""

from __future__ import annotations

from quant_research_platform.contracts.data import DataGrain
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId

CORE_FUNDAMENTAL_FACTORS = (
    "earnings_yield",
    "book_to_market",
    "roe",
    "accruals",
    "asset_growth_yoy",
    "net_debt_to_assets",
)
# Signed numerator terms, positive denominator, final growth adjustment.
FUNDAMENTAL_FORMULAS = {
    "earnings_yield": (("parent_net_income_ttm",), "total_market_cap", 0),
    "book_to_market": (("parent_equity",), "total_market_cap", 0),
    "sales_to_price": (("operating_revenue_ttm",), "total_market_cap", 0),
    "cashflow_to_price": (("operating_cash_flow_ttm",), "total_market_cap", 0),
    "roe": (("parent_net_income_ttm",), "average_parent_equity", 0),
    "roa": (("consolidated_net_income_ttm",), "average_total_assets", 0),
    "gross_profitability": (
        ("operating_revenue_ttm", "-operating_cost_ttm"),
        "average_total_assets",
        0,
    ),
    "gross_margin": (
        ("operating_revenue_ttm", "-operating_cost_ttm"),
        "operating_revenue_ttm",
        0,
    ),
    "cfo_to_assets": (("operating_cash_flow_ttm",), "average_total_assets", 0),
    "accruals": (
        ("consolidated_net_income_ttm", "-operating_cash_flow_ttm"),
        "average_total_assets",
        0,
    ),
    "asset_growth_yoy": (("total_assets",), "prior_year_total_assets", -1),
    "revenue_growth_ttm_yoy": (
        ("operating_revenue_ttm",),
        "prior_year_operating_revenue_ttm",
        -1,
    ),
    "earnings_change_to_assets": (
        ("parent_net_income_ttm", "-prior_year_parent_net_income_ttm"),
        "average_total_assets",
        0,
    ),
    "parent_equity_growth_yoy": (("parent_equity",), "prior_year_parent_equity", -1),
    "capex_to_assets": (("capex_cash_paid_ttm",), "average_total_assets", 0),
    "liabilities_to_assets": (("total_liabilities",), "total_assets", 0),
    "debt_to_assets": (
        (
            "short_term_borrowings",
            "current_portion_noncurrent_liabilities",
            "long_term_borrowings",
            "bonds_payable",
        ),
        "total_assets",
        0,
    ),
    "cash_to_assets": (("cash",), "total_assets", 0),
    "net_debt_to_assets": (
        (
            "short_term_borrowings",
            "current_portion_noncurrent_liabilities",
            "long_term_borrowings",
            "bonds_payable",
            "-cash",
        ),
        "total_assets",
        0,
    ),
}


FUNDAMENTAL_FACTORS = tuple(FUNDAMENTAL_FORMULAS)


def fundamental_inputs(factor: str) -> tuple[str, ...]:
    terms, denominator, _ = FUNDAMENTAL_FORMULAS[factor]
    return tuple(
        dict.fromkeys((*[term.removeprefix("-") for term in terms], denominator))
    )


FUNDAMENTAL_INPUT_COLUMNS = tuple(
    dict.fromkeys(
        column
        for factor in FUNDAMENTAL_FACTORS
        for column in fundamental_inputs(factor)
    )
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
            input_columns=fundamental_inputs(factor),
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
