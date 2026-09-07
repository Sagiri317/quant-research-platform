"""Pure common fundamental factor mathematics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import strict_finite_numeric_series

from .definitions import FUNDAMENTAL_FACTORS, FUNDAMENTAL_INPUT_COLUMNS


def calculate_fundamental_factors(
    panel: pd.DataFrame,
    *,
    factor_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Calculate the 19 registered factors from PIT/derived accounting facts."""

    requested = tuple(dict.fromkeys(factor_columns or FUNDAMENTAL_FACTORS))
    unknown = sorted(set(requested).difference(FUNDAMENTAL_FACTORS))
    if unknown:
        raise ValueError("unknown_fundamental_factor: " + ", ".join(unknown))
    required = {"asset_id", "timestamp"}
    if not required.issubset(panel.columns):
        raise ValueError("fundamental_factor_keys_invalid")
    if not set(FUNDAMENTAL_INPUT_COLUMNS).issubset(panel.columns):
        raise ValueError("fundamental_factor_input_columns_missing")
    result = panel.loc[:, ("asset_id", "timestamp")].copy()
    values = {
        column: strict_finite_numeric_series(
            panel[column], error=f"invalid_fundamental_value: {column}", allow_null=True
        ).astype("float64")
        for column in FUNDAMENTAL_INPUT_COLUMNS
    }

    assets = values["total_assets"]
    prior_assets = values["prior_year_total_assets"]
    avg_assets = values["average_total_assets"]
    parent_equity = values["parent_equity"]
    prior_equity = values["prior_year_parent_equity"]
    avg_equity = values["average_parent_equity"]
    market_cap = values["total_market_cap"]
    revenue = values["operating_revenue_ttm"]
    prior_revenue = values["prior_year_operating_revenue_ttm"]
    operating_cost = values["operating_cost_ttm"]
    consolidated_income = values["consolidated_net_income_ttm"]
    parent_income = values["parent_net_income_ttm"]
    prior_parent_income = values["prior_year_parent_net_income_ttm"]
    cfo = values["operating_cash_flow_ttm"]
    capex = values["capex_cash_paid_ttm"]

    gross_profit = _finite(revenue - operating_cost)
    debt = _finite(
        values["short_term_borrowings"]
        + values["current_portion_noncurrent_liabilities"]
        + values["long_term_borrowings"]
        + values["bonds_payable"]
    )
    cash = values["cash"]
    net_debt = _finite(debt - cash)

    formulas = {
        "earnings_yield": _ratio(parent_income, market_cap),
        "book_to_market": _ratio(parent_equity, market_cap),
        "sales_to_price": _ratio(revenue, market_cap),
        "cashflow_to_price": _ratio(cfo, market_cap),
        "roe": _ratio(parent_income, avg_equity),
        "roa": _ratio(consolidated_income, avg_assets),
        "gross_profitability": _ratio(gross_profit, avg_assets),
        "gross_margin": _ratio(gross_profit, revenue),
        "cfo_to_assets": _ratio(cfo, avg_assets),
        "accruals": _ratio(_finite(consolidated_income - cfo), avg_assets),
        "asset_growth_yoy": _ratio(assets, prior_assets) - 1.0,
        "revenue_growth_ttm_yoy": _ratio(revenue, prior_revenue) - 1.0,
        "earnings_change_to_assets": _ratio(
            _finite(parent_income - prior_parent_income), avg_assets
        ),
        "parent_equity_growth_yoy": _ratio(parent_equity, prior_equity) - 1.0,
        "capex_to_assets": _ratio(capex.where(capex.ge(0.0)), avg_assets),
        "liabilities_to_assets": _ratio(values["total_liabilities"], assets),
        "debt_to_assets": _ratio(debt, assets),
        "cash_to_assets": _ratio(cash, assets),
        "net_debt_to_assets": _ratio(net_debt, assets),
    }
    for column in requested:
        result[column] = _finite(formulas[column]).astype("float64")
    return result


def _ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    valid_denominator = _finite(denominator).where(_finite(denominator).gt(0.0))
    return _finite(numerator).div(valid_denominator)


def _finite(values: pd.Series) -> pd.Series:
    return values.replace([np.inf, -np.inf], np.nan)
