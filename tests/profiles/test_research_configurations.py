"""Explicit recipe selection and scientific configuration boundaries."""

import pytest

from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.prepare import _configured_spec


@pytest.mark.parametrize(
    "factor,sources",
    [
        ("book_to_market", {"financial_balance"}),
        ("earnings_yield", {"financial_income"}),
        ("roe", {"financial_balance", "financial_income"}),
        ("accruals", {"financial_balance", "financial_income", "financial_cashflow"}),
    ],
)
def test_selected_financial_recipe_requires_only_consumed_statements(factor, sources):
    catalog = processed_catalog()
    spec = _configured_spec(
        catalog.get("fundamental_pit"), {"factor_columns": [factor]}
    )
    assert set(spec.source_input_data_ids) == sources | {
        "trading_calendar",
        "financial_first_disclosure",
    }
    factors = _configured_spec(
        catalog.get("fundamental_factors"), {"factor_columns": [factor]}
    )
    assert set(factors.required_columns) == {
        "market_id",
        "asset_id",
        "timestamp",
        "has_legal_fundamental_pit",
        factor,
    }


def test_required_risk_factor_is_explicit_and_order_is_preserved():
    spec = _configured_spec(
        processed_catalog().get("fundamental_factors"),
        {"factor_columns": ["roe", "earnings_yield"]},
    )
    assert [key for key, role in spec.column_roles.items() if role == "factor"] == [
        "roe",
        "earnings_yield",
    ]


@pytest.mark.parametrize("factors", [[], ["roe", "roe"], ["not_a_factor"], [True]])
def test_bad_factor_selection_never_silently_removes_requested_column(factors):
    with pytest.raises(ValueError, match="factor_columns_invalid"):
        _configured_spec(
            processed_catalog().get("fundamental_factors"), {"factor_columns": factors}
        )
