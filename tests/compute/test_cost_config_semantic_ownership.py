from __future__ import annotations

import pytest

from quant_research_platform.compute._portfolio_math import cost_components
from quant_research_platform.contracts.portfolios import CostConfig


def test_compute_costs_are_owned_by_cost_config() -> None:
    zero = cost_components(
        buy_turnover=1.0,
        sell_turnover=1.0,
        turnover=2.0,
        cost_config=CostConfig(),
    )
    detailed = cost_components(
        buy_turnover=1.0,
        sell_turnover=2.0,
        turnover=3.0,
        cost_config=CostConfig(
            commission_bps=10,
            slippage_bps=1,
            impact_bps=2,
        ),
    )
    assert zero["cost_model"] == "commission_slippage_sqrt_impact"
    assert zero["total_cost"] == 0.0
    assert detailed["cost_model"] == "commission_slippage_sqrt_impact"
    assert detailed["total_cost"] == pytest.approx(0.0033)
