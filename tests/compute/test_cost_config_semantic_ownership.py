from __future__ import annotations

from pathlib import Path

import pytest

from quant_research_platform.compute._portfolio_math import cost_components
from quant_research_platform.contracts.portfolios import CostConfig


def test_compute_costs_are_owned_by_cost_config() -> None:
    for path in (
        Path("src/quant_research_platform/compute/_portfolio_math.py"),
        Path("src/quant_research_platform/compute/dataframe/portfolio_helpers.py"),
        Path("src/quant_research_platform/compute/dataframe/portfolio_returns.py"),
        Path("src/quant_research_platform/compute/dataframe/turnover.py"),
        Path("src/quant_research_platform/compute/tensor/turnover.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert "weight_config.cost_bps" not in source
        assert "cost_bps:" not in source
        assert "cost_bps=" not in source

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
            buy_cost_bps=10,
            sell_cost_bps=20,
            slippage_bps=1,
            spread_bps=2,
            impact_model="custom",
        ),
    )

    assert zero["cost_model"] == "zero_cost"
    assert zero["total_cost"] == 0.0
    assert detailed["cost_model"] == "custom"
    assert detailed["total_cost"] == pytest.approx(0.001 + 0.004 + 0.0003 + 0.0006)
