"""Neutrality bands are stock-leg exposure units, with explicit exact option."""

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import (
    ExecutionAssumptions,
    PortfolioTargetBlocked,
)
from quant_research_platform.research.portfolios.construction import (
    _neutrality_context,
    build_simple_target_weights,
)


@pytest.mark.parametrize("limit, expected", [(0.0, 0.5), (0.02, 0.52), (0.1, 0.6)])
def test_industry_limit_projects_to_nearest_permitted_stock_weight(limit, expected):
    weights = build_simple_target_weights(
        pd.Series({"a": 2.0, "b": 1.0}),
        eligible=pd.Series({"a": True, "b": True}),
        float_market_cap=pd.Series({"a": 100.0, "b": 100.0}),
        industry=pd.Series({"a": "A", "b": "B"}),
        neutrality="industry",
        industry_neutrality_limit=limit,
    )
    assert weights["a"] == pytest.approx(expected, abs=1e-8)
    assert weights.sum() == pytest.approx(1.0)


def test_size_band_uses_standardized_log_cap_not_currency_or_log_units():
    caps = pd.Series({"a": 100.0, "b": 200.0, "c": 400.0})
    eligible = pd.Series(True, index=caps.index)
    context = _neutrality_context(eligible, caps, None, "size", size_market_cap=caps)
    assert context["size"].mean() == pytest.approx(0.0, abs=1e-14)
    assert context["size"].std(ddof=0) == pytest.approx(1.0)
    kwargs = dict(eligible=eligible, neutrality="size", size_neutrality_limit=0.2)
    weights = build_simple_target_weights(
        pd.Series({"a": 3.0, "b": 2.0, "c": 1.0}),
        float_market_cap=caps,
        total_market_cap=caps,
        **kwargs,
    )
    rescaled = build_simple_target_weights(
        pd.Series({"a": 3.0, "b": 2.0, "c": 1.0}),
        float_market_cap=caps * 1000,
        total_market_cap=caps * 1000,
        **kwargs,
    )
    assert weights.tolist() == pytest.approx(rescaled.tolist(), abs=1e-8)
    active = weights - context.benchmark_weight
    assert abs(context["size"] @ active) <= 0.2 + 1e-8


def test_unknown_industry_does_not_renormalize_known_reference_weight():
    caps = pd.Series({"a": 50.0, "b": 45.0, "unknown": 5.0})
    eligible = pd.Series(True, index=caps.index)
    industry = pd.Series({"a": "A", "b": "B", "unknown": None})
    context = _neutrality_context(eligible, caps, industry, "industry")
    assert context.benchmark_weight.to_dict() == {"a": 0.5, "b": 0.45}
    assert context.attrs["unknown_industry_weight"] == pytest.approx(0.05)
    with pytest.raises(PortfolioTargetBlocked, match="target_neutrality_infeasible"):
        build_simple_target_weights(
            pd.Series({"a": 3.0, "b": 2.0, "unknown": 1.0}),
            eligible=eligible,
            float_market_cap=caps,
            industry=industry,
            neutrality="industry",
        )


def test_enhanced_cash_does_not_change_stock_leg_industry_neutrality():
    from quant_research_platform.research.portfolios.optimizer import (
        OptimizerConfig,
        optimize_portfolio,
    )

    frame = pd.DataFrame(
        {
            "asset_id": ["a", "b", "__cash__"],
            "current_weight": [0.3, 0.2, 0.5],
            "benchmark_weight": [0.6, 0.4, 0.0],
            "expected_alpha": [0.0, 0.0, 0.0],
            "specific_variance": [0.0, 0.0, 0.0],
            "adv": [1e6, 1e6, np.inf],
            "is_cash": [False, False, True],
            "industry:A": [1.0, 0.0, 0.0],
            "industry:B": [0.0, 1.0, 0.0],
        }
    )
    columns = ("industry:A", "industry:B")
    result = optimize_portfolio(
        frame,
        factor_covariance=pd.DataFrame(0.0, index=columns, columns=columns),
        industry_columns=columns,
        cash_column="is_cash",
        expected_transaction_cost=lambda weights: 0.0,
        nav=100.0,
        assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 1.0, (1.0,)),
        config=OptimizerConfig(
            active_name_limit=1.0,
            tracking_error_limit=1.0,
            target_turnover_limit=1.0,
            industry_active_limit=0.0,
        ),
    )
    assert result.weights.tolist() == pytest.approx([0.3, 0.2, 0.5], abs=1e-8)
