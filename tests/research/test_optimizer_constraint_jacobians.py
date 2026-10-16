"""Independent checks for exact optimizer constraint derivatives."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.contracts.portfolios import ExecutionAssumptions
from quant_research_platform.research.portfolios import optimizer


def _capture_constraints(monkeypatch, asymmetry=0.0):
    benchmark = np.array([0.25, 0.25, 0.25, 0.25, 0.0])
    current = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    assets = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c", "d", "cash"],
            "expected_alpha": [0.01, -0.02, 0.03, -0.01, 0.0],
            "current_weight": current,
            "benchmark_weight": benchmark,
            "specific_variance": [0.01, 0.02, 0.03, 0.04, 0.0],
            "adv": [1000.0, 2000.0, 3000.0, 4000.0, np.inf],
            "is_cash": [False, False, False, False, True],
            "industry": [1.0, 1.0, 0.0, 0.0, 0.0],
            "style": [-1.0, 0.0, 0.5, 2.0, 0.0],
        }
    )
    captured = []

    def solve(fun, x0, **kwargs):
        captured.extend(kwargs["constraints"])
        return SimpleNamespace(success=True, x=benchmark)

    monkeypatch.setattr(optimizer, "minimize", solve)
    optimizer.optimize_portfolio(
        assets,
        factor_covariance=pd.DataFrame(
            [[0.02, 0.003 + asymmetry], [0.003, 0.01]],
            index=["industry", "style"],
            columns=["industry", "style"],
        ),
        industry_columns=("industry",),
        style_columns=("style",),
        expected_transaction_cost=lambda weights: 0.0,
        nav=100.0,
        assumptions=ExecutionAssumptions(100.0, 0.0, 0.0, 0.05, (0.05,)),
        cash_column="is_cash",
        config=optimizer.OptimizerConfig(active_name_limit=0.2),
    )
    return captured, current


@pytest.mark.parametrize("asymmetry", [0.0, 2e-8])
def test_constraint_jacobians_match_independent_central_differences(
    monkeypatch, asymmetry
):
    constraints, _ = _capture_constraints(monkeypatch, asymmetry)
    point = np.array([0.19, 0.21, 0.23, 0.25, 0.12])
    step = 1e-6
    for constraint in constraints:
        derivative = np.column_stack(
            [
                (
                    np.atleast_1d(constraint["fun"](point + direction * step))
                    - np.atleast_1d(constraint["fun"](point - direction * step))
                )
                / (2 * step)
                for direction in np.eye(len(point))
            ]
        )
        assert np.atleast_2d(constraint["jac"](point)) == pytest.approx(
            derivative, abs=1e-10, rel=1e-8
        )


def test_absolute_value_kinks_use_valid_zero_subgradients(monkeypatch):
    constraints, current = _capture_constraints(monkeypatch)
    assert constraints[2]["jac"](current) == pytest.approx(np.zeros(5))
    assert constraints[5]["jac"](current) == pytest.approx(np.zeros((5, 5)))
    cash_only = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    assert constraints[3]["jac"](cash_only) == pytest.approx(
        np.array([[0.02, 0.02, 0.02, 0.02, 0.0]])
    )
    assert constraints[4]["jac"](cash_only) == pytest.approx(
        np.array([[0.2, 0.2, 0.2, 0.2, 0.0]])
    )
