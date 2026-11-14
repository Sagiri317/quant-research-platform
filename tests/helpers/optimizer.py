"""Explicit zero-balance cash leg for synthetic optimizer stock universes."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import LinearConstraint
from scipy.sparse import issparse

# Decimal-return accuracy for independent continuous economic optima.
ECONOMIC_UTILITY_ATOL = 2e-9


def with_cash_asset(stocks):
    assert "is_cash" not in stocks
    cash = {
        column: False if pd.api.types.is_bool_dtype(stocks[column]) else 0.0
        for column in stocks
    }
    cash.update(asset_id="__cash__", adv=np.inf, is_cash=True)
    return pd.concat(
        [stocks.assign(is_cash=False), pd.DataFrame([cash])], ignore_index=True
    )


def capture_solver_problem(monkeypatch, operation):
    from quant_research_platform.research.portfolios import optimizer

    captured = {}

    def inspect(objective, initial, **kwargs):
        captured.update(kwargs, objective=objective, initial=initial.copy())
        # Inspect the problem without pretending its initial point is an optimum.
        raise RuntimeError("optimizer_problem_captured")

    with monkeypatch.context() as patch:
        patch.setattr(optimizer, "minimize", inspect)
        with pytest.raises(RuntimeError, match="^optimizer_problem_captured$"):
            operation()
    return captured


def constraint_value(constraint, point):
    return np.atleast_1d(
        constraint.A @ point
        if isinstance(constraint, LinearConstraint)
        else constraint.fun(point)
    )


def constraint_jacobian(constraint, point):
    matrix = (
        constraint.A
        if isinstance(constraint, LinearConstraint)
        else constraint.jac(point)
    )
    assert issparse(matrix)
    return matrix.toarray()
