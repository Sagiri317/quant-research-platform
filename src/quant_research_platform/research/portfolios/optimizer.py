"""Deterministic SLSQP benchmark-relative optimizer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quant_research_platform.contracts.portfolios import (
    ExecutionAssumptions,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_real_numeric_series,
)
from quant_research_platform.research.portfolios.risk import _covariance_is_psd


@dataclass(frozen=True)
class OptimizerConfig:
    active_name_limit: float = 0.005
    tracking_error_limit: float = 0.05
    target_turnover_limit: float = 0.2
    industry_active_limit: float = 0.02
    style_active_limit: float = 0.2
    solver_ftol: float = 1e-12
    solver_maxiter: int = 1000

    def __post_init__(self) -> None:
        _validate_config(self)


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    predicted_tracking_error: float
    factor_variance: float
    specific_variance: float
    target_turnover: float
    expected_transaction_cost: float
    maximum_adv_participation: float


def optimize_portfolio(
    assets: pd.DataFrame,
    *,
    factor_covariance: pd.DataFrame,
    industry_columns: tuple[str, ...] = (),
    style_columns: tuple[str, ...] = (),
    expected_transaction_cost: Callable[[np.ndarray], float],
    nav: float,
    assumptions: ExecutionAssumptions,
    financial_column: str | None = None,
    cash_column: str | None = None,
    config: OptimizerConfig | None = None,
    eligibility_column: str | None = None,
) -> OptimizationResult:
    """Maximize expected active alpha minus expected trade cost under hard limits."""
    required = {
        "asset_id",
        "expected_alpha",
        "current_weight",
        "benchmark_weight",
        "specific_variance",
        "adv",
        *industry_columns,
        *style_columns,
    }
    if not isinstance(assets, pd.DataFrame) or not required.issubset(assets.columns):
        raise ValueError("optimizer_columns_invalid")
    if not industry_columns and (not style_columns):
        raise ValueError("optimizer_exposure_columns_required")
    if not callable(expected_transaction_cost):
        raise ValueError("optimizer_cost_provider_invalid")
    config = OptimizerConfig() if config is None else config
    if not isinstance(assumptions, ExecutionAssumptions):
        raise ValueError("optimizer_execution_assumptions_invalid")
    if type(nav) is not float or not np.isfinite(nav) or nav <= 0:
        raise ValueError("optimizer_nav_invalid")
    if eligibility_column is not None and eligibility_column not in assets:
        raise ValueError("optimizer_eligibility_column_missing")
    if financial_column is not None and financial_column not in assets:
        raise ValueError("optimizer_financial_column_missing")
    if cash_column is not None and cash_column not in assets:
        raise ValueError("optimizer_cash_column_missing")
    columns = tuple(dict.fromkeys((*industry_columns, *style_columns)))
    if (
        tuple(factor_covariance.index) != columns
        or tuple(factor_covariance.columns) != columns
    ):
        raise ValueError("optimizer_covariance_columns_invalid")
    covariance = factor_covariance.apply(
        strict_finite_numeric_series, error="optimizer_covariance_invalid"
    ).to_numpy(dtype="float64")
    if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T):
        raise ValueError("optimizer_covariance_invalid")
    if not _covariance_is_psd(covariance):
        raise ValueError("optimizer_covariance_not_psd")
    data = assets.copy().reset_index(drop=True)
    if data["asset_id"].duplicated().any():
        raise ValueError("optimizer_asset_duplicate")
    for column in (
        "expected_alpha",
        "current_weight",
        "benchmark_weight",
        "specific_variance",
        "adv",
        *columns,
    ):
        data[column] = strict_real_numeric_series(
            data[column], error="optimizer_numeric_invalid"
        )
    for column in (eligibility_column, financial_column, cash_column):
        if column is not None and (
            not pd.api.types.is_bool_dtype(data[column]) or data[column].isna().any()
        ):
            raise ValueError("optimizer_boolean_column_invalid")
    for column in ("current_weight", "benchmark_weight"):
        if data[column].lt(0.0).any():
            raise ValueError("optimizer_weight_invalid")
    current = data["current_weight"].to_numpy()
    benchmark = data["benchmark_weight"].to_numpy()
    if not np.isclose(current.sum(), 1.0) or not np.isclose(benchmark.sum(), 1.0):
        raise ValueError("optimizer_weight_sum_invalid")
    valid_adv = np.isfinite(data["adv"]) & data["adv"].gt(0.0)
    cash = (
        data[cash_column].to_numpy(dtype=bool)
        if cash_column is not None
        else np.zeros(len(data), dtype=bool)
    )
    frozen = ~valid_adv.to_numpy(dtype=bool) & ~cash
    for column in ("can_enter", "can_exit"):
        if column in data:
            if not pd.api.types.is_bool_dtype(data[column]):
                raise ValueError("optimizer_boolean_column_invalid")
    risk_missing = (
        ~np.isfinite(data["specific_variance"])
        | data["specific_variance"].lt(0.0)
        | ~np.isfinite(data.loc[:, columns]).all(axis=1)
    )
    incomplete = risk_missing | ~np.isfinite(data["expected_alpha"])
    # Frozen holdings remain real active risk. Missing volume never authorizes
    # replacing their exposures or specific variance with zero.
    cannot_reach_benchmark = frozen.copy()
    if "can_enter" in data:
        cannot_reach_benchmark |= (current < benchmark) & ~data.can_enter.fillna(
            False
        ).to_numpy(dtype=bool)
    if "can_exit" in data:
        cannot_reach_benchmark |= (current > benchmark) & ~data.can_exit.fillna(
            False
        ).to_numpy(dtype=bool)
    if (risk_missing & (current != benchmark) & cannot_reach_benchmark).any():
        raise ValueError("optimizer_held_risk_unavailable")
    data["expected_alpha"] = data["expected_alpha"].where(~incomplete, 0.0)
    data["specific_variance"] = data["specific_variance"].where(~risk_missing, 0.0)
    data["adv"] = data["adv"].where(valid_adv, np.inf)
    exposure = (
        data.loc[:, columns].where(~risk_missing, 0.0, axis=0).to_numpy(dtype="float64")
    )
    expected_alpha = data["expected_alpha"].to_numpy(dtype="float64")
    specific = data["specific_variance"].to_numpy(dtype="float64")
    adv = data["adv"].to_numpy(dtype="float64")
    bounds = _bounds(
        data,
        benchmark=benchmark,
        financial_column=financial_column,
        cash_column=cash_column,
        eligibility_column=eligibility_column,
        incomplete=incomplete.to_numpy(dtype=bool),
        frozen=frozen,
        config=config,
    )
    # These are necessary box constraints, independent of the numerical solver.
    # Report proven economic infeasibility separately from solver failures.
    tolerance = 1e-8
    lower, upper = np.asarray(bounds, dtype="float64").T
    if lower.sum() > 1.0 + tolerance or upper.sum() < 1.0 - tolerance:
        raise ValueError("optimizer_budget_bounds_infeasible")
    capacity = assumptions.adv_participation * adv / nav
    reachable_lower = np.maximum(lower, current - capacity)
    reachable_upper = np.minimum(upper, current + capacity)
    if (
        np.any(reachable_lower > reachable_upper + tolerance)
        or reachable_lower.sum() > 1.0 + tolerance
        or reachable_upper.sum() < 1.0 - tolerance
    ):
        raise ValueError("optimizer_capacity_bounds_infeasible")
    minimum_turnover = max(
        np.maximum(reachable_lower - current, 0.0).sum(),
        np.maximum(current - reachable_upper, 0.0).sum(),
    )
    if minimum_turnover > config.target_turnover_limit + tolerance:
        raise ValueError("optimizer_turnover_bounds_infeasible")

    def risk(weights: np.ndarray) -> float:
        active = weights - benchmark
        return sum(_risk_variances(active, exposure, covariance, specific))

    relative_exposure = _stock_leg_exposures(exposure, benchmark, cash)

    # Exact derivatives avoid forward-difference bias at no-trade kinks.
    # np.sign(0) selects the valid zero subgradient of absolute value.
    def risk_jacobian(weights: np.ndarray) -> np.ndarray:
        active = weights - benchmark
        factor_active = active @ exposure
        return -(
            exposure @ (covariance @ factor_active + covariance.T @ factor_active)
            + 2.0 * specific * active
        )

    def exposure_jacobian(weights, matrix, limit):
        return (
            limit * (~cash)[None, :] - np.sign(matrix.T @ weights)[:, None] * matrix.T
        )

    constraints = [
        {
            "type": "eq",
            "fun": lambda weights: float(weights.sum() - 1.0),
            "jac": lambda weights: np.ones(len(weights)),
        },
        {
            "type": "ineq",
            "fun": lambda weights: float(
                config.tracking_error_limit**2 - risk(weights)
            ),
            "jac": risk_jacobian,
        },
        {
            "type": "ineq",
            "fun": lambda weights: float(
                config.target_turnover_limit - 0.5 * np.abs(weights - current).sum()
            ),
            "jac": lambda weights: -0.5 * np.sign(weights - current),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                config.industry_active_limit * weights[~cash].sum()
                - np.abs(relative_exposure[:, : len(industry_columns)].T @ weights)
            ),
            "jac": lambda weights: exposure_jacobian(
                weights,
                relative_exposure[:, : len(industry_columns)],
                config.industry_active_limit,
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                config.style_active_limit * weights[~cash].sum()
                - np.abs(relative_exposure[:, len(industry_columns) :].T @ weights)
            ),
            "jac": lambda weights: exposure_jacobian(
                weights,
                relative_exposure[:, len(industry_columns) :],
                config.style_active_limit,
            ),
        },
        {
            "type": "ineq",
            "fun": lambda weights: (
                assumptions.adv_participation - np.abs(weights - current) * nav / adv
            ),
            "jac": lambda weights: np.diag(-np.sign(weights - current) * nav / adv),
        },
    ]
    result = minimize(
        lambda weights: float(
            -(expected_alpha @ (weights - benchmark))
            + float(expected_transaction_cost(weights)) / nav
        ),
        current.copy(),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": config.solver_ftol, "maxiter": config.solver_maxiter},
    )
    if not result.success:
        raise ValueError("optimizer_failed_closed")
    weights = np.asarray(result.x, dtype="float64")
    _verify_solution(
        weights,
        bounds=bounds,
        current=current,
        benchmark=benchmark,
        exposure=exposure,
        covariance=covariance,
        specific=specific,
        adv=adv,
        config=config,
        industry_count=len(industry_columns),
        nav=nav,
        assumptions=assumptions,
        cash=cash,
    )
    active = weights - benchmark
    factor_variance, specific_variance = _risk_variances(
        active, exposure, covariance, specific
    )
    participation = np.abs(weights - current) * nav / adv
    return OptimizationResult(
        weights=pd.Series(weights, index=data["asset_id"], name="weight"),
        predicted_tracking_error=float(np.sqrt(factor_variance + specific_variance)),
        factor_variance=factor_variance,
        specific_variance=specific_variance,
        target_turnover=float(0.5 * np.abs(weights - current).sum()),
        expected_transaction_cost=float(expected_transaction_cost(weights)),
        maximum_adv_participation=float(participation.max(initial=0.0)),
    )


def _risk_variances(active, exposure, covariance, specific) -> tuple[float, float]:
    nonzero = active != 0.0
    active, exposure, specific = active[nonzero], exposure[nonzero], specific[nonzero]
    factor_active = active @ exposure
    factor = float(factor_active @ covariance @ factor_active)
    residual = float(np.sum(specific * active**2))
    if not np.isfinite((factor, residual, factor + residual)).all():
        raise ValueError("optimizer_risk_nonfinite")
    if factor < 0:
        # Distinguish dot-product roundoff from a genuinely negative admitted form.
        exact_active = [
            sum(
                Fraction(float(a)) * Fraction(float(x))
                for a, x in zip(active, column, strict=True)
            )
            for column in exposure.T
        ]
        exact = sum(
            left * Fraction(float(value)) * right
            for left, row in zip(exact_active, covariance, strict=True)
            for value, right in zip(row, exact_active, strict=True)
        )
        if exact < 0:
            raise ValueError("optimizer_risk_negative")
        try:
            factor = float(exact)
        except OverflowError:
            raise ValueError("optimizer_risk_nonfinite") from None
        if not np.isfinite(factor + residual):
            raise ValueError("optimizer_risk_nonfinite")
    return factor, residual


def _bounds(
    data: pd.DataFrame,
    *,
    benchmark: np.ndarray,
    financial_column: str | None,
    cash_column: str | None,
    eligibility_column: str | None,
    incomplete: np.ndarray,
    frozen: np.ndarray,
    config: OptimizerConfig,
) -> tuple[tuple[float, float], ...]:
    bounds: list[tuple[float, float]] = []
    for position, row in data.iterrows():
        if cash_column is not None and bool(row[cash_column]):
            bounds.append((0.0, 1.0))
            continue
        fixed = (
            financial_column is not None
            and bool(row[financial_column])
            or float(benchmark[position]) == 0.0
            or (eligibility_column is not None and (not bool(row[eligibility_column])))
            or incomplete[position]
        )
        if fixed:
            bounds.append((float(benchmark[position]), float(benchmark[position])))
            continue
        bounds.append(
            (
                max(0.0, float(benchmark[position]) - config.active_name_limit),
                min(1.0, float(benchmark[position]) + config.active_name_limit),
            )
        )
    for position, row in data.iterrows():
        lower, upper = bounds[position]
        current = float(row.current_weight)
        if frozen[position]:
            lower, upper = max(lower, current), min(upper, current)
        if "can_enter" in data and (pd.isna(row.can_enter) or not bool(row.can_enter)):
            upper = min(upper, current)
        if "can_exit" in data and (pd.isna(row.can_exit) or not bool(row.can_exit)):
            lower = max(lower, current)
        if lower > upper:
            raise ValueError("optimizer_frozen_constraints_infeasible")
        bounds[position] = (lower, upper)
    return tuple(bounds)


def _stock_leg_exposures(exposure, benchmark, cash):
    """Center stock exposures on the full benchmark stock leg, excluding cash."""
    reference = benchmark.copy()
    reference[cash] = 0.0
    gross = reference.sum()
    if gross <= 0 and exposure.shape[1]:
        raise ValueError("optimizer_benchmark_stock_leg_missing")
    centered = exposure - (exposure.T @ (reference / gross) if gross > 0 else 0.0)
    centered[cash] = 0.0
    return centered


def _verify_solution(
    weights: np.ndarray,
    *,
    bounds: tuple[tuple[float, float], ...],
    current: np.ndarray,
    benchmark: np.ndarray,
    exposure: np.ndarray,
    covariance: np.ndarray,
    specific: np.ndarray,
    adv: np.ndarray,
    config: OptimizerConfig,
    industry_count: int,
    nav: float,
    assumptions: ExecutionAssumptions,
    cash: np.ndarray | None = None,
) -> None:
    tolerance = 1e-08
    active = weights - benchmark
    risk = sum(_risk_variances(active, exposure, covariance, specific))
    if not np.isfinite(weights).all() or not np.isclose(
        weights.sum(), 1.0, atol=tolerance
    ):
        raise ValueError("optimizer_solution_invalid")
    if any(
        (
            value < lower - tolerance or value > upper + tolerance
            for value, (lower, upper) in zip(weights, bounds, strict=True)
        )
    ):
        raise ValueError("optimizer_solution_invalid")
    if risk > config.tracking_error_limit**2 + tolerance:
        raise ValueError("optimizer_solution_invalid")
    if 0.5 * np.abs(weights - current).sum() > config.target_turnover_limit + tolerance:
        raise ValueError("optimizer_solution_invalid")
    if cash is None:
        cash = np.zeros(len(weights), dtype=bool)
    relative_exposure = _stock_leg_exposures(exposure, benchmark, cash)
    stock_gross = weights[~cash].sum()
    if np.any(
        np.abs(relative_exposure[:, :industry_count].T @ weights)
        > config.industry_active_limit * stock_gross + tolerance
    ):
        raise ValueError("optimizer_solution_invalid")
    if np.any(
        np.abs(relative_exposure[:, industry_count:].T @ weights)
        > config.style_active_limit * stock_gross + tolerance
    ):
        raise ValueError("optimizer_solution_invalid")
    if np.any(
        np.abs(weights - current) * nav / adv
        > assumptions.adv_participation + tolerance
    ):
        raise ValueError("optimizer_solution_invalid")


def _validate_config(config: OptimizerConfig) -> None:
    values = (
        config.active_name_limit,
        config.tracking_error_limit,
        config.target_turnover_limit,
        config.industry_active_limit,
        config.style_active_limit,
        config.solver_ftol,
    )
    if any(
        type(value) is not float or not np.isfinite(value) or value < 0.0
        for value in values
    ):
        raise ValueError("optimizer_config_invalid")
    if (
        config.active_name_limit <= 0.0
        or config.tracking_error_limit <= 0.0
        or config.target_turnover_limit <= 0.0
    ):
        raise ValueError("optimizer_config_invalid")
    if type(config.solver_maxiter) is not int or config.solver_maxiter <= 0:
        raise ValueError("optimizer_config_invalid")


def _project_neutral_target(target, context, *, industry_limit=0.02, size_limit=0.2):
    """Project stock-leg weights within explicit benchmark exposure bands."""
    for limit in (industry_limit, size_limit):
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int | float)
            or not np.isfinite(limit)
            or limit < 0
        ):
            raise ValueError("target_neutrality_limit_invalid")
    q = target.reindex(context.index, fill_value=0.0).to_numpy(dtype="float64")
    if not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("target_weights_invalid")
    b = context.benchmark_weight.to_numpy(dtype="float64")
    exposures = context.drop(columns="benchmark_weight").to_numpy(dtype="float64").T
    limits = np.array(
        [
            industry_limit if name.startswith("industry:") else size_limit
            for name in context.columns.drop("benchmark_weight")
        ],
        dtype="float64",
    )
    exact = exposures[limits == 0]
    industry_rows = np.array(
        [
            name.startswith("industry:")
            for name in context.columns.drop("benchmark_weight")
        ]
    )
    if industry_rows.any() and np.all(exposures[industry_rows].sum(axis=0) == 1):
        reference_coverage = float((exposures[industry_rows] @ b).sum())
        if abs(1.0 - reference_coverage) > limits[industry_rows].sum() + 1e-8:
            raise PortfolioTargetBlocked("target_neutrality_infeasible")
    bounded = exposures[limits > 0]
    bands = limits[limits > 0]
    matrix = np.vstack((np.ones(len(b)), exact))
    right_hand_side = np.concatenate(([1.0], exact @ b))
    # Remove only algebraically redundant equalities; retain their residual checks.
    left, singular, vectors = np.linalg.svd(matrix, full_matrices=False)
    rank = np.count_nonzero(
        singular > singular[0] * max(matrix.shape) * np.finfo(float).eps
    )
    constraints = vectors[:rank]
    values = (left[:, :rank].T @ right_hand_side) / singular[:rank]
    if not np.allclose(
        matrix @ (constraints.T @ values), right_hand_side, rtol=0.0, atol=1e-8
    ):
        raise PortfolioTargetBlocked("target_neutrality_infeasible")
    optimizer_constraints = [
        {
            "type": "eq",
            "fun": lambda w: constraints @ w - values,
            "jac": lambda w: constraints,
        }
    ]
    if len(bands):
        optimizer_constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: np.concatenate(
                    (bands - bounded @ (w - b), bands + bounded @ (w - b))
                ),
                "jac": lambda w: np.vstack((-bounded, bounded)),
            }
        )
    solved = minimize(
        lambda w: float(0.5 * np.dot(w - q, w - q)),
        b / b.sum(),
        jac=lambda w: w - q,
        method="SLSQP",
        bounds=[(0.0, None)] * len(b),
        constraints=optimizer_constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not solved.success:
        raise PortfolioTargetBlocked("target_neutrality_solver_failed")
    weights = np.maximum(solved.x, 0.0)
    if (
        not np.isfinite(weights).all()
        or not np.allclose(matrix @ weights, right_hand_side, rtol=0.0, atol=1e-8)
        or np.any(np.abs(exposures @ (weights - b)) > limits + 1e-8)
    ):
        raise PortfolioTargetBlocked("target_neutrality_infeasible")
    return pd.Series(
        weights, index=context.index, dtype="float64", name="target_weight"
    )
