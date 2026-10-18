"""Deterministic SLSQP benchmark-relative optimizer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

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
    expected_transaction_cost: Callable[
        [np.ndarray, np.ndarray], tuple[float, np.ndarray, np.ndarray]
    ],
    nav: float,
    assumptions: ExecutionAssumptions,
    financial_column: str | None = None,
    cash_column: str,
    config: OptimizerConfig | None = None,
    eligibility_column: str | None = None,
    receivable_weight: float = 0.0,
) -> OptimizationResult:
    """Maximize active alpha less actual costs in a self-financing account.

    Holdings and explicit cash use pre-trade NAV units; cash reserves the true
    fee, while name and risk limits use post-fee NAV. The cost owner supplies
    convex side costs and marginal rates, with zero cash cost. Its smooth
    extension covers signed epigraph iterates; feasibility uses actual trades.
    """
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
    if (
        type(receivable_weight) is not float
        or not np.isfinite(receivable_weight)
        or not 0.0 <= receivable_weight <= 1.0
    ):
        raise ValueError("optimizer_receivable_weight_invalid")
    budget = 1.0 - receivable_weight
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
    if not np.isclose(current.sum(), budget) or not np.isclose(benchmark.sum(), 1.0):
        raise ValueError("optimizer_weight_sum_invalid")
    valid_adv = np.isfinite(data["adv"]) & data["adv"].gt(0.0)
    cash = data[cash_column].to_numpy(dtype=bool)
    if cash.sum() != 1 or benchmark[cash].any():
        raise ValueError("optimizer_cash_asset_invalid")
    frozen = ~valid_adv.to_numpy(dtype=bool) & ~cash
    for column in ("can_enter", "can_exit"):
        if column in data:
            if not pd.api.types.is_bool_dtype(data[column]):
                raise ValueError("optimizer_boolean_column_invalid")
    if (
        data["specific_variance"].lt(0.0).any()
        or np.isinf(
            data.loc[:, ["specific_variance", *columns]].to_numpy(dtype="float64")
        ).any()
    ):
        raise ValueError("optimizer_risk_invalid")
    risk_missing = data["specific_variance"].isna() | data.loc[:, columns].isna().any(
        axis=1
    )
    incomplete = risk_missing | ~np.isfinite(data["expected_alpha"])
    # Frozen holdings remain real active risk. Missing volume never authorizes
    # replacing their exposures or specific variance with zero.
    cannot_reduce = frozen.copy()
    if "can_exit" in data:
        cannot_reduce |= ~data.can_exit.fillna(False).to_numpy(dtype=bool)
    # Fees can raise a frozen underweight to its net benchmark, but cannot
    # reduce an overweight. Keep the same normalization policy at admission.
    if (
        risk_missing
        & (current > benchmark)
        & ~_normalization_matches(current, benchmark)
        & cannot_reduce
    ).any():
        raise ValueError("optimizer_held_risk_unavailable")
    data["expected_alpha"] = data["expected_alpha"].where(~incomplete, 0.0)
    original_specific = data["specific_variance"].to_numpy(dtype="float64")
    original_exposure = data.loc[:, columns].to_numpy(dtype="float64")
    data["specific_variance"] = data["specific_variance"].where(~risk_missing, 0.0)
    data["adv"] = data["adv"].where(valid_adv, np.inf)
    exposure = np.where(np.isfinite(original_exposure), original_exposure, 0.0)
    expected_alpha = data["expected_alpha"].to_numpy(dtype="float64")
    specific = data["specific_variance"].to_numpy(dtype="float64")
    adv = data["adv"].to_numpy(dtype="float64")
    # Scale before multiplying so representable capacity survives tiny ADV/NAV.
    participation_mantissa, participation_exponent = np.frexp(
        assumptions.adv_participation
    )
    adv_mantissa, adv_exponent = np.frexp(adv)
    nav_mantissa, nav_exponent = np.frexp(nav)
    with np.errstate(over="ignore", under="ignore"):
        capacity = np.ldexp(
            participation_mantissa * adv_mantissa / nav_mantissa,
            participation_exponent + adv_exponent - nav_exponent,
        )
    # Monotone side costs at individual maximum trades bound every feasible fee.
    # This only proves necessary box infeasibility; actual constraints below use
    # the exact net trades, never a fee margin or the objective's epigraph slack.
    try:
        with np.errstate(over="raise"):
            maximum_cost = expected_transaction_cost(
                np.where(cash, 0.0, np.minimum(1.0 - current, capacity)) * nav,
                np.where(cash, 0.0, np.minimum(current, capacity)) * nav,
            )[0]
    except (FloatingPointError, OverflowError):
        # An unrepresentable hypothetical fee cannot rule out a finite trade.
        # Cash funds actual fees, so c <= cash <= 1 - receivable_weight.
        minimum_nav_ratio = receivable_weight
    else:
        minimum_nav_ratio = max(receivable_weight, 1.0 - maximum_cost / nav)
    bounds, net_bounds = _bounds(
        data,
        benchmark=benchmark,
        financial_column=financial_column,
        cash_column=cash_column,
        eligibility_column=eligibility_column,
        incomplete=incomplete.to_numpy(dtype=bool),
        frozen=frozen,
        config=config,
        minimum_nav_ratio=minimum_nav_ratio,
    )
    tolerance = 1e-8
    lower, upper = np.asarray(bounds, dtype="float64").T
    net_lower, net_upper = np.asarray(net_bounds, dtype="float64").T
    if (
        net_lower.sum() > budget + tolerance
        or lower.sum() > budget + tolerance
        or upper.sum() < budget - tolerance
    ):
        raise ValueError("optimizer_budget_bounds_infeasible")
    reachable_lower = np.maximum(lower, current - capacity)
    reachable_upper = np.minimum(upper, current + capacity)
    if (
        np.any(reachable_lower > reachable_upper + tolerance)
        or reachable_lower.sum() > budget + tolerance
        or reachable_upper.sum() < budget - tolerance
    ):
        raise ValueError("optimizer_capacity_bounds_infeasible")
    minimum_turnover = max(
        np.maximum(reachable_lower - current, 0.0).sum(),
        np.maximum(current - reachable_upper, 0.0).sum(),
    )
    if minimum_turnover > config.target_turnover_limit + tolerance:
        raise ValueError("optimizer_turnover_bounds_infeasible")

    relative_exposure = _stock_leg_exposures(exposure, benchmark, cash)
    count = len(current)
    identity = np.eye(count)
    cash_direction = cash.astype("float64")

    def actual_cost(weights):
        delta = weights - current
        cost, buy_gradient, sell_gradient = expected_transaction_cost(
            np.maximum(delta, 0.0) * nav,
            np.maximum(-delta, 0.0) * nav,
        )
        # At an unchanged name, use the central subgradient of the convex fee
        # kink. The smooth objective still uses its exact trade epigraph.
        gradient = np.where(
            delta > 0.0,
            buy_gradient,
            np.where(delta < 0.0, -sell_gradient, (buy_gradient - sell_gradient) / 2),
        )
        return cost / nav, gradient

    def net_risk(point, *, derivative=False):
        weights = point[:count]
        cost, gradient = actual_cost(weights)
        active = weights - benchmark + (benchmark - cash_direction) * cost
        variance = sum(_risk_variances(active, exposure, covariance, specific))
        if not derivative:
            return config.tracking_error_limit**2 * (1.0 - cost) ** 2 - variance
        factor_active = active @ exposure
        variance_gradient = (
            exposure @ (covariance @ factor_active + covariance.T @ factor_active)
            + 2.0 * specific * active
        )
        # Apply the rank-one chain rule without materializing an N-by-N matrix.
        risk_gradient = (
            variance_gradient
            + (variance_gradient @ (benchmark - cash_direction)) * gradient
        )
        return np.r_[
            -2.0 * config.tracking_error_limit**2 * (1.0 - cost) * gradient
            - risk_gradient,
            np.zeros(count),
        ]

    def net_name_bands(point, *, derivative=False):
        weights = point[:count]
        cost, gradient = actual_cost(weights)
        if not derivative:
            return np.r_[
                weights[~cash] - net_lower[~cash] * (1.0 - cost),
                net_upper[~cash] * (1.0 - cost) - weights[~cash],
            ]
        return np.pad(
            np.vstack(
                (
                    identity[~cash] + np.outer(net_lower[~cash], gradient),
                    -identity[~cash] - np.outer(net_upper[~cash], gradient),
                )
            ),
            ((0, 0), (0, count)),
        )

    def cash_reserve(point, *, derivative=False):
        cost, gradient = actual_cost(point[:count])
        if not derivative:
            return float(point[:count][cash].sum() - cost)
        return np.r_[cash_direction - gradient, np.zeros(count)]

    def exposure_bands(matrix, limit):
        return np.concatenate(
            (limit * (~cash)[None, :] - matrix.T, limit * (~cash)[None, :] + matrix.T)
        )

    industry_bands = exposure_bands(
        relative_exposure[:, : len(industry_columns)], config.industry_active_limit
    )
    style_bands = exposure_bands(
        relative_exposure[:, len(industry_columns) :], config.style_active_limit
    )

    # Fixed weights also have fixed absolute trades. Leaving their auxiliaries
    # free can create a zero-row budget Jacobian when all weights are fixed.
    trade_upper = np.where(
        lower == upper, np.minimum(capacity, abs(lower - current)), capacity
    )
    trade_lower = np.where(lower == upper, trade_upper, 0.0)
    turnover_jacobian = np.r_[np.zeros(count), np.full(count, -0.5)]
    trade_jacobian = np.block(
        [[-np.eye(count), np.eye(count)], [np.eye(count), np.eye(count)]]
    )
    industry_jacobian = np.pad(industry_bands, ((0, 0), (0, count)))
    style_jacobian = np.pad(style_bands, ((0, 0), (0, count)))
    constraints = [
        {
            "type": "eq",
            "fun": lambda point: float(point[:count].sum() - budget),
            "jac": lambda point: np.r_[np.ones(count), np.zeros(count)],
        },
        {
            "type": "ineq",
            "fun": net_risk,
            "jac": lambda point: net_risk(point, derivative=True),
        },
        {
            "type": "ineq",
            "fun": lambda point: float(
                config.target_turnover_limit - 0.5 * point[count:].sum()
            ),
            "jac": lambda point: turnover_jacobian,
        },
        {
            "type": "ineq",
            "fun": lambda point: industry_bands @ point[:count],
            "jac": lambda point: industry_jacobian,
        },
        {
            "type": "ineq",
            "fun": lambda point: style_bands @ point[:count],
            "jac": lambda point: style_jacobian,
        },
        {
            "type": "ineq",
            "fun": lambda point: np.concatenate(
                (
                    point[count:] - (point[:count] - current),
                    point[count:] + (point[:count] - current),
                )
            ),
            "jac": lambda point: trade_jacobian,
        },
    ]

    constraints.extend(
        (
            {
                "type": "ineq",
                "fun": net_name_bands,
                "jac": lambda point: net_name_bands(point, derivative=True),
            },
            {
                "type": "ineq",
                "fun": cash_reserve,
                "jac": lambda point: cash_reserve(point, derivative=True),
            },
        )
    )

    def objective(point):
        delta, absolute_trade = point[:count] - current, point[count:]
        cost, buy_gradient, sell_gradient = expected_transaction_cost(
            ((absolute_trade + delta) / 2.0) * nav,
            ((absolute_trade - delta) / 2.0) * nav,
        )
        return (
            float(-expected_alpha @ (point[:count] - benchmark) + cost / nav),
            np.r_[
                -expected_alpha + (buy_gradient - sell_gradient) / 2.0,
                (buy_gradient + sell_gradient) / 2.0,
            ],
        )

    result = minimize(
        objective,
        np.r_[current, np.zeros(count)],
        jac=True,
        method="SLSQP",
        bounds=(*bounds, *zip(trade_lower, trade_upper, strict=True)),
        constraints=constraints,
        options={"ftol": config.solver_ftol, "maxiter": config.solver_maxiter},
    )
    if not result.success:
        raise ValueError("optimizer_failed_closed")
    weights = np.asarray(result.x[:count], dtype="float64")
    cost_ratio, _ = actual_cost(weights)
    if cost_ratio >= 1.0 or weights[cash].sum() < cost_ratio - tolerance:
        raise ValueError("optimizer_solution_invalid")
    net_weights = (weights - cash_direction * cost_ratio) / (1.0 - cost_ratio)
    _verify_solution(
        weights,
        net_weights=net_weights,
        net_bounds=net_bounds,
        bounds=bounds,
        current=current,
        benchmark=benchmark,
        exposure=original_exposure,
        covariance=covariance,
        specific=original_specific,
        adv=adv,
        config=config,
        industry_count=len(industry_columns),
        nav=nav,
        assumptions=assumptions,
        cash=cash,
        receivable_weight=receivable_weight,
    )
    active = _risk_active_weights(
        net_weights, benchmark, original_exposure, original_specific
    )
    factor_variance, specific_variance = _risk_variances(
        active, original_exposure, covariance, original_specific
    )
    participation = np.abs(weights - current) * nav / adv
    return OptimizationResult(
        weights=pd.Series(weights, index=data["asset_id"], name="weight"),
        predicted_tracking_error=float(np.sqrt(factor_variance + specific_variance)),
        factor_variance=factor_variance,
        specific_variance=specific_variance,
        target_turnover=float(0.5 * np.abs(weights - current).sum()),
        expected_transaction_cost=float(cost_ratio * nav),
        maximum_adv_participation=float(participation.max(initial=0.0)),
    )


def _normalization_matches(weights, reference):
    """Compare normalized coefficients only within scale-relative roundoff."""
    roundoff = 8 * np.finfo(float).eps * np.maximum(abs(weights), abs(reference))
    return (
        np.isfinite(weights)
        & np.isfinite(reference)
        & (abs(weights - reference) <= roundoff)
    )


def _risk_active_weights(weights, benchmark, exposure, specific):
    """Keep known risk exact; certify zero unknown-risk benchmark coefficients.

    A fixed net-benchmark holding can differ by a few rounding bits after fees
    and NAV division. Only that normalization-sized residual may omit unknown
    risk. A true position against a zero benchmark never meets this relative
    test, and known-risk coefficients retain their full numerical value.
    """
    active = weights - benchmark
    unavailable = (
        ~np.isfinite(exposure).all(axis=1) | ~np.isfinite(specific) | (specific < 0.0)
    )
    active[unavailable & _normalization_matches(weights, benchmark)] = 0.0
    return active


def _risk_variances(active, exposure, covariance, specific) -> tuple[float, float]:
    nonzero = active != 0.0
    active, exposure, specific = active[nonzero], exposure[nonzero], specific[nonzero]
    factor_active = active @ exposure
    factor = float(factor_active @ covariance @ factor_active)
    residual = float(np.sum(specific * active**2))
    if not np.isfinite((factor, residual, factor + residual)).all():
        raise ValueError("optimizer_risk_nonfinite")
    if factor < 0:
        # Rounded covariance entries and BLAS reductions both contribute error.
        # Bound it in variance units; an admitted but materially indefinite
        # covariance (including a negative scalar variance) still fails here.
        magnitude = float(
            np.abs(factor_active) @ np.abs(covariance) @ np.abs(factor_active)
        )
        roundoff = 8 * len(factor_active) * np.finfo(float).eps * magnitude
        if not np.isfinite(roundoff) or factor < -roundoff:
            raise ValueError("optimizer_risk_negative")
        factor = 0.0
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
    minimum_nav_ratio: float,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
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
    net_bounds = tuple(bounds)
    for position, row in data.iterrows():
        lower, upper = bounds[position]
        lower *= minimum_nav_ratio
        current = float(row.current_weight)
        if frozen[position]:
            lower, upper = max(lower, current), min(upper, current)
        if "can_enter" in data and (pd.isna(row.can_enter) or not bool(row.can_enter)):
            upper = min(upper, current)
        if "can_exit" in data and (pd.isna(row.can_exit) or not bool(row.can_exit)):
            lower = max(lower, current)
        if lower > upper:
            if not _normalization_matches(lower, upper):
                raise ValueError("optimizer_frozen_constraints_infeasible")
            # A rounding-only intersection must preserve the physical holding.
            lower, upper = current, current
        bounds[position] = (lower, upper)
    return tuple(bounds), net_bounds


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


def _stock_leg_active_exposure(weights, benchmark, exposure, cash):
    """Ignore only normalization roundoff; unknown required facts stay NaN."""
    stock = ~cash
    gross, reference_gross = weights[stock].sum(), benchmark[stock].sum()
    if gross <= 0.0 or reference_gross <= 0.0:
        return np.full(exposure.shape[1], np.nan)
    normalized = weights[stock] / gross
    reference = benchmark[stock] / reference_gross
    active = normalized - reference
    nonzero = ~_normalization_matches(normalized, reference)
    selected = exposure[stock]
    complete = np.isfinite(selected[nonzero]).all(axis=0)
    result = np.full(exposure.shape[1], np.nan)
    known = np.where(np.isfinite(selected), selected, 0.0)
    result[complete] = active @ known[:, complete]
    return result


def _verify_solution(
    weights: np.ndarray,
    *,
    net_weights: np.ndarray,
    net_bounds: tuple[tuple[float, float], ...],
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
    receivable_weight: float = 0.0,
) -> None:
    tolerance = 1e-08
    active = _risk_active_weights(net_weights, benchmark, exposure, specific)
    risk = sum(_risk_variances(active, exposure, covariance, specific))
    if not np.isfinite(weights).all() or not np.isclose(
        weights.sum(), 1.0 - receivable_weight, atol=tolerance
    ):
        raise ValueError("optimizer_solution_invalid")
    if any(
        (
            value < lower - tolerance or value > upper + tolerance
            for value, (lower, upper) in zip(weights, bounds, strict=True)
        )
    ):
        raise ValueError("optimizer_solution_invalid")
    if any(
        value < lower - tolerance or value > upper + tolerance
        for value, (lower, upper) in zip(net_weights, net_bounds, strict=True)
    ):
        raise ValueError("optimizer_solution_invalid")
    if risk > config.tracking_error_limit**2 + tolerance:
        raise ValueError("optimizer_solution_invalid")
    if 0.5 * np.abs(weights - current).sum() > config.target_turnover_limit + tolerance:
        raise ValueError("optimizer_solution_invalid")
    if cash is None:
        cash = np.zeros(len(weights), dtype=bool)
    stock_gross = weights[~cash].sum()
    if (
        stock_gross > 0.0
        and not np.isfinite(
            _stock_leg_active_exposure(weights, benchmark, exposure, cash)
        ).all()
    ):
        raise ValueError("optimizer_stock_exposure_unavailable")
    # Unknown rows may be eliminated only after their actual coefficients were
    # proved zero above. Keep the original absolute-band feasibility checker.
    relative_exposure = _stock_leg_exposures(
        np.where(np.isfinite(exposure), exposure, 0.0), benchmark, cash
    )
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
    eligible = context.attrs.get("eligible")
    if eligible is not None:
        eligible = eligible.reindex(context.index, fill_value=False)
        # Fixed-zero reference names must not inflate the equality rank.
        matrix[:, ~eligible.to_numpy(dtype=bool)] = 0.0
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
    bounds = [(0.0, None)] * len(b)
    initial = q if eligible is not None else b / b.sum()
    if eligible is not None:
        # Reference-only constituents contribute exposures, never buyable weights.
        bounds = [(0.0, None if value else 0.0) for value in eligible]
        feasible = linprog(
            np.zeros(len(b)),
            A_ub=np.vstack((bounded, -bounded)) if len(bands) else None,
            b_ub=np.concatenate((bands + bounded @ b, bands - bounded @ b))
            if len(bands)
            else None,
            A_eq=constraints,
            b_eq=values,
            bounds=bounds,
            method="highs",
        )
        if feasible.status == 2:
            raise PortfolioTargetBlocked("target_neutrality_infeasible")
        if not feasible.success:
            raise PortfolioTargetBlocked("target_neutrality_solver_failed")
        initial = feasible.x
    solved = minimize(
        lambda w: float(0.5 * np.dot(w - q, w - q)),
        initial,
        jac=lambda w: w - q,
        method="SLSQP",
        bounds=bounds,
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
        raise PortfolioTargetBlocked("target_neutrality_solution_invalid")
    return pd.Series(
        weights, index=context.index, dtype="float64", name="target_weight"
    )
