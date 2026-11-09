"""Self-financing SLSQP targets with relative or explicit absolute risk limits."""

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
from quant_research_platform.contracts.risk import RiskInputsUnavailable
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_real_numeric_series,
)
from quant_research_platform.research.risk import (
    _validated_covariance,
)


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
class AbsoluteRiskConfig:
    """Optional absolute limits, independent of benchmark-relative defaults.

    Name and industry limits are fractions of post-fee total NAV. Style limits
    bound absolute NAV-weighted exposure in the supplied style units. Total
    volatility and its covariance/specific inputs are annualized; short-horizon
    alpha is never added to an annualized risk penalty. None means no limit,
    not a default risk budget. Turnover is half the absolute pre-trade-NAV
    weight change, including cash. Solver tolerances are numerical settings.
    """

    name_weight_limit: float | None = None
    total_volatility_limit: float | None = None
    target_turnover_limit: float | None = None
    industry_weight_limit: float | None = None
    style_exposure_limit: float | None = None
    solver_ftol: float = 1e-12
    solver_maxiter: int = 1000

    def __post_init__(self) -> None:
        _validate_config(self)


@dataclass(frozen=True)
class OptimizationResult:
    """Target weights and risk in the configured reference and annualized units.

    Absolute results never label total volatility as tracking error. Missing
    total-risk inputs leave its diagnostics undefined when no such limit is set.
    """

    weights: pd.Series
    predicted_tracking_error: float | None
    factor_variance: float | None
    specific_variance: float | None
    target_turnover: float
    expected_transaction_cost: float
    maximum_adv_participation: float | None
    predicted_total_volatility: float | None = None
    risk_mode: str = "benchmark_relative"


def optimize_portfolio(
    assets: pd.DataFrame,
    *,
    factor_covariance: pd.DataFrame | None = None,
    industry_columns: tuple[str, ...] = (),
    style_columns: tuple[str, ...] = (),
    expected_transaction_cost: Callable[
        [np.ndarray, np.ndarray], tuple[float, np.ndarray, np.ndarray]
    ],
    nav: float,
    assumptions: ExecutionAssumptions | None = None,
    financial_column: str | None = None,
    cash_column: str,
    config: OptimizerConfig | AbsoluteRiskConfig | None = None,
    eligibility_column: str | None = None,
    receivable_weight: float = 0.0,
) -> OptimizationResult:
    """Maximize expected alpha less actual costs in a self-financing account.

    Holdings and explicit cash use pre-trade NAV units; cash reserves the true
    fee, while name and risk limits use post-fee NAV. The cost owner supplies
    convex side costs and marginal rates, with zero cash cost. Its smooth
    extension covers signed epigraph iterates; feasibility uses actual trades.

    AbsoluteRiskConfig selects total risk without a benchmark. Optional
    trade_capacity_weight is this decision's remaining capacity, never daily
    ADV reused at each tick; absent capacity leaves execution to the account.
    Optional sellable_weight bounds sales in pre-trade NAV units (including
    partial T+1 availability). Missing absolute-mode alpha freezes the actual
    holding; it is not a zero signal. Targets are not actual fills.
    """
    config = OptimizerConfig() if config is None else config
    absolute = isinstance(config, AbsoluteRiskConfig)
    risk_limit = (
        config.total_volatility_limit if absolute else config.tracking_error_limit
    )
    industry_limit = (
        config.industry_weight_limit if absolute else config.industry_active_limit
    )
    style_limit = config.style_exposure_limit if absolute else config.style_active_limit
    required = {
        "asset_id",
        "expected_alpha",
        "current_weight",
        *industry_columns,
        *style_columns,
    }
    if not absolute:
        required.update(("benchmark_weight", "specific_variance", "adv"))
    if not isinstance(assets, pd.DataFrame) or not required.issubset(assets.columns):
        raise ValueError("optimizer_columns_invalid")
    if not absolute and not industry_columns and not style_columns:
        raise ValueError("optimizer_exposure_columns_required")
    if not callable(expected_transaction_cost):
        raise ValueError("optimizer_cost_provider_invalid")
    if absolute and (
        (industry_limit is not None and not industry_columns)
        or (style_limit is not None and not style_columns)
    ):
        raise RiskInputsUnavailable("optimizer_exposure_columns_required")
    if not absolute and not isinstance(assumptions, ExecutionAssumptions):
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
    covariance = None
    if factor_covariance is None:
        if risk_limit is not None:
            raise RiskInputsUnavailable("optimizer_covariance_unavailable")
    else:
        if (
            tuple(factor_covariance.index) != columns
            or tuple(factor_covariance.columns) != columns
        ):
            raise ValueError("optimizer_covariance_columns_invalid")
        covariance = factor_covariance.apply(
            strict_finite_numeric_series, error="optimizer_covariance_invalid"
        ).to_numpy(dtype="float64")
        if columns:
            covariance = _validated_covariance(
                covariance, error_prefix="optimizer_covariance"
            )
    data = assets.copy().reset_index(drop=True)
    if data["asset_id"].duplicated().any():
        raise ValueError("optimizer_asset_duplicate")
    if absolute and "specific_variance" not in data:
        data["specific_variance"] = np.nan
    numeric_columns = [
        "expected_alpha",
        "current_weight",
        "specific_variance",
        *columns,
    ]
    numeric_columns.extend(
        ("trade_capacity_weight",)
        if absolute and "trade_capacity_weight" in data
        else ()
    )
    if not absolute:
        numeric_columns.extend(("benchmark_weight", "adv"))
    if "sellable_weight" in data:
        numeric_columns.append("sellable_weight")
    if absolute and "maximum_weight" in data:
        numeric_columns.append("maximum_weight")
    for column in numeric_columns:
        data[column] = strict_real_numeric_series(
            data[column], error="optimizer_numeric_invalid"
        )
    for column in (eligibility_column, financial_column, cash_column):
        if column is not None and (
            not pd.api.types.is_bool_dtype(data[column]) or data[column].isna().any()
        ):
            raise ValueError("optimizer_boolean_column_invalid")
    for column in (
        ("current_weight",) if absolute else ("current_weight", "benchmark_weight")
    ):
        if data[column].lt(0.0).any():
            raise ValueError("optimizer_weight_invalid")
    current = data["current_weight"].to_numpy()
    benchmark = None if absolute else data["benchmark_weight"].to_numpy()
    if not np.isclose(current.sum(), budget) or (
        benchmark is not None and not np.isclose(benchmark.sum(), 1.0)
    ):
        raise ValueError("optimizer_weight_sum_invalid")
    cash = data[cash_column].to_numpy(dtype=bool)
    if cash.sum() != 1 or (benchmark is not None and benchmark[cash].any()):
        raise ValueError("optimizer_cash_asset_invalid")
    sellable = current
    if "sellable_weight" in data:
        sellable = data["sellable_weight"].to_numpy(dtype="float64", copy=True)
        sellable[cash] = current[cash]
        if (
            not np.isfinite(sellable).all()
            or (sellable < 0.0).any()
            or (sellable > current).any()
        ):
            raise ValueError("optimizer_sellable_weight_invalid")
    if absolute:
        capacity = np.full(len(current), np.inf)
        if "trade_capacity_weight" in data:
            capacity[~cash] = data.loc[~cash, "trade_capacity_weight"]
            if not np.isfinite(capacity[~cash]).all() or (capacity[~cash] < 0.0).any():
                raise ValueError("optimizer_trade_capacity_invalid")
        frozen = (capacity == 0.0) & ~cash
        data.loc[cash, ["specific_variance", *columns]] = 0.0
        adv = None
    else:
        valid_adv = np.isfinite(data["adv"]) & data["adv"].gt(0.0)
        frozen = ~valid_adv.to_numpy(dtype=bool) & ~cash
        data["adv"] = data["adv"].where(valid_adv, np.inf)
        adv = data["adv"].to_numpy(dtype="float64")
        capacity = assumptions.adv_participation * adv / nav
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
    alpha_missing = ~np.isfinite(data["expected_alpha"])
    if absolute:
        constrained_missing = pd.Series(False, index=data.index)
        if risk_limit is not None:
            constrained_missing |= risk_missing
        for names, limit in (
            (industry_columns, industry_limit),
            (style_columns, style_limit),
        ):
            if limit is not None:
                constrained_missing |= data.loc[:, names].isna().any(axis=1)
        if (constrained_missing & (current > 0.0)).any():
            raise RiskInputsUnavailable("optimizer_held_risk_unavailable")
        incomplete = constrained_missing
        frozen |= alpha_missing.to_numpy(dtype=bool) & ~cash
    else:
        incomplete = risk_missing | alpha_missing
        # Frozen holdings remain real active risk. Missing volume never
        # authorizes replacing their exposures or specific variance with zero.
        cannot_reduce = frozen.copy() | (sellable == 0.0)
        if "can_exit" in data:
            cannot_reduce |= ~data.can_exit.fillna(False).to_numpy(dtype=bool)
        if (
            risk_missing
            & (current > benchmark)
            & ~_normalization_matches(current, benchmark)
            & cannot_reduce
        ).any():
            raise RiskInputsUnavailable("optimizer_held_risk_unavailable")
    data["expected_alpha"] = data["expected_alpha"].where(
        ~(incomplete | alpha_missing), 0.0
    )
    original_specific = data["specific_variance"].to_numpy(dtype="float64")
    original_exposure = data.loc[:, columns].to_numpy(dtype="float64")
    data["specific_variance"] = data["specific_variance"].where(~risk_missing, 0.0)
    exposure = np.where(np.isfinite(original_exposure), original_exposure, 0.0)
    expected_alpha = data["expected_alpha"].to_numpy(dtype="float64")
    specific = data["specific_variance"].to_numpy(dtype="float64")
    # Side costs at the maximum trades bound every feasible fee. Actual
    # self-financing constraints below still price each candidate's net trades.
    maximum_cost = expected_transaction_cost(
        np.where(cash, 0.0, np.minimum(1.0 - current, capacity)) * nav,
        np.where(cash, 0.0, np.minimum(current, capacity)) * nav,
    )[0]
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
        sellable=sellable,
    )
    tolerance = 1e-8
    if absolute and "maximum_weight" in data:
        maximum = data.maximum_weight.to_numpy(dtype="float64")
        if not np.isfinite(maximum).all() or (maximum < 0).any():
            raise ValueError("optimizer_maximum_weight_invalid")
        bounds = tuple(
            (lower, min(upper, limit))
            for (lower, upper), limit in zip(bounds, maximum, strict=True)
        )
        if any(lower > upper + tolerance for lower, upper in bounds):
            raise ValueError("optimizer_inventory_bounds_infeasible")
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
    if (
        config.target_turnover_limit is not None
        and minimum_turnover > config.target_turnover_limit + tolerance
    ):
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
        fee_direction = -cash_direction if absolute else benchmark - cash_direction
        active = weights + fee_direction * cost
        if not absolute:
            active = active - benchmark
        variance = sum(_risk_variances(active, exposure, covariance, specific))
        if not derivative:
            return risk_limit**2 * (1.0 - cost) ** 2 - variance
        factor_active = active @ exposure
        variance_gradient = (
            exposure @ (covariance @ factor_active + covariance.T @ factor_active)
            + 2.0 * specific * active
        )
        # Apply the rank-one chain rule without materializing an N-by-N matrix.
        risk_gradient = (
            variance_gradient + (variance_gradient @ fee_direction) * gradient
        )
        return np.r_[
            -2.0 * risk_limit**2 * (1.0 - cost) * gradient - risk_gradient,
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
        if absolute:
            bands = np.concatenate((-matrix.T, matrix.T))
        else:
            bands = np.concatenate(
                (
                    limit * (~cash)[None, :] - matrix.T,
                    limit * (~cash)[None, :] + matrix.T,
                )
            )
        padded = np.pad(bands, ((0, 0), (0, count)))

        def value(point):
            value = bands @ point[:count]
            if absolute:
                value += limit * (1.0 - actual_cost(point[:count])[0])
            return value

        def jacobian(point):
            if not absolute:
                return padded
            return (
                padded - np.r_[limit * actual_cost(point[:count])[1], np.zeros(count)]
            )

        return {"type": "ineq", "fun": value, "jac": jacobian}

    turnover_jacobian = np.r_[np.zeros(count), np.full(count, -0.5)]
    trade_jacobian = np.block(
        [[-np.eye(count), np.eye(count)], [np.eye(count), np.eye(count)]]
    )
    constraints = [
        {
            "type": "eq",
            "fun": lambda point: float(point[:count].sum() - budget),
            "jac": lambda point: np.r_[np.ones(count), np.zeros(count)],
        }
    ]
    if risk_limit is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": net_risk,
                "jac": lambda point: net_risk(point, derivative=True),
            }
        )
    if config.target_turnover_limit is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda point: float(
                    config.target_turnover_limit - 0.5 * point[count:].sum()
                ),
                "jac": lambda point: turnover_jacobian,
            }
        )
    for matrix, limit in (
        (relative_exposure[:, : len(industry_columns)], industry_limit),
        (relative_exposure[:, len(industry_columns) :], style_limit),
    ):
        if limit is not None:
            constraints.append(exposure_bands(matrix, limit))
    constraints.append(
        {
            "type": "ineq",
            "fun": lambda point: np.concatenate(
                (
                    point[count:] - (point[:count] - current),
                    point[count:] + (point[:count] - current),
                )
            ),
            "jac": lambda point: trade_jacobian,
        }
    )

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
            float(
                -expected_alpha
                @ (point[:count] if absolute else point[:count] - benchmark)
                + cost / nav
            ),
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
        bounds=(*bounds, *((0.0, limit) for limit in capacity)),
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
        capacity=capacity,
    )
    active = _risk_active_weights(
        net_weights, benchmark, original_exposure, original_specific
    )
    complete = (
        covariance is not None
        and np.isfinite(original_exposure[active != 0.0]).all()
        and np.isfinite(original_specific[active != 0.0]).all()
    )
    factor_variance, specific_variance = (
        _risk_variances(active, original_exposure, covariance, original_specific)
        if complete
        else (None, None)
    )
    volatility = (
        float(np.sqrt(factor_variance + specific_variance)) if complete else None
    )
    return OptimizationResult(
        weights=pd.Series(weights, index=data["asset_id"], name="weight"),
        predicted_tracking_error=None if absolute else volatility,
        factor_variance=factor_variance,
        specific_variance=specific_variance,
        target_turnover=float(0.5 * np.abs(weights - current).sum()),
        expected_transaction_cost=float(cost_ratio * nav),
        maximum_adv_participation=None
        if absolute
        else float((np.abs(weights - current) * nav / adv).max(initial=0.0)),
        predicted_total_volatility=volatility if absolute else None,
        risk_mode="absolute" if absolute else "benchmark_relative",
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
    if benchmark is None:
        return weights.copy()
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


def _normalized_risk_excess(values, limits):
    return np.maximum(0.0, values - limits - 1e-8) / np.maximum(limits, 1e-8)


def _portfolio_risk_limits(
    weights, benchmark, exposure, specific, covariance, cash, columns, config
):
    """Report post-fee risk and rank only excess beyond the configured bands."""
    active = _risk_active_weights(weights, benchmark, exposure, specific)
    nonzero = active != 0.0
    complete = (
        covariance is not None
        and np.isfinite(exposure[nonzero]).all()
        and np.isfinite(specific[nonzero]).all()
        and (specific[nonzero] >= 0.0).all()
    )
    has_stocks = weights[~cash].sum() > 0.0
    stock_positive = has_stocks and benchmark[~cash].sum() > 0.0
    factor_active = _stock_leg_active_exposure(weights, benchmark, exposure, cash)
    industry = np.array([name.startswith("industry_") for name in columns])
    exposure_complete = stock_positive and np.isfinite(factor_active).all()
    diagnostics = {
        "realized_exposure_status": "DEFINED" if exposure_complete else "UNDEFINED",
        "realized_exposure_reason": None
        if exposure_complete
        else "stock_active_exposure_unavailable"
        if stock_positive
        else "nonpositive_stock_value",
    }
    limits = {
        "tracking_error": (None, config.tracking_error_limit),
        "name_active_weight": (
            np.abs(weights[~cash] - benchmark[~cash]).max(initial=0.0),
            config.active_name_limit,
        ),
        "industry_active_exposure": (
            np.abs(factor_active[industry]).max(initial=0.0)
            if stock_positive and np.isfinite(factor_active[industry]).all()
            else None,
            config.industry_active_limit,
        ),
        "style_active_exposure": (
            np.abs(factor_active[~industry]).max(initial=0.0)
            if stock_positive and np.isfinite(factor_active[~industry]).all()
            else None,
            config.style_active_limit,
        ),
    }
    if complete:
        factor, residual = _risk_variances(active, exposure, covariance, specific)
        diagnostics.update(
            realized_factor_variance=factor, realized_specific_variance=residual
        )
        limits["tracking_error"] = (
            np.sqrt(factor + residual),
            config.tracking_error_limit,
        )
    unavailable = [name for name, (value, _) in limits.items() if value is None]
    breached = []
    excess = []
    for name, (value, limit) in limits.items():
        diagnostics[f"realized_{name}_status"] = (
            "UNDEFINED" if value is None else "DEFINED"
        )
        if value is None:
            continue
        diagnostics[f"realized_{name}"] = float(value)
        value, bound = (
            (value**2, limit**2) if name == "tracking_error" else (value, limit)
        )
        violation = max(0.0, value - bound - 1e-8)
        values = (
            np.abs(weights[~cash] - benchmark[~cash])
            if name == "name_active_weight"
            else np.abs(factor_active[industry])
            if name == "industry_active_exposure"
            else np.abs(factor_active[~industry])
            if name == "style_active_exposure"
            else np.array([value])
        )
        excess.extend(_normalized_risk_excess(values, bound))
        if violation > 0:
            breached.append(name)
    diagnostics.update(
        realized_constraints_status="UNDEFINED"
        if unavailable
        else "BREACHED"
        if breached
        else "SATISFIED",
        realized_constraints_reason=(
            "factor_risk_factor_history_insufficient"
            if covariance is None
            else "active_risk_unavailable"
            if not complete
            else "stock_active_exposure_unavailable"
            if stock_positive
            else "nonpositive_stock_value"
        )
        if unavailable
        else ",".join(breached)
        if breached
        else None,
    )
    required_unavailable = any(
        has_stocks or name not in {"industry_active_exposure", "style_active_exposure"}
        for name in unavailable
    )
    violation = (
        (np.inf, np.inf)
        if required_unavailable
        else (max(excess, default=0.0), float(np.square(excess).sum()))
    )
    return diagnostics, violation


def _bounds(
    data: pd.DataFrame,
    *,
    benchmark: np.ndarray | None,
    financial_column: str | None,
    cash_column: str | None,
    eligibility_column: str | None,
    incomplete: np.ndarray,
    frozen: np.ndarray,
    config: OptimizerConfig | AbsoluteRiskConfig,
    minimum_nav_ratio: float,
    sellable: np.ndarray,
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float], ...]]:
    bounds: list[tuple[float, float]] = []
    for position, row in data.iterrows():
        if cash_column is not None and bool(row[cash_column]):
            bounds.append((0.0, 1.0))
            continue
        if benchmark is None:
            bounds.append(
                (
                    0.0,
                    0.0
                    if incomplete[position]
                    else (
                        1.0
                        if config.name_weight_limit is None
                        else config.name_weight_limit
                    ),
                )
            )
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
        lower = max(lower, current - sellable[position])
        if (
            benchmark is None
            and eligibility_column is not None
            and not bool(row[eligibility_column])
            and not bool(row[cash_column])
        ):
            upper = min(upper, current)
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
    if benchmark is None:
        result = exposure.copy()
        result[cash] = 0.0
        return result
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
    benchmark: np.ndarray | None,
    exposure: np.ndarray,
    covariance: np.ndarray | None,
    specific: np.ndarray,
    adv: np.ndarray | None,
    config: OptimizerConfig | AbsoluteRiskConfig,
    industry_count: int,
    nav: float,
    assumptions: ExecutionAssumptions | None,
    cash: np.ndarray | None = None,
    receivable_weight: float = 0.0,
    capacity: np.ndarray | None = None,
) -> None:
    tolerance = 1e-08
    absolute = isinstance(config, AbsoluteRiskConfig)
    risk_limit = (
        config.total_volatility_limit if absolute else config.tracking_error_limit
    )
    active = _risk_active_weights(net_weights, benchmark, exposure, specific)
    risk = (
        sum(_risk_variances(active, exposure, covariance, specific))
        if risk_limit is not None
        else None
    )
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
    if risk_limit is not None and risk > risk_limit**2 + tolerance:
        raise ValueError("optimizer_solution_invalid")
    if (
        config.target_turnover_limit is not None
        and 0.5 * np.abs(weights - current).sum()
        > config.target_turnover_limit + tolerance
    ):
        raise ValueError("optimizer_solution_invalid")
    if cash is None:
        cash = np.zeros(len(weights), dtype=bool)
    if absolute:
        held = net_weights != 0.0
        for matrix, limit in (
            (exposure[:, :industry_count], config.industry_weight_limit),
            (exposure[:, industry_count:], config.style_exposure_limit),
        ):
            if limit is not None and (
                not np.isfinite(matrix[held]).all()
                or np.any(
                    np.abs(matrix[held].T @ net_weights[held]) > limit + tolerance
                )
            ):
                raise ValueError("optimizer_solution_invalid")
        if np.any(np.abs(weights - current) > capacity + tolerance):
            raise ValueError("optimizer_solution_invalid")
        return
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


def _validate_config(config: OptimizerConfig | AbsoluteRiskConfig) -> None:
    if isinstance(config, AbsoluteRiskConfig):
        limits = (
            config.name_weight_limit,
            config.total_volatility_limit,
            config.target_turnover_limit,
            config.industry_weight_limit,
            config.style_exposure_limit,
        )
        values = (*[value for value in limits if value is not None], config.solver_ftol)
    else:
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
    if isinstance(config, AbsoluteRiskConfig) and any(
        value is not None and value > 1.0
        for value in (config.name_weight_limit, config.industry_weight_limit)
    ):
        raise ValueError("optimizer_config_invalid")
    if not isinstance(config, AbsoluteRiskConfig) and (
        config.active_name_limit <= 0.0
        or config.tracking_error_limit <= 0.0
        or config.target_turnover_limit <= 0.0
    ):
        raise ValueError("optimizer_config_invalid")
    if type(config.solver_maxiter) is not int or config.solver_maxiter <= 0:
        raise ValueError("optimizer_config_invalid")


def _neutrality_bands(context, *, industry_limit, size_limit):
    """One stock-leg exposure system for target projection and actual fills."""
    b = context.benchmark_weight.to_numpy(dtype="float64")
    names = np.asarray(context.columns.drop("benchmark_weight"))
    exposures = (
        context.drop(columns="benchmark_weight").to_numpy(dtype="float64", copy=True).T
    )
    limits = np.array(
        [
            industry_limit if name.startswith("industry:") else size_limit
            for name in context.columns.drop("benchmark_weight")
        ],
        dtype="float64",
    )
    industry_rows = np.array(
        [
            name.startswith("industry:")
            for name in context.columns.drop("benchmark_weight")
        ],
        dtype=bool,
    )
    reference = exposures @ b
    denominators = np.ones_like(exposures)
    if industry_rows.any():
        known = exposures[industry_rows].sum(axis=0)
        known_reference = float(known @ b)
        if known_reference > 0:
            # Industry bands are proportions of each known-industry stock leg.
            exposures[industry_rows] -= (
                reference[industry_rows, None] / known_reference * known
            )
            reference[industry_rows] = 0.0
            denominators[industry_rows] = known
        else:
            exposures = exposures[~industry_rows]
            reference = reference[~industry_rows]
            denominators = denominators[~industry_rows]
            limits = limits[~industry_rows]
            names = names[~industry_rows]
    return exposures, reference, denominators, limits, names


def _neutrality_risk_limits(notionals, context, bands):
    exposures, reference, denominators, limits, names = bands
    selected = notionals.reindex(context.index, fill_value=0.0)
    gross = selected.sum()
    if gross <= 0.0:
        return {
            "realized_constraints_status": "UNDEFINED",
            "realized_constraints_reason": "nonpositive_stock_value",
        }, (np.inf, np.inf)
    weights = selected.to_numpy(dtype="float64") / gross
    denominators = denominators @ weights
    values = np.divide(
        np.abs(exposures @ weights - reference),
        denominators,
        out=np.full(len(limits), np.nan),
        where=denominators > 0,
    )
    known = np.isfinite(values)
    excess = _normalized_risk_excess(values[known], limits[known])
    breached = names[known][excess > 0]
    outside = notionals.loc[~notionals.index.isin(context.index)].sum() > 0.0
    unavailable = (
        outside
        or not known.all()
        or (
            "industry_known" in context.attrs
            and not any(name.startswith("industry:") for name in names)
        )
    )
    facts = {
        "realized_constraints_status": "UNDEFINED"
        if unavailable
        else ("BREACHED" if len(breached) else "SATISFIED"),
        "realized_constraints_reason": "outside_benchmark_domain"
        if outside
        else (
            "no_known_industry_weight"
            if unavailable
            else (",".join(breached) if len(breached) else None)
        ),
    }
    for prefix, metric in (("industry:", "industry"), ("size", "style")):
        mask = np.array([str(name).startswith(prefix) for name in names])
        if mask.any():
            defined = bool(known[mask].all())
            facts[f"realized_{metric}_active_exposure_status"] = (
                "DEFINED" if defined else "UNDEFINED"
            )
            if defined:
                facts[f"realized_{metric}_active_exposure"] = float(values[mask].max())
    return facts, (max(excess, default=0.0), float(np.square(excess).sum()))


def _project_neutral_target(
    target, context, *, industry_limit=0.02, size_limit=0.2, fixed_weights=None
):
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
    exposures, reference, denominators, limits, _ = _neutrality_bands(
        context, industry_limit=industry_limit, size_limit=size_limit
    )
    exact = exposures[limits == 0]
    bounded = exposures[limits > 0]
    bands = limits[limits > 0]
    upper_matrix = np.vstack((bounded, -bounded)) - np.tile(
        bands[:, None] * denominators[limits > 0], (2, 1)
    )
    upper_values = np.concatenate((reference[limits > 0], -reference[limits > 0]))
    matrix = np.vstack((np.ones(len(b)), exact))
    right_hand_side = np.concatenate(([1.0], reference[limits == 0]))
    eligible = context.attrs.get("eligible")
    bounds = [(0.0, None)] * len(b)
    if eligible is not None:
        eligible = eligible.reindex(context.index, fill_value=False)
        bounds = [(0.0, None if value else 0.0) for value in eligible]
    if fixed_weights is not None:
        for asset, value in fixed_weights.items():
            position = context.index.get_loc(asset)
            if bounds[position][1] == 0.0 and value != 0.0:
                raise PortfolioTargetBlocked("target_neutrality_infeasible")
            bounds[position] = (float(value), float(value))

    def valid_weights(weights):
        return (
            np.isfinite(weights).all()
            and np.allclose(matrix @ weights, right_hand_side, rtol=0.0, atol=1e-8)
            and not np.any(
                np.abs(exposures @ weights - reference)
                > (limits + 1e-8) * (denominators @ weights)
            )
            and all(
                value >= lower - 1e-8 and (upper is None or value <= upper + 1e-8)
                for value, (lower, upper) in zip(weights, bounds, strict=True)
            )
        )

    # A feasible raw target is the exact minimizer; avoid inventing tiny positions.
    if valid_weights(q):
        return pd.Series(q, index=context.index, dtype="float64", name="target_weight")
    fixed = np.array([lower == upper for lower, upper in bounds])
    fixed_values = np.array([lower for lower, _ in bounds])[fixed]
    free_matrix = matrix.copy()
    free_matrix[:, fixed] = 0.0
    free_rhs = right_hand_side - matrix[:, fixed] @ fixed_values
    # Remove only algebraically redundant equalities; retain their residual checks.
    left, singular, vectors = np.linalg.svd(free_matrix, full_matrices=False)
    rank = np.count_nonzero(
        singular > singular[0] * max(matrix.shape) * np.finfo(float).eps
    )
    constraints = vectors[:rank]
    values = (left[:, :rank].T @ free_rhs) / singular[:rank]
    if not np.allclose(
        free_matrix @ (constraints.T @ values), free_rhs, rtol=0.0, atol=1e-8
    ):
        raise PortfolioTargetBlocked("target_neutrality_infeasible")
    optimizer_constraints = (
        [
            {
                "type": "eq",
                "fun": lambda w: constraints @ w - values,
                "jac": lambda w: constraints,
            }
        ]
        if rank
        else []
    )
    if len(bands):
        optimizer_constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: upper_values - upper_matrix @ w,
                "jac": lambda w: -upper_matrix,
            }
        )
    initial = b / b.sum()
    if fixed.any():
        feasible = linprog(
            np.zeros(len(b)),
            A_ub=upper_matrix if len(bands) else None,
            b_ub=upper_values if len(bands) else None,
            A_eq=constraints if rank else None,
            b_eq=values if rank else None,
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
    if not valid_weights(weights):
        raise PortfolioTargetBlocked("target_neutrality_solution_invalid")
    return pd.Series(
        weights, index=context.index, dtype="float64", name="target_weight"
    )
