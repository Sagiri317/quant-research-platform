"""Sparse self-financing targets with relative or explicit absolute risk limits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import qr
from scipy.optimize import (
    Bounds,
    LinearConstraint,
    NonlinearConstraint,
    OptimizeResult,
    brentq,
    linprog,
    minimize,
)
from scipy.sparse.linalg import LinearOperator, lsmr

from quant_research_platform.contracts.portfolios import (
    ExecutionAssumptions,
    PortfolioObjective,
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
    solver_gtol: float = 1e-10
    solver_xtol: float = 1e-12
    solver_barrier_tol: float = 1e-12
    solver_maxiter: int | None = 1000

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
    solver_gtol: float = 1e-10
    solver_xtol: float = 1e-12
    solver_barrier_tol: float = 1e-12
    solver_maxiter: int | None = 1000
    fully_invested: bool = False

    def __post_init__(self) -> None:
        _validate_config(self)


@dataclass(frozen=True)
class OptimizationResult:
    """Target weights and risk in the configured reference and annualized units.

    Absolute results never label total volatility as tracking error. Missing
    total-risk inputs leave its diagnostics undefined when no such limit is set.
    expected_return is calibrated cross-sectional alpha, not total market
    return. Solver residuals describe the terminated iterate before exact affine
    and actual-fee closure; economic diagnostics use the returned target weights.
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
    expected_return: float | None = None
    robust_penalty: float | None = None
    holding_variance: float | None = None
    risk_penalty: float | None = None
    objective_value: float | None = None
    solver_iterations: int | None = None
    solver_variables: int | None = None
    solver_optimality: float | None = None
    solver_constraint_violation: float | None = None
    solver_complementarity: float | None = None
    solver_complementarity_sum: float | None = None
    solver_complementarity_count: int | None = None
    solver_termination: str | None = None


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
    objective_kind: PortfolioObjective = PortfolioObjective.CALIBRATED,
    uncertainty_loading: pd.Series | None = None,
    risk_aversion: float = 0.0,
    holding_sessions: int | None = None,
    risk_annualization: float = 252.0,
) -> OptimizationResult:
    """Optimize calibrated alpha, weighted rank or its annualized risk ratio.

    Rank objectives consume prediction_rank_score and preserve actual fee funding.

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

    uncertainty_loading is the asset-indexed, confidence-scaled loading of a
    rank-one mean-parameter ellipsoid, in decimal holding-return units. Its
    support function is abs(loading @ active_weights). A positive risk_aversion
    adds half that coefficient times annualized variance scaled by the explicit
    holding_sessions/risk_annualization; it does not resize the uncertainty set.
    """
    config = OptimizerConfig() if config is None else config
    objective_kind = PortfolioObjective(objective_kind)
    net_return = objective_kind == PortfolioObjective.NET_RETURN_TO_RISK
    if (
        isinstance(risk_aversion, bool)
        or not isinstance(risk_aversion, int | float)
        or not np.isfinite(risk_aversion)
        or risk_aversion < 0.0
        or (
            (risk_aversion > 0.0 or net_return)
            and (
                type(holding_sessions) is not int
                or holding_sessions <= 0
                or isinstance(risk_annualization, bool)
                or not isinstance(risk_annualization, int | float)
                or not np.isfinite(risk_annualization)
                or risk_annualization <= 0.0
            )
        )
    ):
        raise ValueError("optimizer_risk_horizon_invalid")
    absolute = isinstance(config, AbsoluteRiskConfig)
    if net_return and not (absolute and config.fully_invested):
        raise ValueError("optimizer_net_return_ratio_requires_funded_account")
    ranks = objective_kind.uses_rank
    ratio = objective_kind.is_ratio
    no_positive_score = (
        "optimizer_no_positive_feasible_net_return"
        if net_return
        else "optimizer_no_positive_feasible_rank"
    )
    if (ranks or net_return) and (
        not absolute or risk_aversion or uncertainty_loading is not None
    ):
        raise ValueError("optimizer_rank_objective_configuration_invalid")
    signal_column = objective_kind.signal_column
    risk_limit = (
        config.total_volatility_limit if absolute else config.tracking_error_limit
    )
    industry_limit = (
        config.industry_weight_limit if absolute else config.industry_active_limit
    )
    style_limit = config.style_exposure_limit if absolute else config.style_active_limit
    required = {
        "asset_id",
        signal_column,
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
        if risk_limit is not None or risk_aversion > 0.0 or ratio:
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
        signal_column,
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
    risk_missing, alpha_missing, incomplete = _portfolio_incomplete(
        data,
        industry_columns,
        style_columns,
        config,
        risk_aversion > 0.0 or ratio,
        signal_column,
    )
    if absolute:
        # An enforced zero target has no signal or risk contribution. Only a
        # holding that can actually be sold in full may use that observation.
        zero_target = np.zeros(len(data), dtype=bool)
        if ratio:
            zero_target |= incomplete
            if "maximum_weight" in data:
                zero_target |= data.maximum_weight.eq(0.0)
        can_liquidate = (sellable >= current) & (capacity >= current)
        if "can_exit" in data:
            can_liquidate &= data.can_exit.fillna(False).to_numpy(dtype=bool)
        required = (current > 0.0) & ~(zero_target & can_liquidate)
        if (ranks or net_return) and (alpha_missing & required).any():
            raise RiskInputsUnavailable("robust_held_prediction_unavailable")
        if (incomplete & required).any():
            raise RiskInputsUnavailable("optimizer_held_risk_unavailable")
        frozen |= alpha_missing.to_numpy(dtype=bool) & ~zero_target & ~cash
    else:
        if risk_aversion > 0.0 and (risk_missing & (benchmark > 0.0)).any():
            raise RiskInputsUnavailable("optimizer_held_risk_unavailable")
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
    expected_alpha = (
        data[signal_column]
        .where(~(incomplete | alpha_missing), 0.0)
        .to_numpy(dtype="float64")
    )
    original_specific = data["specific_variance"].to_numpy(dtype="float64")
    original_exposure = data.loc[:, columns].to_numpy(dtype="float64")
    data["specific_variance"] = data["specific_variance"].where(~risk_missing, 0.0)
    exposure = np.where(np.isfinite(original_exposure), original_exposure, 0.0)
    specific = data["specific_variance"].to_numpy(dtype="float64")
    loading = np.zeros(len(data))
    if uncertainty_loading is not None:
        if (
            not isinstance(uncertainty_loading, pd.Series)
            or not uncertainty_loading.index.is_unique
        ):
            raise ValueError("optimizer_uncertainty_loading_invalid")
        loading = strict_finite_numeric_series(
            uncertainty_loading.reindex(data.asset_id),
            error="optimizer_uncertainty_loading_invalid",
        ).to_numpy(dtype="float64")
        if loading[cash].any():
            raise ValueError("optimizer_cash_uncertainty_invalid")
    horizon_scale = (
        holding_sessions / risk_annualization
        if risk_aversion > 0.0 or net_return
        else 0.0
    )
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
    frozen_stock = (lower == upper) & ~cash
    fee_lower, fee_upper = 0.0, maximum_cost / nav
    for position in np.flatnonzero(frozen_stock):
        fixed_value = lower[position]
        if net_lower[position] > 0.0 and not _normalization_matches(
            fixed_value, net_lower[position]
        ):
            fee_lower = max(fee_lower, 1.0 - fixed_value / net_lower[position])
        if net_upper[position] > 0.0:
            cap = (
                0.0
                if _normalization_matches(fixed_value, net_upper[position])
                else 1.0 - fixed_value / net_upper[position]
            )
            fee_upper = min(fee_upper, cap)
    if fee_upper < fee_lower:
        raise ValueError("optimizer_frozen_constraints_infeasible")
    if fee_lower == fee_upper:
        # A physical holding at its post-fee ceiling can force fees to zero.
        # Positive marginal fees then prove that trade direction unavailable.
        if fee_upper == 0.0 and maximum_cost > 0.0:
            _, buy_gradient, sell_gradient = expected_transaction_cost(
                np.zeros(len(current)), np.zeros(len(current))
            )
            upper = np.where(
                (buy_gradient > 0.0) & ~cash, np.minimum(upper, current), upper
            )
            lower = np.where(
                (sell_gradient > 0.0) & ~cash, np.maximum(lower, current), lower
            )
        adjustable = ~cash & ~frozen_stock
        lower[adjustable] = np.maximum(
            lower[adjustable], net_lower[adjustable] * (1.0 - fee_lower)
        )
        upper[adjustable] = np.minimum(
            upper[adjustable], net_upper[adjustable] * (1.0 - fee_lower)
        )
        bounds = tuple(zip(lower, upper, strict=True))
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
    cash_direction = cash.astype("float64")
    reference = np.zeros(count) if absolute else benchmark
    free_weights = np.flatnonzero(reachable_lower != reachable_upper)
    one_sided = (reachable_lower >= current) | (reachable_upper <= current)
    free_trades = np.flatnonzero(~one_sided)
    if config.target_turnover_limit is None:
        # Cash has no side cost; without turnover its trade auxiliary is unused.
        free_trades = free_trades[~cash[free_trades]]
    if (ranks or maximum_cost == 0.0) and config.target_turnover_limit is None:
        free_trades = np.empty(0, dtype=int)
    trade_sign = np.where(reachable_lower >= current, 1.0, -1.0)
    fixed_weights = np.clip(current, reachable_lower, reachable_upper)
    loading_scale = float(np.abs(loading).max(initial=0.0))
    direction = loading / loading_scale if loading_scale > 0.0 else loading
    weight_count, trade_count = len(free_weights), len(free_trades)
    fee_variable = fee_lower < fee_upper
    fee_position = weight_count + trade_count
    variable_count = fee_position + int(fee_variable) + int(loading_scale > 0.0)

    def unpack(point):
        weights = fixed_weights.copy()
        weights[free_weights] = point[:weight_count]
        trades = np.abs(weights - current)
        trades[free_trades] = point[weight_count:fee_position]
        fee = float(point[fee_position]) if fee_variable else fee_lower
        return weights, trades, fee

    def gradient_point(
        weight_gradient, trade_gradient=None, *, fee_gradient=0.0, robust_gradient=0.0
    ):
        result = np.zeros(variable_count)
        result[:weight_count] = weight_gradient[free_weights]
        if trade_gradient is not None:
            result[:weight_count] += (trade_gradient * trade_sign * one_sided)[
                free_weights
            ]
            result[weight_count:fee_position] = trade_gradient[free_trades]
        if fee_variable:
            result[fee_position] = fee_gradient
        if loading_scale > 0.0:
            result[-1] = robust_gradient
        return result

    def actual_cost(weights):
        delta = weights - current
        cost, buy_gradient, sell_gradient = expected_transaction_cost(
            np.maximum(delta, 0.0) * nav, np.maximum(-delta, 0.0) * nav
        )
        buying = (delta > 0.0) | ((delta == 0.0) & one_sided & (trade_sign > 0.0))
        selling = (delta < 0.0) | ((delta == 0.0) & one_sided & (trade_sign < 0.0))
        gradient = np.where(
            buying,
            buy_gradient,
            np.where(selling, -sell_gradient, (buy_gradient - sell_gradient) / 2.0),
        )
        return cost / nav, gradient

    def covariance_product(active):
        factor = active @ exposure
        return (
            exposure @ ((covariance + covariance.T) @ factor) / 2.0 + specific * active
        )

    # Express the holding-period numerator per unit of annualized volatility.
    score_scale = np.sqrt(horizon_scale) if net_return else 1.0
    priced_ratio = net_return and maximum_cost > 0.0
    signal_gradient = gradient_point(expected_alpha) / score_scale

    def positive_income(weights, fee):
        gross = expected_alpha @ weights
        if not net_return:
            return gross > 0.0
        # Gross return and aggregate fees accumulate independently at break-even.
        rounding = np.finfo(float).eps * (np.count_nonzero(weights) + count + 2)
        return gross - fee > rounding * (
            np.abs(expected_alpha) @ np.abs(weights) + abs(fee)
        )

    def negative_score(point):
        if priced_ratio:
            # Price the proposed trades directly, including trial points where
            # the reserved-fee equality has not yet converged.
            value, gradient = objective(point)
            return value / score_scale, gradient / score_scale
        weights, _, _ = unpack(point)
        return -float(expected_alpha @ weights) / score_scale, -signal_gradient

    def ratio_terms(point):
        weights, _, _ = unpack(point)
        loss, gradient = negative_score(point)
        score, slope = -loss, -gradient
        variance = sum(_risk_variances(weights, exposure, covariance, specific))
        # q / sqrt(1 + q^2) preserves ratio ordering and is smooth at zero variance
        # for a positive score. No artificial volatility enters the reported q.
        scale = np.hypot(score, np.sqrt(variance))
        direction = (
            gradient_point(covariance_product(weights)) + score * slope
        ) / scale
        return score / scale, direction, scale, slope

    def signal_ratio(point):
        value, direction, scale, slope = ratio_terms(point)
        return -value, (value * direction - slope) / scale

    def ratio_hessian(point):
        value, direction, scale, slope = ratio_terms(point)

        def product(vector):
            delta = np.zeros(count)
            delta[free_weights] = vector[:weight_count]
            ds, dv = slope @ vector, direction @ vector
            curvature = (
                slope * dv
                + direction * ds
                + value
                * (
                    gradient_point(covariance_product(delta))
                    + slope * ds
                    - 3 * direction * dv
                )
            )
            return curvature / scale**2

        return LinearOperator((variable_count, variable_count), matvec=product)

    def zero_hessian(point):
        return sparse.csr_matrix((variable_count, variable_count))

    net_trades = ~one_sided
    net_trades[free_trades] = False

    def side_cost(point):
        weights, trades, _ = unpack(point)
        delta = weights - current
        trades[one_sided] = trade_sign[one_sided] * delta[one_sided]
        cost, buy_gradient, sell_gradient = expected_transaction_cost(
            ((trades + delta) / 2.0) * nav, ((trades - delta) / 2.0) * nav
        )
        spread = (buy_gradient + sell_gradient) / 2.0
        gradient = (buy_gradient - sell_gradient) / 2.0
        gradient[net_trades] += np.sign(delta[net_trades]) * spread[net_trades]
        return cost / nav, gradient_point(gradient, spread)

    def objective(point):
        weights, _, _ = unpack(point)
        cost, cost_gradient = side_cost(point)
        active = weights - reference
        penalty, risk_gradient = 0.0, np.zeros(count)
        if risk_aversion > 0.0:
            penalty = (
                0.5
                * risk_aversion
                * horizon_scale
                * sum(_risk_variances(active, exposure, covariance, specific))
            )
            risk_gradient = risk_aversion * horizon_scale * covariance_product(active)
        robust = loading_scale * point[-1] if loading_scale > 0.0 else 0.0
        return float(-expected_alpha @ active + cost + penalty + robust), (
            cost_gradient
            + gradient_point(
                -expected_alpha + risk_gradient, robust_gradient=loading_scale
            )
        )

    def objective_hessian(point):
        def product(vector):
            norm = np.linalg.norm(vector)
            if norm == 0.0:
                return np.zeros(variable_count)
            step = np.cbrt(np.finfo(float).eps) * (1.0 + np.linalg.norm(point)) / norm
            value = (
                side_cost(point + step * vector)[1]
                - side_cost(point - step * vector)[1]
            ) / (2.0 * step)
            if risk_aversion > 0.0:
                weights = np.zeros(count)
                weights[free_weights] = vector[:weight_count]
                value += gradient_point(
                    risk_aversion * horizon_scale * covariance_product(weights)
                )
            return value

        return LinearOperator((variable_count, variable_count), matvec=product)

    constraints = []
    fixed_part = fixed_weights.copy()
    fixed_part[free_weights] = 0.0

    def add_linear(
        matrix, lower_value, upper_value, *, fee=0.0, trades=None, robust=0.0
    ):
        matrix = sparse.csr_matrix(matrix, dtype="float64")
        rows = matrix.shape[0]
        constant = matrix @ fixed_part
        if not fee_variable:
            constant += np.broadcast_to(fee, (rows,)) * fee_lower
        parts = [
            matrix[:, free_weights],
            sparse.csr_matrix((rows, trade_count)) if trades is None else trades,
        ]
        if fee_variable:
            parts.append(sparse.csr_matrix(np.broadcast_to(fee, (rows,))[:, None]))
        if loading_scale > 0.0:
            parts.append(sparse.csr_matrix(np.broadcast_to(robust, (rows,))[:, None]))
        coefficients = sparse.hstack(parts, format="csr")
        moving = coefficients.getnnz(axis=1) > 0
        # Constant rows remain covered by the original economic post-checks.
        if moving.any():
            constraint = LinearConstraint(
                coefficients[moving],
                (np.broadcast_to(lower_value, (rows,)) - constant)[moving],
                (np.broadcast_to(upper_value, (rows,)) - constant)[moving],
            )
            constraints.append(constraint)
            return constraint

    add_linear(np.ones((1, count)), budget, budget)
    stock = np.flatnonzero(~cash)
    selector = sparse.csr_matrix(
        (np.ones(len(stock)), (np.arange(len(stock)), stock)), shape=(len(stock), count)
    )
    cash_fee_floor = None
    if maximum_cost > 0.0:
        exact_names = net_lower[stock] == net_upper[stock]
        add_linear(
            selector[exact_names],
            net_lower[stock][exact_names],
            net_lower[stock][exact_names],
            fee=net_lower[stock][exact_names],
        )
        lower_names = (net_lower[stock] > 0.0) & ~exact_names
        upper_names = (net_upper[stock] < 1.0) & ~exact_names
        add_linear(
            selector[lower_names],
            net_lower[stock][lower_names],
            np.inf,
            fee=net_lower[stock][lower_names],
        )
        add_linear(
            selector[upper_names],
            -np.inf,
            net_upper[stock][upper_names],
            fee=net_upper[stock][upper_names],
        )
        cash_fee_floor = add_linear(cash_direction[None, :], 0.0, np.inf, fee=-1.0)
    if trade_count:
        selection = sparse.csr_matrix(
            (np.ones(trade_count), (np.arange(trade_count), free_trades)),
            shape=(trade_count, count),
        )
        add_linear(
            sparse.vstack((-selection, selection)),
            np.r_[-current[free_trades], current[free_trades]],
            np.inf,
            trades=sparse.vstack(
                (sparse.eye(trade_count), sparse.eye(trade_count)), format="csr"
            ),
        )
    if config.target_turnover_limit is not None:
        add_linear(
            (0.5 * trade_sign * one_sided)[None, :],
            -np.inf,
            config.target_turnover_limit + 0.5 * (trade_sign * one_sided) @ current,
            trades=sparse.csr_matrix(np.full((1, trade_count), 0.5)),
        )
    for matrix, limit in (
        (relative_exposure[:, : len(industry_columns)], industry_limit),
        (relative_exposure[:, len(industry_columns) :], style_limit),
    ):
        if limit is None:
            continue
        if limit == 0.0:
            add_linear(matrix.T, 0.0, 0.0)
            continue
        if absolute:
            add_linear(np.vstack((-matrix.T, matrix.T)), -limit, np.inf, fee=-limit)
        else:
            add_linear(
                np.vstack(
                    (
                        limit * (~cash)[None, :] - matrix.T,
                        limit * (~cash)[None, :] + matrix.T,
                    )
                ),
                0.0,
                np.inf,
            )
    if loading_scale > 0.0:
        add_linear(
            np.vstack((direction, -direction)),
            -np.inf,
            [direction @ reference, -direction @ reference],
            robust=-1.0,
        )
    unknown_stock_reference = not absolute and np.any(
        ~np.isfinite(original_exposure).all(axis=1) & (benchmark > 0.0)
    )
    if unknown_stock_reference:
        if receivable_weight > 0.0:
            raise RiskInputsUnavailable("optimizer_stock_exposure_unavailable")
        add_linear(cash_direction[None, :], 0.0, 0.0, fee=-1.0)
    affine_solution = None
    equalities = [item for item in constraints if np.array_equal(item.lb, item.ub)]
    if equalities:
        matrix = sparse.vstack([item.A for item in equalities], format="csr")
        values = np.concatenate([item.lb for item in equalities])
        # Equality count is the low-dimensional exposure/fixed-holding domain.
        _, triangular, pivots = qr(matrix.T.toarray(), mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(triangular))
        rank = int(
            np.count_nonzero(
                diagonal
                > diagonal.max(initial=0.0) * max(matrix.shape) * np.finfo(float).eps
            )
        )
        chosen = pivots[:rank]
        coefficients = np.linalg.lstsq(
            matrix[chosen].toarray().T, matrix.toarray().T, rcond=None
        )[0]
        discrepancy = coefficients.T @ values[chosen] - values
        resolution = (
            32
            * np.finfo(float).eps
            * max(matrix.shape)
            * (
                np.linalg.norm(values, ord=np.inf)
                + np.linalg.norm(coefficients, ord=1)
                * np.linalg.norm(values[chosen], ord=np.inf)
            )
        )
        if np.any(np.abs(discrepancy) > resolution):
            raise ValueError("optimizer_frozen_constraints_infeasible")
        constraints = [
            item for item in constraints if not np.array_equal(item.lb, item.ub)
        ]
        if rank:
            constraints.insert(
                0, LinearConstraint(matrix[chosen], values[chosen], values[chosen])
            )
        physical = np.r_[
            np.arange(weight_count), [fee_position] if fee_variable else []
        ].astype(int)
        if rank == len(physical):
            affine_solution = np.linalg.solve(
                matrix[chosen][:, physical].toarray(), values[chosen]
            )
    if maximum_cost > 0.0:

        def net_cost(point):
            weights, _, _ = unpack(point)
            cost, gradient = actual_cost(weights)
            return cost, gradient_point(gradient)

        def fee_constraint(cost, upper):
            fee_gradient = gradient_point(np.zeros(count), fee_gradient=1.0)

            def value(point):
                return unpack(point)[2] - cost(point)[0]

            def jacobian(point):
                return sparse.csr_matrix((fee_gradient - cost(point)[1])[None, :])

            return NonlinearConstraint(value, 0.0, upper, jac=jacobian, hess="2-point")

        fee_row = len(constraints)
        actual_fee = fee_constraint(net_cost, 0.0)
        constraints.append(actual_fee)
    # Constant risk is checked in the original units after solving, like fixed
    # linear rows; an infeasible zero-Jacobian constraint cannot guide a step.
    if risk_limit is not None and (
        fee_variable
        or np.any(specific[free_weights])
        or np.any(exposure[free_weights] @ covariance)
    ):
        fee_direction = -cash_direction if absolute else benchmark - cash_direction

        def net_risk(point):
            weights, _, fee = unpack(point)
            active = weights + fee_direction * fee - reference
            return risk_limit**2 * (1.0 - fee) ** 2 - sum(
                _risk_variances(active, exposure, covariance, specific)
            )

        def net_risk_jacobian(point):
            weights, _, fee = unpack(point)
            active = weights + fee_direction * fee - reference
            variance_gradient = 2.0 * covariance_product(active)
            return sparse.csr_matrix(
                gradient_point(
                    -variance_gradient,
                    fee_gradient=-2.0 * risk_limit**2 * (1.0 - fee)
                    - variance_gradient @ fee_direction,
                )[None, :]
            )

        constraints.append(
            NonlinearConstraint(
                net_risk, 0.0, np.inf, jac=net_risk_jacobian, hess="2-point"
            )
        )

    def pack(weights):
        return np.r_[
            weights[free_weights],
            np.abs(weights - current)[free_trades],
            [actual_cost(weights)[0]] if fee_variable else [],
            [abs(direction @ (weights - reference))] if loading_scale > 0.0 else [],
        ]

    initial = pack(fixed_weights)
    if affine_solution is not None:
        initial[:weight_count] = affine_solution[:weight_count]
        if fee_variable:
            initial[fee_position] = affine_solution[-1]
        affine_weights, _, _ = unpack(initial)
        initial[weight_count:fee_position] = np.abs(affine_weights - current)[
            free_trades
        ]
        if loading_scale > 0.0:
            initial[-1] = abs(direction @ (affine_weights - reference))
    lower_bounds = np.r_[
        reachable_lower[free_weights],
        np.full(trade_count, -np.inf),
        [fee_lower] if fee_variable else [],
        [-np.inf] if loading_scale > 0.0 else [],
    ]
    upper_bounds = np.r_[
        reachable_upper[free_weights],
        np.full(trade_count, np.inf),
        [fee_upper] if fee_variable else [],
        [np.inf] if loading_scale > 0.0 else [],
    ]

    def feasible(point, atol=tolerance):
        for constraint in constraints:
            values = (
                constraint.A @ point
                if isinstance(constraint, LinearConstraint)
                else constraint.fun(point)
            )
            if np.any(values < constraint.lb - atol) or np.any(
                values > constraint.ub + atol
            ):
                return False
        return np.all(point >= lower_bounds - atol) and np.all(
            point <= upper_bounds + atol
        )

    # Preserve the no-trade optimum on flat objectives using a global support
    # certificate, rather than letting an interior barrier create gratuitous fills.
    no_risk_gain = (
        risk_aversion == 0.0
        or sum(_risk_variances(current - reference, exposure, covariance, specific))
        == 0.0
    )
    no_robust_loss = loading @ (current - reference) == 0.0
    mean_maximum = np.max(expected_alpha)
    mean_at_maximum = np.all(expected_alpha[current > 0.0] == mean_maximum)
    certified = no_risk_gain and no_robust_loss and mean_at_maximum
    no_positive_return = not np.any(expected_alpha[reachable_upper > 0.0] > 0.0)
    if not current[~cash].any() and no_risk_gain and no_robust_loss:
        _, buying_gradient, _ = expected_transaction_cost(
            np.zeros(count), np.zeros(count)
        )
        required_mean = expected_alpha if ranks else expected_alpha - buying_gradient
        available = (~cash) & (reachable_upper > 0.0)
        if net_return:
            no_positive_return = np.all(required_mean[available] <= 0.0)
        positive, negative, zero = (
            available & (loading > 0.0),
            available & (loading < 0.0),
            available & (loading == 0.0),
        )
        lower_theta = max(
            -1.0, np.max(required_mean[positive] / loading[positive], initial=-1.0)
        )
        upper_theta = min(
            1.0, np.min(required_mean[negative] / loading[negative], initial=1.0)
        )
        certified |= np.all(required_mean[zero] <= 0.0) and lower_theta <= upper_theta
    if ratio:
        certified = not np.any(expected_alpha[reachable_upper > 0.0])
    complementarity = 0.0
    complementarity_sum = 0.0
    complementarity_count = 0
    projected_optimality = 0.0

    def numerical_convergence(point, state):
        nonlocal complementarity
        nonlocal complementarity_sum, complementarity_count
        nonlocal projected_optimality
        complementarity = 0.0
        complementarity_sum, complementarity_count = 0.0, 0
        lagrangian = state.grad.copy()
        for owner, values, multiplier, jacobian in zip(
            [*constraints, Bounds(lower_bounds, upper_bounds)],
            state.constr,
            state.v,
            state.jac,
            strict=True,
        ):
            lower_value = np.broadcast_to(owner.lb, np.shape(values))
            upper_value = np.broadcast_to(owner.ub, np.shape(values))
            projected = np.clip(
                multiplier,
                np.where(np.isfinite(lower_value), -np.inf, 0.0),
                np.where(np.isfinite(upper_value), np.inf, 0.0),
            )
            lagrangian += jacobian.T @ projected
            inequality = lower_value != upper_value
            for selected, dual, slack in (
                (
                    inequality & np.isfinite(lower_value),
                    np.maximum(-projected, 0.0),
                    values - lower_value,
                ),
                (
                    inequality & np.isfinite(upper_value),
                    np.maximum(projected, 0.0),
                    upper_value - values,
                ),
            ):
                residual = abs(dual[selected] * slack[selected])
                complementarity_count += int(selected.sum())
                complementarity_sum += float(residual.sum())
                complementarity = max(complementarity, float(residual.max(initial=0.0)))
        projected_optimality = float(abs(lagrangian).max(initial=0.0))
        return (
            projected_optimality <= config.solver_gtol
            and state.constr_violation <= config.solver_gtol
            and complementarity_sum <= complementarity_count * config.solver_gtol
        )

    def solve(fun, hess, start, *, barrier=0.1, witness=None):
        accepted = False

        def stop(point, state):
            nonlocal accepted
            accepted = numerical_convergence(point, state) or (
                witness is not None
                and state.constr_violation <= config.solver_gtol
                and witness(point)
            )
            return accepted

        solver = _minimize_in_percentage_points if priced_ratio else minimize
        solution = solver(
            fun,
            start,
            jac=True,
            hess=hess,
            method="trust-constr",
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=constraints,
            callback=stop,
            options={
                "gtol": 0.0,
                "xtol": config.solver_xtol,
                "barrier_tol": config.solver_barrier_tol,
                "maxiter": np.inf
                if config.solver_maxiter is None
                else config.solver_maxiter,
                "initial_barrier_parameter": barrier,
                "initial_barrier_tolerance": barrier,
                "sparse_jacobian": True,
                "factorization_method": "AugmentedSystem",
            },
        )
        if not accepted:
            error = ValueError("optimizer_failed_closed")
            error.add_note(
                f"stage={fun.__name__}; {solution.message}; iterations={solution.nit}; "
                f"projected_gradient={projected_optimality}; "
                f"constraint_violation={solution.constr_violation}; "
                f"complementarity_sum={complementarity_sum}; "
                f"complementarity_count={complementarity_count}"
            )
            raise error
        return solution

    analytic_solution = not weight_count or (
        affine_solution is not None and feasible(initial)
    )
    funding_iterations = 0
    if absolute and config.fully_invested and not analytic_solution:
        # Maximize invested principal; fees paid are never a way to buy exposure.
        # Once fully funded, different allocations can reserve their own true fee.
        cash_gradient = gradient_point(cash_direction)

        def funded_weights(fee, buyable):
            ceiling = np.clip(net_upper * (1.0 - fee), reachable_lower, reachable_upper)
            weights = np.clip(
                np.maximum(current, net_lower * (1.0 - fee)), reachable_lower, ceiling
            )
            room = np.where(buyable & ~cash, ceiling - weights, 0.0)
            remaining = budget - fee - weights[stock].sum()
            capacity_left = room.sum()
            if remaining > 0.0 and capacity_left > 0.0:
                weights += room * min(1.0, remaining / capacity_left)
            weights[cash] = budget - weights[stock].sum()
            return weights

        # A feasible allocation paying its own fee proves full funding.
        full_funding = False
        for buyable in (expected_alpha > 0.0, ~cash) if ratio else (~cash,):
            fee = (
                brentq(
                    lambda paid: paid - actual_cost(funded_weights(paid, buyable))[0],
                    0.0,
                    maximum_cost / nav,
                )
                if maximum_cost
                else 0.0
            )
            weights = funded_weights(fee, buyable)
            candidate = pack(weights)
            if abs(weights[cash].item() - fee) <= config.solver_gtol and feasible(
                candidate, config.solver_gtol
            ):
                initial = candidate
                full_funding = True
                break
        start_weights, _, start_fee = unpack(initial)
        if start_weights[cash].item() - start_fee > config.solver_gtol or not feasible(
            initial, config.solver_gtol
        ):
            # While maximizing invested principal, reserving more than the
            # true fee only tightens long-only cash, name and risk limits.
            if maximum_cost > 0.0:
                constraints[fee_row] = fee_constraint(side_cost, np.inf)

            def reserved_cash(point):
                return unpack(point)[0][cash].item(), cash_gradient

            funding = solve(reserved_cash, zero_hessian, initial)
            initial = funding.x.copy()
            if maximum_cost > 0.0:
                constraints[fee_row] = actual_fee
                if fee_variable:
                    initial[fee_position] = actual_cost(unpack(initial)[0])[0]
            funding_iterations = funding.nit
        allocated, _, fee = unpack(initial)
        idle_cash = max(0.0, allocated[cash].item() - fee)
        exact_cash = full_funding or (
            idle_cash == 0.0 and feasible(initial, config.solver_gtol)
        )
        if cash_fee_floor is not None:
            constraints.remove(cash_fee_floor)
        add_linear(
            cash_direction[None, :],
            0.0 if exact_cash or maximum_cost else -np.inf,
            0.0 if exact_cash else idle_cash + config.solver_gtol,
            fee=-1.0,
        )
        if exact_cash:
            # The funding equality owns cash; an identical bound has no extra meaning.
            lower_bounds[:weight_count][cash[free_weights]] = -np.inf

    analytic_solution |= (
        certified and np.array_equal(unpack(initial)[0], current) and feasible(initial)
    )
    if net_return and no_positive_return:
        raise RiskInputsUnavailable(no_positive_score)
    if not analytic_solution and not ratio and (ranks or maximum_cost == 0.0):
        # The budget-and-box linear relaxation bounds every remaining target.
        # A feasible maximizer with zero penalties attains that global bound.
        box_weights = reachable_lower.copy()
        remaining = budget - box_weights.sum()
        if remaining >= 0.0:
            for position in np.argsort(-expected_alpha, kind="stable"):
                change = min(
                    remaining, reachable_upper[position] - box_weights[position]
                )
                box_weights[position] += change
                remaining -= change
                if remaining <= 0.0:
                    break
            box_point = pack(box_weights)
            zero_penalties = loading @ (box_weights - reference) == 0.0 and (
                risk_aversion == 0.0
                or sum(
                    _risk_variances(
                        box_weights - reference, exposure, covariance, specific
                    )
                )
                == 0.0
            )
            if zero_penalties and feasible(box_point):
                initial, analytic_solution = box_point, True
    if analytic_solution:
        result = OptimizeResult(success=True, x=initial, nit=0)
    else:
        if ratio:
            barrier = (
                0.1
                if config.fully_invested and risk_limit is not None
                else config.solver_gtol
            )
            # A long-only budget already bounds each weight by the whole account.
            upper_bounds[:weight_count][reachable_upper[free_weights] >= budget] = (
                np.inf
            )
            keep_positive = not (config.fully_invested and exact_cash)
            if priced_ratio:
                score_domain = NonlinearConstraint(
                    lambda p: negative_score(p)[0],
                    -np.inf,
                    0.0,
                    jac=lambda p: sparse.csr_matrix(negative_score(p)[1][None, :]),
                    hess="2-point",
                    keep_feasible=keep_positive,
                )

                def positive_score(point):
                    weights, _, _ = unpack(point)
                    return score_domain.fun(point) < 0.0 and positive_income(
                        weights, actual_cost(weights)[0]
                    )
            else:
                score_domain = LinearConstraint(
                    sparse.csr_matrix(signal_gradient[None, :]),
                    -(expected_alpha @ fixed_part) / score_scale,
                    np.inf,
                    keep_feasible=keep_positive,
                )

                def positive_score(point):
                    return (score_domain.A @ point)[0] > score_domain.lb[0]

            if not feasible(initial, config.solver_gtol) or not positive_score(initial):
                # Any feasible positive numerator starts the ratio problem.
                # Only proving its absence requires optimizing the seed objective.
                seed = solve(
                    negative_score,
                    "2-point" if priced_ratio else zero_hessian,
                    initial,
                    barrier=barrier,
                    witness=positive_score,
                )
                initial = seed.x
                funding_iterations += seed.nit
            if not positive_score(initial):
                raise RiskInputsUnavailable(no_positive_score)
            # With no idle cash, strictly positive stock scores already imply
            # the rank domain. An all-cash start would leave the ratio undefined.
            if (
                not config.fully_invested
                or not exact_cash
                or priced_ratio
                or np.any((expected_alpha <= 0.0) & (reachable_upper > 0.0) & ~cash)
            ):
                constraints.append(score_domain)
            result = solve(
                signal_ratio,
                "2-point" if priced_ratio else ratio_hessian,
                initial,
                barrier=barrier,
            )
        else:
            result = solve(
                negative_score if ranks else objective,
                zero_hessian if ranks else objective_hessian,
                initial,
                barrier=config.solver_gtol
                if absolute and config.fully_invested
                else 0.1,
            )
    point = result.x.copy()
    moving = np.arange(weight_count)
    equations = [
        item
        for item in constraints
        if isinstance(item, LinearConstraint) and np.array_equal(item.lb, item.ub)
    ]
    raw_weights, _, _ = unpack(point)
    raw_fee = actual_cost(raw_weights)[0]
    if raw_weights[cash].sum() < raw_fee and np.any(cash[free_weights]):
        coefficients = gradient_point(cash_direction, fee_gradient=-1.0)
        target = -cash_direction @ fixed_part + (0.0 if fee_variable else fee_lower)
        equations.append(
            LinearConstraint(sparse.csr_matrix(coefficients[None, :]), target, target)
        )
    base = point.copy()
    if fee_variable:
        base[fee_position] = 0.0
    direction_fee = np.zeros(variable_count)
    if fee_variable:
        direction_fee[fee_position] = 1.0
    if equations:
        matrix = sparse.vstack([item.A for item in equations], format="csr")
        target = np.concatenate([item.lb for item in equations])
        if len(moving):
            reduced = matrix[:, moving]
            base[moving] += lsmr(
                reduced,
                target - matrix @ base,
                atol=np.finfo(float).eps,
                btol=np.finfo(float).eps,
            )[0]
            if fee_variable:
                direction_fee[moving] = lsmr(
                    reduced,
                    -matrix[:, fee_position].toarray().ravel(),
                    atol=np.finfo(float).eps,
                    btol=np.finfo(float).eps,
                )[0]
    if fee_variable:

        def fee_equation(value):
            weights, _, _ = unpack(base + direction_fee * value)
            return value - actual_cost(weights)[0]

        closed_fee = brentq(
            fee_equation,
            fee_lower,
            fee_upper,
            xtol=np.finfo(float).eps * fee_upper,
            rtol=4 * np.finfo(float).eps,
        )
        point = base + direction_fee * closed_fee
    else:
        point = base
    weights, _, solved_fee = unpack(point)
    weights = np.asarray(weights, dtype="float64")
    cost_ratio, _ = actual_cost(weights)
    if abs(solved_fee - cost_ratio) > config.solver_gtol:
        raise ValueError("optimizer_solution_invalid")
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
    objective_facts = _portfolio_objective(
        weights,
        objective_kind=objective_kind,
        benchmark=benchmark,
        expected_alpha=expected_alpha,
        loading=loading,
        cost_ratio=cost_ratio,
        risk_aversion=risk_aversion,
        horizon_scale=horizon_scale,
        exposure=exposure,
        covariance=covariance,
        specific=specific,
    )
    if ratio:
        score = (
            expected_alpha @ weights - (cost_ratio if net_return else 0.0)
        ) / score_scale
        if not positive_income(weights, cost_ratio):
            raise RiskInputsUnavailable(no_positive_score)
        variance = factor_variance + specific_variance
        null_candidate = np.any(
            (specific == 0) & (expected_alpha > 0) & (reachable_upper > 0)
        ) and not np.any((specific > 0) & (reachable_lower > 0))
        if null_candidate:
            # Test the risk-null face in the original account, including actual fees.
            # Dropping tiny positive-risk holdings and funding the surviving stock mix
            # supplies a witness only if the same account can hold it.
            mix = np.where((specific == 0) & ~cash, np.maximum(weights, 0.0), 0.0)
            if (
                mix.sum()
                and expected_alpha @ mix > 0
                and not sum(_risk_variances(mix, exposure, covariance, specific))
            ):
                mix /= mix.sum()

                def null_portfolio(principal):
                    return principal * mix + (budget - principal) * cash_direction

                idle = max(0.0, weights[cash].item() - cost_ratio)
                available = budget - idle
                if available >= actual_cost(null_portfolio(0.0))[0]:
                    principal = brentq(
                        lambda v: available - v - actual_cost(null_portfolio(v))[0],
                        0.0,
                        available,
                    )
                    witness = null_portfolio(principal)
                    if (
                        principal > 0
                        and positive_income(witness, actual_cost(witness)[0])
                        and (witness >= reachable_lower).all()
                        and (witness <= reachable_upper).all()
                        and feasible(pack(witness), config.solver_gtol)
                    ):
                        raise RiskInputsUnavailable(
                            "optimizer_net_ratio_zero_volatility"
                            if net_return
                            else "optimizer_rank_ratio_zero_volatility"
                        )
        if variance == 0 or (
            null_candidate
            and 1 - score / np.hypot(score, np.sqrt(variance))
            <= max(config.solver_gtol, complementarity_sum + projected_optimality)
        ):
            raise RiskInputsUnavailable(
                "optimizer_net_ratio_risk_unresolved"
                if net_return
                else "optimizer_rank_ratio_risk_unresolved"
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
        **objective_facts,
        solver_iterations=int(result.nit + funding_iterations),
        solver_variables=variable_count,
        solver_optimality=projected_optimality,
        solver_constraint_violation=0.0
        if analytic_solution
        else float(result.constr_violation),
        solver_complementarity=complementarity,
        solver_complementarity_sum=complementarity_sum,
        solver_complementarity_count=complementarity_count,
        solver_termination="analytic_optimum"
        if analytic_solution and not funding_iterations
        else "converged",
    )


def _minimize_in_percentage_points(
    fun, start, *, hess, bounds, constraints, callback, **options
):
    """Condition small net-return steps while certifying KKT in original NAV units."""
    unit = 0.01
    scaled = []
    for constraint in constraints:
        lb, ub = constraint.lb / unit, constraint.ub / unit
        if isinstance(constraint, LinearConstraint):
            row = LinearConstraint(
                constraint.A, lb, ub, keep_feasible=constraint.keep_feasible
            )
        else:
            row = NonlinearConstraint(
                lambda y, c=constraint: c.fun(y * unit) / unit,
                lb,
                ub,
                jac=lambda y, c=constraint: c.jac(y * unit),
                hess=constraint.hess,
                keep_feasible=constraint.keep_feasible,
            )
        scaled.append(row)

    def objective(y):
        loss, gradient = fun(y * unit)
        return loss / unit, gradient

    def convergence(y, state):
        original = OptimizeResult(state)
        original.constr = [value * unit for value in state.constr]
        original.constr_violation = state.constr_violation * unit
        return callback(y * unit, original)

    result = minimize(
        objective,
        start / unit,
        hess=(lambda y: hess(y * unit) * unit) if callable(hess) else hess,
        bounds=Bounds(bounds.lb / unit, bounds.ub / unit),
        constraints=scaled,
        callback=convergence,
        **options,
    )
    result.x *= unit
    result.constr_violation *= unit
    return result


def _portfolio_objective(
    weights,
    *,
    objective_kind=PortfolioObjective.CALIBRATED,
    benchmark,
    expected_alpha,
    loading,
    cost_ratio,
    risk_aversion,
    horizon_scale,
    exposure,
    covariance,
    specific,
):
    """Value a target or actual allocation in the original pre-fee NAV units."""
    active = weights if benchmark is None else weights - benchmark
    expected_return = float(expected_alpha @ active)
    if objective_kind == PortfolioObjective.NET_RETURN_TO_RISK:
        variance = horizon_scale * sum(
            _risk_variances(active, exposure, covariance, specific)
        )
        return {
            "expected_return": expected_return,
            "holding_variance": variance,
            "objective_value": float((expected_return - cost_ratio) / np.sqrt(variance))
            if variance
            else None,
        }
    if objective_kind != PortfolioObjective.CALIBRATED:
        value = expected_return
        if objective_kind == PortfolioObjective.RANK_TO_RISK:
            vol = np.sqrt(sum(_risk_variances(active, exposure, covariance, specific)))
            value = float(expected_return / vol) if vol else None
        return {"objective_value": value}
    robust_penalty = float(abs(loading @ active)) if loading is not None else 0.0
    holding_variance = (
        float(
            horizon_scale * sum(_risk_variances(active, exposure, covariance, specific))
        )
        if risk_aversion > 0.0
        else None
    )
    risk_penalty = (
        0.5 * risk_aversion * holding_variance if holding_variance is not None else 0.0
    )
    return {
        "expected_return": expected_return,
        "robust_penalty": robust_penalty,
        "holding_variance": holding_variance,
        "risk_penalty": risk_penalty,
        "objective_value": expected_return - robust_penalty - risk_penalty - cost_ratio,
    }


def _portfolio_incomplete(
    data,
    industry_columns,
    style_columns,
    config,
    objective_needs_covariance,
    signal_column,
):
    columns = (*industry_columns, *style_columns)
    risk_missing = data.specific_variance.isna() | data.loc[:, columns].isna().any(
        axis=1
    )
    alpha_missing = ~np.isfinite(data[signal_column])
    if not isinstance(config, AbsoluteRiskConfig):
        return risk_missing, alpha_missing, risk_missing | alpha_missing
    incomplete = pd.Series(False, index=data.index)
    if config.total_volatility_limit is not None or objective_needs_covariance:
        incomplete |= risk_missing
    for names, limit in (
        (industry_columns, config.industry_weight_limit),
        (style_columns, config.style_exposure_limit),
    ):
        if limit is not None:
            incomplete |= data.loc[:, names].isna().any(axis=1)
    return risk_missing, alpha_missing, incomplete


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


def _exposure_roundoff(weights, exposure):
    return 8 * len(weights) * np.finfo(float).eps * (abs(weights) @ abs(exposure))


def _risk_variance_roundoff(
    active, exposure, covariance, specific, coefficient_roundoff
):
    nonzero = active != 0.0
    active, exposure, specific = active[nonzero], exposure[nonzero], specific[nonzero]
    coefficient_roundoff = coefficient_roundoff[nonzero]
    factor = abs(active @ exposure)
    error = _exposure_roundoff(active, exposure) + coefficient_roundoff @ abs(exposure)
    covariance = abs(covariance)
    return float(
        8 * len(factor) * np.finfo(float).eps * (factor @ covariance @ factor)
        + 2 * error @ covariance @ factor
        + error @ covariance @ error
        + 8 * len(active) * np.finfo(float).eps * np.sum(specific * active**2)
        + np.sum(
            specific
            * (2 * abs(active) * coefficient_roundoff + coefficient_roundoff**2)
        )
    )


def _normalized_risk_excess(values, limits, *, strict=False, roundoff=0.0):
    tolerance = (
        np.maximum(
            roundoff, 8 * np.finfo(float).eps * np.maximum(abs(values), abs(limits))
        )
        if strict
        else 1e-8
    )
    return np.maximum(0.0, values - limits - tolerance) / np.maximum(limits, 1e-8)


def _portfolio_risk_limits(
    weights,
    benchmark,
    exposure,
    specific,
    covariance,
    cash,
    columns,
    config,
    *,
    strict=False,
    net_bounds=None,
):
    """Report post-fee risk and rank only excess beyond the configured bands."""
    if isinstance(config, AbsoluteRiskConfig):
        return _absolute_portfolio_risk_limits(
            weights,
            exposure,
            specific,
            covariance,
            cash,
            columns,
            config,
            strict=strict,
        )
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
    exposure_roundoff = np.zeros(len(columns))
    if strict and stock_positive:
        normalized = weights[~cash] / weights[~cash].sum()
        reference = benchmark[~cash] / benchmark[~cash].sum()
        exposure_roundoff = _exposure_roundoff(
            abs(normalized) + abs(reference),
            np.where(np.isfinite(exposure[~cash]), exposure[~cash], 0.0),
        )
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
        values = (
            np.abs(weights[~cash] - benchmark[~cash])
            if name == "name_active_weight"
            else np.abs(factor_active[industry])
            if name == "industry_active_exposure"
            else np.abs(factor_active[~industry])
            if name == "style_active_exposure"
            else np.array([value])
        )
        roundoff = (
            exposure_roundoff[industry]
            if name == "industry_active_exposure"
            else exposure_roundoff[~industry]
            if name == "style_active_exposure"
            else 8
            * np.finfo(float).eps
            * np.maximum(abs(weights[~cash]), abs(benchmark[~cash]))
            if name == "name_active_weight"
            else _risk_variance_roundoff(
                active,
                exposure,
                covariance,
                specific,
                8 * np.finfo(float).eps * np.maximum(abs(weights), abs(benchmark)),
            )
            if strict and name == "tracking_error"
            else 0.0
        )
        differences = _normalized_risk_excess(
            values, bound, strict=strict, roundoff=roundoff
        )
        excess.extend(differences)
        if np.any(differences > 0.0):
            breached.append(name)
    if net_bounds is not None:
        lower, upper = np.asarray(net_bounds).T
        fixed = lower == upper
        below = _normalized_risk_excess(lower[fixed], weights[fixed], strict=strict)
        above = _normalized_risk_excess(weights[fixed], upper[fixed], strict=strict)
        excess.extend(np.r_[below, above])
        if np.any(below > 0.0) or np.any(above > 0.0):
            breached.append("fixed_name_weight")
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


def _absolute_portfolio_risk_limits(
    weights, exposure, specific, covariance, cash, columns, config, *, strict=False
):
    """Evaluate the absolute optimizer's limits on actual post-fee holdings."""
    held = weights != 0.0
    industry = np.array([name.startswith("industry_") for name in columns])
    complete = (
        covariance is not None
        and np.isfinite(exposure[held]).all()
        and np.isfinite(specific[held]).all()
    )
    if not weights[~cash].any():
        # Cash has known zero exposure; covariance is irrelevant without stocks.
        variance = 0.0
    else:
        variance = (
            sum(_risk_variances(weights, exposure, covariance, specific))
            if complete
            else None
        )
    diagnostics = {}
    values = {
        "total_volatility": (
            np.sqrt(variance) if variance is not None else None,
            config.total_volatility_limit,
        ),
        "name_weight": (np.max(weights[~cash], initial=0.0), config.name_weight_limit),
    }
    exposure_values, exposure_roundoff = {}, {}
    for label, mask, limit in (
        ("industry_exposure", industry, config.industry_weight_limit),
        ("style_exposure", ~industry, config.style_exposure_limit),
    ):
        selected = exposure[held][:, mask]
        value = None
        if np.isfinite(selected).all():
            components = np.abs(weights[held] @ selected)
            value = np.max(components, initial=0.0)
            if strict:
                exposure_values[label] = components
                exposure_roundoff[label] = _exposure_roundoff(weights[held], selected)
        values[label] = (value, limit)
    unavailable, breached, excess = [], [], []
    for label, (value, limit) in values.items():
        diagnostics[f"realized_{label}_status"] = (
            "UNDEFINED" if value is None else "DEFINED"
        )
        if value is not None:
            diagnostics[f"realized_{label}"] = float(value)
        if limit is None:
            continue
        if value is None:
            unavailable.append(label)
            continue
        measured = (
            exposure_values.get(label, np.array([value]))
            if strict
            else np.array([value])
        )
        roundoff = exposure_roundoff.get(label, 0.0)
        if strict and label == "total_volatility":
            measured, limit = np.array([variance]), limit**2
            roundoff = (
                _risk_variance_roundoff(
                    weights,
                    exposure,
                    covariance,
                    specific,
                    8 * np.finfo(float).eps * abs(weights),
                )
                if complete
                else 0.0
            )
        differences = _normalized_risk_excess(
            measured, limit, strict=strict, roundoff=roundoff
        )
        excess.extend(differences)
        if np.any(differences > 0.0):
            breached.append(label)
    diagnostics.update(
        realized_constraints_status="UNDEFINED"
        if unavailable
        else "BREACHED"
        if breached
        else "SATISFIED",
        realized_constraints_reason="absolute_risk_unavailable"
        if unavailable
        else ",".join(breached)
        if breached
        else None,
    )
    return diagnostics, (
        (np.inf, np.inf)
        if unavailable
        else (max(excess, default=0.0), float(np.square(excess).sum()))
    )


def _net_name_bounds(
    data: pd.DataFrame,
    *,
    benchmark: np.ndarray | None,
    financial_column: str | None,
    cash_column: str | None,
    eligibility_column: str | None,
    incomplete: np.ndarray,
    config: OptimizerConfig | AbsoluteRiskConfig,
) -> tuple[tuple[float, float], ...]:
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
    return tuple(bounds)


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
    net_bounds = _net_name_bounds(
        data,
        benchmark=benchmark,
        financial_column=financial_column,
        cash_column=cash_column,
        eligibility_column=eligibility_column,
        incomplete=incomplete,
        config=config,
    )
    bounds = list(net_bounds)
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
    if (
        isinstance(config, AbsoluteRiskConfig)
        and type(config.fully_invested) is not bool
    ):
        raise ValueError("optimizer_config_invalid")
    if any(
        type(value) is not float or not np.isfinite(value) or value <= 0.0
        for value in (config.solver_gtol, config.solver_xtol, config.solver_barrier_tol)
    ):
        raise ValueError("optimizer_config_invalid")
    if isinstance(config, AbsoluteRiskConfig):
        limits = (
            config.name_weight_limit,
            config.total_volatility_limit,
            config.target_turnover_limit,
            config.industry_weight_limit,
            config.style_exposure_limit,
        )
        values = tuple(value for value in limits if value is not None)
    else:
        values = (
            config.active_name_limit,
            config.tracking_error_limit,
            config.target_turnover_limit,
            config.industry_active_limit,
            config.style_active_limit,
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
    if config.solver_maxiter is not None and (
        type(config.solver_maxiter) is not int or config.solver_maxiter <= 0
    ):
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
